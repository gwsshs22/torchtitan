from enum import Enum, auto
import io
import json
from pathlib import Path
from typing import Any
import os

import numpy as np
import torch
import torch.distributed as dist

class CheckpointLoadAction(Enum):
    NONE = auto()
    LOCAL = auto()
    SEND = auto()
    RECV = auto()

class SnapshotGroup:

    def __init__(self, process_group: dist.ProcessGroup | None):
        self._global_rank = dist.get_rank() if dist.is_initialized() else 0
        self._global_world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.process_group = process_group
        if self.process_group is not None:
            self._group_rank = dist.get_rank(self.process_group)
            self._group_size = dist.get_world_size(self.process_group)
        else:
            self._group_rank = self._global_rank
            self._group_size = self._global_world_size

        assert self._group_size > 1 and self._group_size % 2 == 0

        # Peer rank within the group (or global if group is None)
        self._peer_group_rank = (self._group_rank + self._group_size // 2) % self._group_size

        if self.process_group is not None:
            global_ranks = dist.get_process_group_ranks(self.process_group)
            self._peer_global_rank = global_ranks[self._peer_group_rank]
        else:
            self._peer_global_rank = self._peer_group_rank

        # Initialize Gloo process group for object exchange
        # Gloo backend is required for send/recv_object_list
        self._gloo_pg = None
        self._p2p_pg = None
        self._peer_p2p_rank = None

        if dist.is_initialized():
            self._gloo_pg = dist.new_group(backend="gloo")

            # Create 2-rank P2P subgroups (one per peer pair)
            # All ranks must call new_group for every pair (it's a global collective)
            my_pair = tuple(sorted([self._global_rank, self._peer_global_rank]))
            all_pairs = [None] * self._global_world_size
            dist.all_gather_object(all_pairs, my_pair, group=self._gloo_pg)
            unique_pairs = sorted(set(all_pairs))

            for pair in unique_pairs:
                pg = dist.new_group(
                    ranks=list(pair),
                    device_id=torch.device("cuda", torch.cuda.current_device()),
                )
                if self._global_rank in pair:
                    self._p2p_pg = pg

            self._peer_p2p_rank = 1 - dist.get_rank(self._p2p_pg)

    @property
    def global_rank(self) -> int:
        return self._global_rank

    @property
    def group_rank(self) -> int:
        return self._group_rank
    
    @property
    def group_size(self) -> int:
        return self._group_size

    @property
    def peer_group_rank(self) -> int:
        return self._peer_group_rank

    @property
    def peer_global_rank(self) -> int:
        return self._peer_global_rank

    def warmup_p2p_pg(self, input_tensor, output_tensor):
        self.sendrecv_tensor(input_tensor, output_tensor)

    def broadcast_strategy(self, strategy: dict) -> dict | None:
        strategy_list = [strategy]
        dist.broadcast_object_list(
            strategy_list, group_src=0, group=self.process_group
        )
        return strategy_list[0]

    def exchange_object(self, obj: Any) -> Any:
        if self._gloo_pg is None:
            return obj

        recv_objs = [None]
        send_objs = [obj]

        # Determine order to avoid deadlock
        if self._global_rank < self._peer_global_rank:
            # Send then Recv
            dist.send_object_list(
                send_objs, group=self._gloo_pg, group_dst=self._peer_global_rank
            )
            dist.recv_object_list(
                recv_objs, group=self._gloo_pg, group_src=self._peer_global_rank
            )
        else:
            # Recv then Send
            dist.recv_object_list(
                recv_objs, group=self._gloo_pg, group_src=self._peer_global_rank
            )
            dist.send_object_list(
                send_objs, group=self._gloo_pg, group_dst=self._peer_global_rank
            )

        return recv_objs[0]

    def sendrecv_tensor(self, input_tensor, output_tensor):
        ops = [
            dist.P2POp(dist.isend, input_tensor, group_peer=self._peer_p2p_rank, group=self._p2p_pg),
            dist.P2POp(dist.irecv, output_tensor, group_peer=self._peer_p2p_rank, group=self._p2p_pg),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    def send_checkpoint(self, checkpoint_path):
        file_size = os.path.getsize(checkpoint_path)

        size_tensor = torch.tensor([file_size], dtype=torch.long)
        dist.send(
            size_tensor,
            group_dst=self._peer_global_rank,
            group=self._gloo_pg
        )

        np_data = np.memmap(checkpoint_path, dtype=np.uint8, mode='r')
        data_tensor = torch.from_numpy(np_data)
        dist.send(
            data_tensor,
            group_dst=self._peer_global_rank,
            group=self._gloo_pg
        )

    def recv_checkpoint(self, tmp_path):
        size_tensor = torch.tensor([0], dtype=torch.long)
        dist.recv(size_tensor, group=self._gloo_pg, group_src=self._peer_global_rank)
        file_size = size_tensor.item()

        with open(tmp_path, 'wb') as f:
            f.truncate(file_size)

        np_mmap = np.memmap(tmp_path, dtype=np.uint8, mode='r+', shape=(file_size,))
        data_tensor = torch.from_numpy(np_mmap)
        dist.recv(data_tensor, group=self._gloo_pg, group_src=self._peer_global_rank)
        np_mmap.flush()
        del data_tensor
        del np_mmap
 
        checkpoint = torch.load(tmp_path, map_location='cpu', weights_only=False)
        os.unlink(tmp_path)
        return checkpoint

    def load_gaps(self, comm_gaps_folder: str) -> dict[int, float] | None:
        # Only rank 0 loads the gaps
        if self._group_rank != 0:
            return None

        # Find the gap times file for this FSDP group
        # FSDP rank 0 writes the file, so we need to find which global rank that is
        if self.process_group is not None:
            fsdp_global_ranks = dist.get_process_group_ranks(self.process_group)
            writer_global_rank = fsdp_global_ranks[0]  # FSDP rank 0 writes the gaps
        else:
            writer_global_rank = 0

        gap_file = Path(comm_gaps_folder) / f"comm_gaps_rank_{writer_global_rank}.json"

        if not gap_file.exists():
            raise FileNotFoundError(
                f"[Gemini] Gap times file not found: {gap_file}. "
                f"Run profiling first with gemini_profile_comm_gaps=true."
            )

        with open(gap_file) as f:
            data = json.load(f)
            gap_times = {
                int(k): v for k, v in data.get("gap_times_ms", {}).items()
            }

        return gap_times

    def get_checkpoint_load_action(self, has_checkpoint):
        # Exchange has_checkpoint status with peer
        peer_has_checkpoint = self.exchange_object(has_checkpoint)

        # Determine action based on (has_checkpoint, peer_has_checkpoint)
        if has_checkpoint and peer_has_checkpoint:
            return CheckpointLoadAction.LOCAL
        elif has_checkpoint and not peer_has_checkpoint:
            return CheckpointLoadAction.SEND
        elif not has_checkpoint and peer_has_checkpoint:
            return CheckpointLoadAction.RECV
        else:  # not has_checkpoint and not peer_has_checkpoint
            return CheckpointLoadAction.NONE

    def validate_steps(self, step):
        all_steps = [None] * self._global_world_size
        dist.all_gather_object(all_steps, step, group=self._gloo_pg)

        # Check if all steps are the same
        first_step = all_steps[0]
        if not all(s == first_step for s in all_steps):
            raise RuntimeError(
                f"Step mismatch detected across ranks. Steps: {all_steps}. "
                f"All ranks must be at the same training step."
            )