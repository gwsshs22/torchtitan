import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor

from torchtitan.components.gemini.in_mem_state import InMemState
from torchtitan.components.gemini.snapshot_container import SnapshotContainer
from torchtitan.components.gemini.snapshot_group import (
    SnapshotGroup,
    CheckpointLoadAction
)
from torchtitan.components.gemini.snapshot_strategy import get_snapshot_strategy
from torchtitan.components.gemini.utils import InMemStateType
from torchtitan.tools.logging import logger

# Leto integration for checkpoint timing reporting
try:
    from leto.launch.worker_controller_client import (
        report_duration,
        DURATION_CHECKPOINT_STAGING,
    )
    _LETO_AVAILABLE = True
except ImportError:
    _LETO_AVAILABLE = False

# InMemState types
LOCAL = 0
REMOTE = 1

# Double-buffer indices
CURR = 0
PREV = 1

class SnapshotExecutor:

    def __init__(
        self,
        enable: bool,
        states: dict,
        model_wrapper,
        optimizers,
        comm_gaps_folder: str,
        block_size: int = 4 * 1024 * 1024,  # 4M elements default
        fsdp_process_group: dist.ProcessGroup | None = None,
        mem_fs_folder: str = "",
        # Strategy computation parameters
        bandwidth_gbps: float = 100.0,  # Network bandwidth in Gbps
        gap_threshold_ms: float = 3.0,  # Minimum gap to consider (ms)
        alpha: float = 0.8,  # Utilization factor
        max_blocks_per_gap: int = 1024 * 1024,  # Max blocks to send per gap
    ):
        self.enable = enable
        if not self.enable:
            return
        assert mem_fs_folder != ""
        self.optimizers = optimizers
        self.states = states
        self.block_size = block_size
        self.comm_gaps_folder = comm_gaps_folder
        self.mem_fs_folder = mem_fs_folder
        os.makedirs(self.mem_fs_folder, exist_ok=True)

        # Strategy parameters
        self._bandwidth_gbps = bandwidth_gbps
        self._gap_threshold_ms = gap_threshold_ms
        self._alpha = alpha
        self._max_blocks_per_gap = max_blocks_per_gap

        self._fsdp_pg = fsdp_process_group
        self.snapshot_group = SnapshotGroup(self._fsdp_pg)

        #TODO: create a new group for p2p based on self._fdsp_pg
        self.p2p_pg = None

        self.local_checkpoint_path = f"{self.mem_fs_folder}/rank_{self.snapshot_group._global_rank}_local.pt"
        self.remote_checkpoint_path = f"{self.mem_fs_folder}/rank_{self.snapshot_group._global_rank}_remote.pt"
        self.tmp_checkpoint_path = f"{self.mem_fs_folder}/rank_{self.snapshot_group._global_rank}_tmp.pt"
        self.container_log_path = f"{self.mem_fs_folder}/rank_{self.snapshot_group._global_rank}_log.txt"

        self.snapshot_container = SnapshotContainer(
            self.local_checkpoint_path,
            self.remote_checkpoint_path,
            self.container_log_path,
            self.snapshot_group._global_rank
        )

        self.has_checkpoint = os.path.exists(self.local_checkpoint_path) and os.path.exists(
            self.remote_checkpoint_path
        )

        sample_tensor = next(iter(model_wrapper.state_dict().values()))
        self._dtype_size = sample_tensor.element_size()

        # [LOCAL/REMOTE][CURR/PREV]
        self.in_mem_states: list[list[InMemState]] = [
            [
                InMemState(
                    state_id,
                    model_wrapper,
                    optimizers,
                    states,
                    state_type,
                    self.snapshot_container
                ) for state_id in range(2)
            ] for state_type in [InMemStateType.LOCAL, InMemStateType.REMOTE]
        ]

        self._curr_version = CURR
        self._cur_block_id = 0
        self._comm_gap_id = 0

        self._snapshot_strategy: dict[int, int] = {}
        self._have_strategy = False

        self._gpu_buffers = [
            torch.zeros(block_size, dtype=sample_tensor.dtype, device="cuda"),
            torch.zeros(block_size, dtype=sample_tensor.dtype, device="cuda"),
        ]
        self._gpu_buffer_id = 0

        self._local_copy_stream = torch.cuda.Stream()
        self._copy_stream = torch.cuda.Stream()
        self._p2p_stream = torch.cuda.Stream()

        # Distributed setup - compute ranks within FSDP group
        self._peer_global_rank = self.snapshot_group.peer_global_rank

        # Thread pool for async snapshot operations (Gloo exchange)
        self._snapshot_thread_pool = ThreadPoolExecutor(max_workers=1)
        self._snapshot_future: Future | None = None

        # Track if we're in a snapshot step
        self._is_snapshot_step = False

    @property
    def local_curr(self) -> InMemState:
        return self.in_mem_states[LOCAL][self._curr_version]

    @property
    def local_prev(self) -> InMemState:
        return self.in_mem_states[LOCAL][1 - self._curr_version]

    @property
    def remote_curr(self) -> InMemState:
        return self.in_mem_states[REMOTE][self._curr_version]

    @property
    def remote_prev(self) -> InMemState:
        return self.in_mem_states[REMOTE][1 - self._curr_version]

    def _swap_buffers(self):
        self._curr_version = 1 - self._curr_version

    def _reset_for_new_step(self):
        self._cur_block_id = 0
        self._comm_gap_id = 0
        self._swap_buffers()

    def load(self, step) -> bool:
        if not self.enable:
            return False
        loaded = True

        self.local_curr.init_cpu_tensors()
        load_action = self.snapshot_group.get_checkpoint_load_action(self.has_checkpoint)
        if load_action == CheckpointLoadAction.NONE:
            loaded = False
        elif load_action == CheckpointLoadAction.LOCAL:
            self.local_curr.load_state_dict(
                torch.load(self.local_checkpoint_path, map_location="cpu", weights_only=False)
            )
        elif load_action == CheckpointLoadAction.SEND:
            self.local_curr.load_state_dict(
                torch.load(self.local_checkpoint_path, map_location="cpu", weights_only=False)
            )
            self.snapshot_group.send_checkpoint(self.remote_checkpoint_path)
        elif load_action == CheckpointLoadAction.RECV:
            self.local_curr.load_state_dict(
                self.snapshot_group.recv_checkpoint(self.tmp_checkpoint_path)
            )
        else:
            raise ValueError(f"Unknown load action: {load_action}")
        if loaded:
            loaded_step = self.states["train_state"].step
            logger.info(f"Loaded checkpoint at step {loaded_step}, load_action={load_action}")
            self.snapshot_group.validate_steps(loaded_step)

        self.local_prev.init_cpu_tensors()
        self.remote_curr.init_cpu_tensors()
        self.remote_prev.init_cpu_tensors()

        self._gpu_blocks = self.remote_curr.compute_tensor_blocks(self.block_size, return_gpu_blocks=True)
        self.remote_prev.compute_tensor_blocks(self.block_size)

        self._total_blocks = len(self._gpu_blocks)
        self._block_sizes = [t.numel() for t in self._gpu_blocks]
        self._load_gaps_and_compute_strategy()
        self._init_sendrecv() # Warmup
        return loaded

    def _load_gaps_and_compute_strategy(self):
        strategy = None

        # FSDP rank 0 loads gaps and computes strategy
        gap_times = self.snapshot_group.load_gaps(self.comm_gaps_folder)
        if gap_times is not None:
            # Compute strategy using actual block sizes
            strategy = get_snapshot_strategy(
                gap_times,
                self._block_sizes,
                self._dtype_size,
                self._bandwidth_gbps,
                self._gap_threshold_ms,
                self._alpha,
                self._max_blocks_per_gap,
            )

            logger.info(
                f"[Gemini] Rank {self.snapshot_group.global_rank} (FSDP rank 0): Computed strategy with "
                f"{len(strategy)} gaps, {len(self._block_sizes)} blocks"
            )
            logger.info(f"[Gemini] Strategy: {strategy}")

        strategy = self.snapshot_group.broadcast_strategy(strategy)
        if strategy is not None:
            self._snapshot_strategy = strategy
            self._have_strategy = True

    def _init_sendrecv(self):
        with torch.cuda.stream(self._p2p_stream):
            self.snapshot_group.sendrecv_tensor(self._gpu_buffers[0], self._gpu_buffers[1])

    def _snapshot_background(self, cpu_metadata_state_dict):
        """Background thread: GPU→CPU copy + Gloo metadata exchange."""
        local_copy_event = torch.cuda.Event()
        with torch.cuda.stream(self._local_copy_stream):
            self.local_curr.snapshot_gpu_state()
            local_copy_event.record()

        remote_cpu_metadata_state_dict = self.snapshot_group.exchange_object(cpu_metadata_state_dict)
        self.remote_curr.set_cpu_metadata_state_dict(remote_cpu_metadata_state_dict)
        return local_copy_event

    def snapshot(self, curr_step):
        if not self.enable:
            return

        self._is_snapshot_step = True
        self._snapshot_step = curr_step
        self._staging_start_time = time.time()  # Track staging start time
        self._reset_for_new_step()

        cpu_metadata_state_dict = self.local_curr.snapshot_cpu_metadata_state()

        self._snapshot_future = self._snapshot_thread_pool.submit(
            self._snapshot_background, cpu_metadata_state_dict
        )

    def begin_collective(self):
        torch.cuda.current_stream().wait_stream(self._p2p_stream)

    def end_collective(self, async_op: bool, handle):
        if not self.enable or not self._is_snapshot_step:
            return

        if not self._have_strategy:
            return

        self._comm_gap_id += 1

        if self._comm_gap_id in self._snapshot_strategy:
            num_blocks = self._snapshot_strategy[self._comm_gap_id]

            comm_stream = torch.cuda.current_stream()
            with torch.cuda.stream(self._p2p_stream):
                if async_op:
                    handle.wait()
                else:
                    self._p2p_stream.wait_stream(comm_stream)

                self._snapshot_blocks(num_blocks)

    def _snapshot_blocks(self, num_blocks: int):
        if num_blocks == -1:
            num_blocks = self._total_blocks - self._cur_block_id

        for _ in range(num_blocks):
            if self._cur_block_id >= self._total_blocks:
                break
            self._snapshot_block()

    def _snapshot_block(self):
        if self._cur_block_id >= self._total_blocks:
            return

        block_id = self._cur_block_id
        gpu_block = self._gpu_blocks[block_id]
        block_size = self._block_sizes[block_id]

        output_tensor = self._gpu_buffers[self._gpu_buffer_id][:block_size]
        self._gpu_buffer_id = 1 - self._gpu_buffer_id
        self.snapshot_group.sendrecv_tensor(gpu_block, output_tensor)

        with torch.cuda.stream(self._copy_stream):
            cpu_block = self.remote_curr.get_block(block_id)
            self._copy_stream.wait_stream(self._p2p_stream)
            cpu_block.copy_(output_tensor, non_blocking=True)

        self._cur_block_id += 1

    def _sendrecv_tensor(self, input_tensor, output_tensor):
        ops = [
            dist.P2POp(dist.isend, input_tensor, self._peer_global_rank),
            dist.P2POp(dist.irecv, output_tensor, self._peer_global_rank),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    def maybe_wait_for_staging(self):
        if not self.enable:
            return

        if self._is_snapshot_step:
            assert self._cur_block_id == self._total_blocks

            # Wait for background thread (Gloo exchange + local copy launch)
            local_copy_event = self._snapshot_future.result()

            # Sync streams to ensure all copies are done
            curr_stream = torch.cuda.current_stream()
            curr_stream.wait_stream(self._copy_stream)
            curr_stream.wait_stream(self._p2p_stream)
            curr_stream.wait_event(local_copy_event)
            curr_stream.synchronize()

            self.snapshot_container.commit(self._curr_version, self._snapshot_step)

            # Report staging duration to leto
            if _LETO_AVAILABLE:
                staging_duration = time.time() - self._staging_start_time
                report_duration(DURATION_CHECKPOINT_STAGING, staging_duration, self._snapshot_step)

            # Mark snapshot step as complete
            self._is_snapshot_step = False

    def close(self):
        if not self.enable:
            return

        self._snapshot_thread_pool.shutdown(wait=True)
        self.snapshot_container.close()
