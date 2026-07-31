from enum import Enum, auto
import json
from pathlib import Path
from typing import Any
import os
import socket

import numpy as np
import torch
import torch.distributed as dist

from torchtitan.components.snapshot import partner
from torchtitan.tools.logging import logger

class CheckpointLoadAction(Enum):
    NONE = auto()
    LOCAL = auto()
    SEND = auto()
    RECV = auto()


def elect_consistent_step(all_info: list, world_size: int):
    """Pure consistent-step election over the gathered per-rank
    ``(available_steps, peer_global_rank)`` tuples (no dist calls, so the
    decision table is unit-testable).

    For each peer pair the recoverable steps are the union of both members'
    sets; the elected step is the latest step recoverable by ALL pairs.

    Availability fallback (2026-07-30 gptoss_pp2 gemini_ftft forensics):
    when at least one pair's recoverable set is EMPTY while others are not,
    both replicas of that pair are gone — the signature of a fatal mem_fs
    wipe of a host that contained BOTH members of a pair. Since the shared
    partner rule (snapshot.partner) that pairing is now cross-host wherever
    the candidate pool allows it, so this shape means the layout itself
    could not offer cross-host redundancy (SnapshotGroup logs an ERROR at
    init in that case). No election can serve those ranks, so instead of
    raising (which crash-loops every restart attempt) fall back to a
    world-consistent fresh start. Pairs that are all non-empty but disjoint
    still raise: that shape is not producible by any kill instant of the
    snapshot cycle and indicates real inconsistency.

    Returns:
        (target_step, pair_recoverable_sets); target_step is None when there
        is nothing (consistently) loadable.
    """
    seen_pairs = set()
    pair_recoverable_sets = []
    for rank in range(world_size):
        steps, peer = all_info[rank]
        pair = (min(rank, peer), max(rank, peer))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            steps_a = all_info[pair[0]][0] or set()
            steps_b = all_info[pair[1]][0] or set()
            pair_recoverable_sets.append(steps_a | steps_b)

    if not pair_recoverable_sets:
        return None, pair_recoverable_sets

    consistent_steps = set.intersection(*pair_recoverable_sets)

    if not consistent_steps:
        # If no rank has any steps, that's fine — no checkpoint to load.
        all_steps = set().union(*pair_recoverable_sets)
        if not all_steps:
            return None, pair_recoverable_sets
        if any(not s for s in pair_recoverable_sets):
            # Wiped-pair shape (e.g. [set(), set(), {55}, {55}]): both
            # replicas of some pair destroyed. Degrade to a fresh start
            # instead of crash-looping the recovery.
            logger.error(
                f"[Gemini] Some FSDP pairs have NO recoverable checkpoint "
                f"(both replicas lost — pair members likely co-located on a "
                f"wiped host). Falling back to a world-consistent FRESH "
                f"START; surviving dumps will be discarded. Per-pair "
                f"recoverable steps: {pair_recoverable_sets}"
            )
            return None, pair_recoverable_sets
        raise RuntimeError(
            f"No consistent checkpoint step found across all FSDP pairs, "
            f"but some ranks have steps. Per-pair recoverable steps: {pair_recoverable_sets}"
        )

    return max(consistent_steps), pair_recoverable_sets


def assign_faulty_roles(
    election_action: CheckpointLoadAction,
    target_step,
    my_steps: set,
    peer_steps: set,
    self_faulty: bool,
    pair_faulty: bool,
    global_rank: int,
    peer_global_rank: int,
) -> CheckpointLoadAction:
    """Faulty-rank-driven role assignment (plan §3.7 gemini retrofit, D3/D5).

    The consistent-step election stays the STEP-chooser; roles come from the
    faulty set: faulty rank -> RECV (its own dump is never trusted, transient
    included), its pair -> SEND, everyone else -> LOCAL. The election's
    action is kept as a cross-check: a disagreement is logged as an error and
    the faulty-driven role wins — unless the faulty-driven role is
    unservable (RECV with a peer that lacks the step / SEND without the step
    locally), in which case the election's action is kept, loudly.

    Pure function (no dist) so the mapping is unit-testable.
    """
    if target_step is None or election_action == CheckpointLoadAction.NONE:
        return election_action  # nothing to load; roles are moot
    if self_faulty and pair_faulty:
        # Both members of the pair are faulty (e.g. single-host fatal):
        # nobody can serve anybody — keep the election's verdict (its steps,
        # if any, came from surviving metadata).
        logger.error(
            f"[Gemini] rank {global_rank} and its pair {peer_global_rank} "
            f"are BOTH faulty; peer-fetch impossible — keeping election "
            f"action {election_action.name} for step {target_step}"
        )
        return election_action
    if self_faulty:
        role = CheckpointLoadAction.RECV
    elif pair_faulty:
        role = CheckpointLoadAction.SEND
    else:
        role = CheckpointLoadAction.LOCAL
    if role != election_action:
        logger.error(
            f"[Gemini] faulty-rank-driven role {role.name} disagrees with "
            f"the consistent-step election ({election_action.name}) at step "
            f"{target_step} (rank={global_rank}, peer={peer_global_rank}, "
            f"self_faulty={self_faulty}, pair_faulty={pair_faulty}) — "
            f"preferring the faulty-driven role"
        )
    if role == CheckpointLoadAction.RECV and target_step not in peer_steps:
        logger.error(
            f"[Gemini] faulty rank {global_rank} must RECV step "
            f"{target_step} but pair {peer_global_rank} does not hold it "
            f"(peer_steps={sorted(peer_steps)}); falling back to election "
            f"action {election_action.name}"
        )
        return election_action
    if role == CheckpointLoadAction.SEND and target_step not in my_steps:
        logger.error(
            f"[Gemini] rank {global_rank} must SEND step {target_step} to "
            f"faulty pair {peer_global_rank} but does not hold it "
            f"(my_steps={sorted(my_steps)}); falling back to election "
            f"action {election_action.name}"
        )
        return election_action
    return role

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

        # Candidate pool = this rank's FSDP process group (global ranks in
        # group-rank order), i.e. the ranks that own the same model chunk.
        if self.process_group is not None:
            self._pool_ranks = tuple(
                int(r) for r in dist.get_process_group_ranks(self.process_group)
            )
        else:
            self._pool_ranks = tuple(range(self._global_world_size))

        # Fallback pairing for the not-initialized case (no collective, so no
        # host information): the legacy `+ size//2` offset, which is what
        # partner.plan_pairing also yields when nothing better exists.
        self._partner_plan = None
        self._peer_group_rank = (
            self._group_rank + self._group_size // 2
        ) % self._group_size
        self._peer_global_rank = self._pool_ranks[self._peer_group_rank]

        # Initialize Gloo process group for object exchange
        # Gloo backend is required for send/recv_object_list
        self._gloo_pg = None
        self._p2p_pg = None
        self._peer_p2p_rank = None

        if dist.is_initialized():
            self._gloo_pg = dist.new_group(backend="gloo")

            # ONE init-time all_gather carries (a) this rank's pool
            # membership and (b) its hostname. From it every rank rebuilds
            # EVERY pool's pairing with the shared rule
            # (snapshot.partner.plan_pairing: mutual, same-stage,
            # cross-host-maximizing) — which is required anyway because the
            # 2-rank P2P new_group calls are world collectives. The peer is
            # therefore no longer the hardcoded `+ size//2` offset that made
            # both replicas co-located on FSDP meshes that don't span hosts
            # (2026-07-30 gptoss_pp2 forensics).
            all_info = [None] * self._global_world_size
            dist.all_gather_object(
                all_info,
                (self._pool_ranks, socket.gethostname()),
                group=self._gloo_pg,
            )
            host_by_rank = [host for _pool, host in all_info]
            plans = partner.plan_pools(
                (pool for pool, _host in all_info), host_by_rank
            )
            self._partner_plan = plans[self._pool_ranks]
            self._peer_global_rank = self._partner_plan.partner_of(
                self._global_rank
            )
            self._peer_group_rank = self._partner_plan.partner_position_of(
                self._global_rank
            )
            partner.log_pairing(
                self._partner_plan,
                self._global_rank,
                method="gemini",
                world_hosts=host_by_rank,
                log=logger,
            )
            unique_pairs = partner.all_pairs(plans)
            logger.info(
                f"[SnapshotGroup] rank={self._global_rank} all_gather_object done; "
                f"peer={self._peer_global_rank} (pool={list(self._pool_ranks)}, "
                f"offset={self._partner_plan.offset}); "
                f"{len(unique_pairs)} pair PGs to create"
            )

            for i, pair in enumerate(unique_pairs):
                pg = dist.new_group(
                    ranks=list(pair),
                    device_id=torch.device("cuda", torch.cuda.current_device()),
                )
                if self._global_rank in pair:
                    self._p2p_pg = pg

            self._peer_p2p_rank = 1 - dist.get_rank(self._p2p_pg)
            logger.info(
                f"[SnapshotGroup] rank={self._global_rank} all pair PGs ready"
            )

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

    @property
    def partner_plan(self):
        """The shared-rule pairing of this rank's candidate pool (None when
        torch.distributed is not initialized)."""
        return self._partner_plan

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

    def find_consistent_step_and_action(
        self, available_steps: set, faulty_ranks: list | None = None
    ):
        """Find the latest consistent checkpoint step across all ranks.

        Each rank provides its set of available steps and its peer rank.
        For each peer pair, the recoverable steps are the union.
        The consistent step is the latest step recoverable by ALL pairs.

        With a non-empty ``faulty_ranks`` (global ranks, from leto's
        LETO_FAULTY_RANKS — plan §3.7), the election still chooses the step
        but the ROLE comes from the faulty set via ``assign_faulty_roles``
        (faulty -> RECV, its pair -> SEND, others -> LOCAL), with the
        election's action retained as a logged cross-check. Empty/None
        faulty_ranks (first start / legacy launcher) leaves behavior
        unchanged.

        Args:
            available_steps: Set of steps this rank has checkpoint data for.
            faulty_ranks: Faulty global ranks of the fault being recovered
                from (empty/None -> pure election).

        Returns:
            (target_step, load_action): target_step is None if no consistent step found.
        """
        # All-gather (available_steps, peer_global_rank) from every rank
        all_info = [None] * self._global_world_size
        dist.all_gather_object(
            all_info,
            (available_steps, self._peer_global_rank),
            group=self._gloo_pg,
        )

        # Build all unique peer pairs, compute recoverable steps per pair and
        # elect the latest step recoverable by ALL pairs (pure logic in
        # elect_consistent_step, incl. the wiped-pair fresh-start fallback).
        target_step, _pair_sets = elect_consistent_step(
            all_info, self._global_world_size
        )
        if target_step is None:
            return None, CheckpointLoadAction.NONE

        # Determine load action for this rank
        my_steps = all_info[self._global_rank][0] or set()
        peer_steps = all_info[self._peer_global_rank][0] or set()
        has_step = target_step in my_steps
        peer_has_step = target_step in peer_steps

        if has_step and peer_has_step:
            election_action = CheckpointLoadAction.LOCAL
        elif has_step and not peer_has_step:
            election_action = CheckpointLoadAction.SEND
        elif not has_step and peer_has_step:
            election_action = CheckpointLoadAction.RECV
        else:
            raise RuntimeError(
                f"Step {target_step} is in consistent_steps but neither rank has it. "
                f"available={available_steps}, peer={peer_steps}"
            )

        if not faulty_ranks:
            return target_step, election_action

        # Faulty-rank-driven roles (plan §3.7): the faulty rank's own dump is
        # never trusted — transient included — so it RECVs from its pair.
        action = assign_faulty_roles(
            election_action,
            target_step,
            my_steps,
            peer_steps,
            self_faulty=self._global_rank in faulty_ranks,
            pair_faulty=self._peer_global_rank in faulty_ranks,
            global_rank=self._global_rank,
            peer_global_rank=self._peer_global_rank,
        )
        return target_step, action