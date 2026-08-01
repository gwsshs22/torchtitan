"""MoEvement upstream logging over torch.distributed.pipelining (plan §3.5,
§9-M4, §9-M12).

Each rank logs what its pipeline stages *RECEIVE* at stage boundaries: every
chunk's forward activations arriving from the previous stage, and every
chunk's gradients arriving from the next one. Entries are keyed
``(iteration, true micro-batch index, GLOBAL virtual-stage id of the
PRODUCING stage, direction)`` — never a pipeline buffer id, fixing the
reference's ``buffer_id`` aliasing defect outright (MoEvement
upstream_logging.py keys alias whenever micro_batches > num_pipe_buffers;
moe_upstream report §4).

WHY RECEIVE-SIDE (M12; the reference and M4 logged the SEND side)
-----------------------------------------------------------------
The paper's recovery ships a recovering stage's logs from its LIVE surviving
neighbours over the wire, so logging what you send is the right choice there.
leto kills every rank: there are no survivors to ship from, and a rank's own
*sent* tensors are useless to itself during replay — under loop-mapped
Interleaved1F1B every adjacent stage pair lives on different ranks, so the
sender-keyed recv-override provably never fires (measured: M4 e2e deviation
(3), fill_hits == 0).

Every boundary tensor is sent exactly once and received exactly once, so
logging it on the receive side costs the same bytes and the same copies — but
it makes each rank SELF-SUFFICIENT: after relaunch it reloads its own
persisted logs, feeds its own boundary inputs, and its stages replay without
waiting on any neighbour. That is the paper's bubble-elimination benefit,
reachable without survivors.

Tee points (receive completion), spiked against torch 2.11.0.dev20260120:

- ``_PipelineScheduleRuntime._step_microbatches`` (schedules.py:2062)
  dispatches every action with the TRUE ``action.microbatch_index``.
- forward: ``stage._retrieve_recv_activations(mb)`` (stage.py:548), called
  from ``forward_one_chunk`` (:691) strictly AFTER the schedule waited on
  RECV_F (schedules.py:2148) or a same-rank producer ran
  ``set_local_fwd_input`` (:2161). It reads ``args_recv_info[mb]`` — exactly
  the buffers the posted irecv wrote.
- backward: ``stage._retrieve_recv_grads(mb)`` (stage.py:556), called from
  ``backward_one_chunk`` (:779) after RECV_B was waited (schedules.py:2177 /
  :2201) or ``set_local_bwd_input`` ran (:2190 / :2212). It reads
  ``grad_recv_info[mb]``.
- Both are invoked exactly once per (mb, direction) and only on stages that
  actually receive (``is_first`` skips the forward branch, ``is_last`` the
  backward one) — no gating of our own is needed.

Log-fed replay, and why it cannot deadlock
------------------------------------------
``get_fwd_recv_ops``/``get_bwd_recv_ops`` fill ``args_recv_info`` /
``grad_recv_info`` from the reloaded logs and return ``[]``
(``_batch_p2p([])`` / ``_wait_batch_p2p([])`` are no-ops, schedules.py:460 /
:498); same-rank adjacencies go through ``set_local_fwd_input`` /
``set_local_bwd_input`` instead of wire ops and are overridden too.

A skipped recv leaves the matching isend unmatched, so the SENDER has to skip
it as well. The decision is therefore made from one world-agreed bit vector:
``set_log_fed_ranks`` records which GLOBAL ranks replay from their own logs,
and **a p2p transfer happens iff its RECEIVER is not log-fed**. Receiver and
sender evaluate the same predicate, so sends and recvs stay matched by
construction — including the mixed case a FATAL produces (the wiped host's
ranks have no logs and keep receiving live, their peers keep sending to
them). Correctness never depends on logs being present: a rank without them
simply replays over live p2p.

Storage: one additional shm ring pool per rank (retention 2*w_sparse
iterations), registered with the SAME SnapshotContainer under state key
"logs" with its own ``log_i{step}`` commit-key stream; commits ride the
engine's existing committer thread and are event-confirmed. Logs are
deliberately NOT replicated to the partner (plan §3.7). Tee copies are async
D2H enqueued on the COMPUTE stream (a deliberate deviation from the plan's
side-stream design — see log_recv: side-stream tees of in-flight schedule
payloads corrupted training numerics on the A100-PCIe cluster). The pool is
sized from the first logged iteration (payload shapes are schedule-static, so
per-iteration log bytes are constant); iteration 1 stages into per-tensor
pinned host buffers and is copied into the pool once allocated behind a full
device sync.
"""

import logging
import os
import weakref
from typing import Any

import torch
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.stage import _RecvInfo

from torchtitan.components.moevement.snapshot_engine import CudaTransferBackend

logger = logging.getLogger(__name__)

# Directions, keyed by the PRODUCING stage (unchanged from the send-side
# keying, so keys, dumps and reload are byte-compatible). A receiver stage s
# resolves its forward input from (iter, mb, s-1, DIR_FWD) and its grad recv
# from (iter, mb, s+1, DIR_BWD) — i.e. from the `source` field of its own
# _RecvInfo, which is what the receive-side tee records.
DIR_FWD = "fwd_out"  # forward output of stage s == fwd input of stage s+1
DIR_BWD = "bwd_grad_in"  # input-grads of stage s == grad recv of stage s-1

_SLOT_ALIGN = 4096
_ENTRY_ALIGN = 64  # covers element alignment for every dtype


# ---------------------------------------------------------------------------
# Tee stage registry: stages are built (pipeline_parallel.pipeline_module_split)
# before the checkpoint manager exists; the manager attaches the logger later
# via attach_logger_to_stages. Weakrefs so stale trainers don't leak.
# ---------------------------------------------------------------------------

_tee_stages: list = []


def reset_tee_registry() -> None:
    _tee_stages.clear()


def registered_tee_stages() -> list:
    alive = []
    for ref in _tee_stages:
        stage = ref()
        if stage is not None:
            alive.append(stage)
    return alive


def attach_logger_to_stages(
    upstream_logger: "UpstreamLogger",
    model_parts: list,
    stage_ids: list[int],
) -> int:
    """Attach the logger to every registered tee stage whose submodule is one
    of ``model_parts`` (stages of other/stale trainers are skipped), verifying
    each stage's true ``stage_index`` against the manager's computed ids
    (pinned P1: the 'loop' mapping must agree with how pipeline_llm built the
    parts). Returns the number of stages attached."""
    part_idx = {id(part): idx for idx, part in enumerate(model_parts)}
    attached = 0
    for stage in registered_tee_stages():
        idx = part_idx.get(id(stage.submod))
        if idx is None:
            continue
        assert stage.stage_index == stage_ids[idx], (
            f"[moevement] stage-id mapping mismatch: model_parts[{idx}] is "
            f"pipeline stage {stage.stage_index} but the loop mapping gives "
            f"{stage_ids[idx]}"
        )
        stage.upstream_logger = upstream_logger
        attached += 1
    return attached


class UpstreamTeePipelineStage(PipelineStage):
    """PipelineStage with receive-payload tees and log-fed replay overrides.

    Copies only — never mutates payloads or reorders schedule ops. Inert
    (plain PipelineStage behavior) until ``upstream_logger`` is attached.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.upstream_logger: UpstreamLogger | None = None
        _tee_stages.append(weakref.ref(self))

    # -- capture tees (RECEIVE completion) ---------------------------------

    def _retrieve_recv_activations(self, fwd_chunk_id: int):
        """Forward-boundary tee. Called from ``forward_one_chunk`` only on
        non-first stages, strictly after the schedule waited on RECV_F (or a
        same-rank producer ran ``set_local_fwd_input``): ``args_recv_info``
        now holds exactly the bytes that crossed the boundary."""
        activations = super()._retrieve_recv_activations(fwd_chunk_id)
        lg = self.upstream_logger
        if lg is not None:
            lg.log_recv_infos(
                DIR_FWD, self.args_recv_info[fwd_chunk_id], fwd_chunk_id
            )
        return activations

    def _retrieve_recv_grads(self, bwd_chunk_id: int):
        """Backward-boundary tee. Called from ``backward_one_chunk`` only on
        non-last stages, after RECV_B was waited (or ``set_local_bwd_input``
        ran)."""
        grads = super()._retrieve_recv_grads(bwd_chunk_id)
        lg = self.upstream_logger
        if lg is not None:
            lg.log_recv_infos(
                DIR_BWD, self.grad_recv_info[bwd_chunk_id], bwd_chunk_id
            )
        return grads

    # -- log-fed replay: sender side ---------------------------------------
    #
    # A log-fed receiver never posts its irecv, so the matching isend must not
    # be posted either. Both sides read the same world-agreed vector, so the
    # predicate "this transfer happens iff its RECEIVER is not log-fed" keeps
    # sends and recvs matched without any extra handshake.

    def _drop_log_fed_sends(self, ops):
        lg = self.upstream_logger
        if not ops or lg is None or not lg.wire_skip_active:
            return ops
        return [op for op in ops if not lg.peer_is_log_fed(op.peer)]

    def get_fwd_send_ops(self, fwd_chunk_id: int):
        return self._drop_log_fed_sends(super().get_fwd_send_ops(fwd_chunk_id))

    def get_bwd_send_ops(self, bwd_chunk_id: int):
        return self._drop_log_fed_sends(super().get_bwd_send_ops(bwd_chunk_id))

    # -- log-fed replay: receiver side -------------------------------------

    def get_fwd_recv_ops(self, fwd_chunk_id: int):
        lg = self.upstream_logger
        if lg is not None and lg.replay_active and not self.is_first:
            if lg.fill_recv_infos(
                self.args_recv_info[fwd_chunk_id], fwd_chunk_id, DIR_FWD
            ):
                return []
        return super().get_fwd_recv_ops(fwd_chunk_id)

    def get_bwd_recv_ops(self, bwd_chunk_id: int):
        lg = self.upstream_logger
        if (
            lg is not None
            and lg.replay_active
            and self.has_backward
            and not self.is_last
        ):
            if lg.fill_recv_infos(
                self.grad_recv_info[bwd_chunk_id], bwd_chunk_id, DIR_BWD
            ):
                return []
        return super().get_bwd_recv_ops(bwd_chunk_id)

    def set_local_fwd_input(self, prev_stage_outputs: Any, mb_index: int) -> None:
        lg = self.upstream_logger
        if lg is not None and lg.replay_active:
            if lg.fill_local_fwd_input(self.args_recv_info[mb_index], mb_index):
                return
        super().set_local_fwd_input(prev_stage_outputs, mb_index)

    def set_local_bwd_input(self, next_stage_bwd_outputs, mb_index: int) -> None:
        lg = self.upstream_logger
        if lg is not None and lg.replay_active:
            if lg.fill_local_bwd_input(self.grad_recv_info[mb_index], mb_index):
                return
        super().set_local_bwd_input(next_stage_bwd_outputs, mb_index)


class UpstreamLogger:
    """Per-rank upstream-RECEIVE log over a container-attached shm ring pool.

    Iteration protocol (driven by the checkpoint manager):
      begin_iteration(step)          — evict the ring slot being reused
      log_recv_infos(...) x N        — tees, async D2H (compute-stream-ordered)
      finish_iteration(step)         — record event, enqueue commit

    Replay protocol:
      arm_replay(load_persisted_logs(...), required_iterations=...) reports
      whether this rank is SELF-SUFFICIENT for the whole replay window; the
      manager all-gathers that bit and calls set_log_fed_ranks with the world
      vector. Then begin_iteration(step, replaying=True) per replayed step;
      the tee stages consult fill_recv_infos / fill_local_*; end_replay() when
      the window is done. Replay iterations never re-log (reference behavior).
    """

    def __init__(
        self,
        engine,
        mem_fs_folder: str,
        rank: int,
        retention_iters: int,
        backend=None,
    ):
        self._engine = engine
        self._mem_fs_folder = mem_fs_folder
        self._rank = rank
        self._capacity = max(2, int(retention_iters))
        # Backend supplies pool/host allocation, events, and the full-sync
        # fence; tee copies themselves ride the compute stream (see
        # log_recv for why the plan's side-stream D2H was abandoned).
        self._backend = backend if backend is not None else CudaTransferBackend()

        self._pool: torch.UntypedStorage | None = None
        self._pool_nbytes = 0
        self._slot_bytes = 0
        self._slot_base = 0
        self._slot_steps: dict[int, int] = {}  # ring slot -> committed step

        self._iteration: int | None = None
        self._replaying = False
        self._active = False
        self._iter_entries: list[dict[str, Any]] = []
        self._iter_offset = 0
        # First-iteration host staging (pool not yet sized).
        self._staged: list[tuple[dict[str, Any], torch.Tensor]] = []

        self._replay_logs: dict | None = None
        # Log-fed replay gate (see the module docstring): whether THIS rank
        # feeds its boundary recvs from its own logs, and which global ranks
        # the world agreed are doing so (None until the manager announces it,
        # e.g. in single-process unit tests).
        self._required_iterations: tuple[int, ...] = ()
        self._missing_iterations: list[int] = []
        self._self_sufficient = False
        self._log_fed = False
        self._log_fed_ranks: frozenset[int] | None = None
        self.fill_hits = 0  # replay-override engagements (tests/verification)
        self.fill_misses = 0  # live-p2p fallbacks

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def replay_active(self) -> bool:
        """This rank feeds its boundary recvs from its own reloaded logs."""
        return self._replaying and self._replay_logs is not None and self._log_fed

    @property
    def self_sufficient(self) -> bool:
        """The reloaded store covers every iteration this rank must replay."""
        return self._self_sufficient

    @property
    def missing_iterations(self) -> list[int]:
        return list(self._missing_iterations)

    @property
    def wire_fill_enabled(self) -> bool:
        """A WIRE recv may only be skipped once the world vector is settled —
        that vector is what made the peer drop the matching send. Without it,
        skipping the recv leaves an unmatched isend and the pipeline hangs
        (reproduced deliberately on a 2-rank gloo probe). Same-rank (local)
        boundaries carry no wire op and are not gated by this."""
        return self.replay_active and self._log_fed_ranks is not None

    @property
    def wire_skip_active(self) -> bool:
        """The world vector is known and we are replaying: send ops toward
        log-fed receivers must be dropped."""
        return self._replaying and self._log_fed_ranks is not None

    def peer_is_log_fed(self, peer_rank: int) -> bool:
        return (
            self._log_fed_ranks is not None
            and int(peer_rank) in self._log_fed_ranks
        )

    # -- capture path -------------------------------------------------------

    def begin_iteration(self, step: int, replaying: bool = False) -> None:
        self._iteration = step
        self._replaying = replaying
        if replaying:
            self._active = False
            return
        self._active = True
        self._iter_entries = []
        self._iter_offset = 0
        if self._pool is not None:
            slot = step % self._capacity
            old = self._slot_steps.pop(slot, None)
            if old is not None and old != step:
                # Torn-write safety: drop the evicted iteration's ledger
                # entry BEFORE its slot bytes are overwritten. This IS the
                # retention policy: the ring keeps exactly the last
                # `capacity` iterations (continuous eviction — equivalent to
                # the reference's window-boundary GC horizon).
                self._engine.invalidate_log_step(old)
            self._slot_base = slot * self._slot_bytes

    def log_recv_infos(self, direction: str, recv_infos, mb_index: int) -> None:
        """Receive-completion tee: record the arrived boundary buffers,
        grouped by the stage that PRODUCED them (``_RecvInfo.source``), so the
        key space is identical to the send-side keying and ``_resolve_fills``
        consumes the per-source lists in exactly the order it recorded them.
        """
        if not self._active:
            return
        by_source: dict[int, list[torch.Tensor]] = {}
        for info in recv_infos:
            if isinstance(info, _RecvInfo) and isinstance(
                info.buffer, torch.Tensor
            ):
                by_source.setdefault(info.source, []).append(info.buffer)
        for source, tensors in by_source.items():
            self.log_recv(direction, source, mb_index, tensors)

    @torch.no_grad()
    def log_recv(
        self,
        direction: str,
        stage_index: int,
        mb_index: int,
        tensors,
    ) -> None:
        """``stage_index`` is the PRODUCING (peer) stage — the key a receiver
        looks the payload up under during replay."""
        if not self._active:
            return
        payload = [t for t in tensors if isinstance(t, torch.Tensor)]
        if not payload:
            return
        # Copies are enqueued on the CURRENT (compute) stream, non-blocking
        # into pinned shm — stream-ordered with the completed recv (the
        # schedule waited on it before this call) and with every later
        # consumer, so no cross-stream race surface exists by construction.
        # DEVIATION from plan §3.5's "logger's own side stream": on this
        # cluster (A100 PCIe, driver 580.82.07), teeing the in-flight schedule
        # payloads from a dedicated side stream corrupted training numerics
        # (NaN from step 1; M4 bisect) even though the side stream
        # wait_stream'd the producer — the same pattern the snapshot engine
        # uses safely OUTSIDE the schedule. Cost: the D2H serializes into the
        # compute stream (~2-3 ms/step at gptoss-pp2's 58 MB/iter) instead of
        # overlapping; still no CPU sync.
        for elem_idx, tensor in enumerate(payload):
            src = tensor.detach()
            if not src.is_contiguous():
                src = src.contiguous()
            elem_size = src.element_size()
            nbytes = src.numel() * elem_size
            offset = (
                (self._iter_offset + _ENTRY_ALIGN - 1)
                // _ENTRY_ALIGN
                * _ENTRY_ALIGN
            )
            meta = {
                "iteration": self._iteration,
                "mb": mb_index,
                "stage": stage_index,
                "direction": direction,
                "elem": elem_idx,
                "dtype": str(src.dtype).removeprefix("torch."),
                "shape": list(src.shape),
                "offset": offset,  # slot-relative; absolute at commit
                "nbytes": nbytes,
            }
            if self._pool is not None:
                if offset + nbytes > self._slot_bytes:
                    raise RuntimeError(
                        f"[moevement] upstream log slot overflow at "
                        f"iteration {self._iteration}: offset {offset} + "
                        f"{nbytes}B > slot {self._slot_bytes}B — per-"
                        f"iteration log volume grew after pool sizing"
                    )
                dst = torch.empty(0, dtype=src.dtype)
                dst.set_(
                    source=self._pool,
                    storage_offset=(self._slot_base + offset) // elem_size,
                    size=(src.numel(),),
                )
                dst.copy_(src.reshape(-1), non_blocking=True)
            else:
                host = self._backend.alloc_host(src.numel(), src.dtype)
                host.copy_(src.reshape(-1), non_blocking=True)
                self._staged.append((meta, host))
            self._iter_offset = offset + nbytes
            self._iter_entries.append(meta)

    def finish_iteration(self, step: int) -> None:
        if not self._active:
            return
        self._active = False
        assert step == self._iteration, (
            f"finish_iteration({step}) without begin_iteration "
            f"(current {self._iteration})"
        )
        if not self._iter_entries:
            return
        # The commit is gated on this event: copies ran on the compute
        # stream, so record there.
        event = self._backend.record_event_on_current_stream()
        if self._pool is None:
            # One-time: confirm the staged D2H, size the pool from this
            # iteration (payload shapes are schedule-static => constant
            # per-iteration bytes), and migrate the staged copies in.
            event.synchronize()
            self._allocate_pool()
            self._slot_base = (step % self._capacity) * self._slot_bytes
            for meta, host in self._staged:
                elem_size = host.element_size()
                dst = torch.empty(0, dtype=host.dtype)
                dst.set_(
                    source=self._pool,
                    storage_offset=(self._slot_base + meta["offset"])
                    // elem_size,
                    size=(host.numel(),),
                )
                dst.copy_(host)
            self._staged.clear()
        header = [
            dict(meta, offset=self._slot_base + meta["offset"])
            for meta in self._iter_entries
        ]
        self._engine.enqueue_log_iteration(step, event, header, [])
        self._slot_steps[step % self._capacity] = step
        self._iter_entries = []
        self._iter_offset = 0

    def _allocate_pool(self) -> None:
        slot = int(self._iter_offset * 1.05) + _SLOT_ALIGN
        self._slot_bytes = (slot + _SLOT_ALIGN - 1) // _SLOT_ALIGN * _SLOT_ALIGN
        self._pool_nbytes = self._slot_bytes * self._capacity
        path = os.path.join(
            self._mem_fs_folder,
            f"moevement_logpool_rank{self._rank}_pid{os.getpid()}",
        )
        # Quiesce the device before pinning: registering a multi-hundred-MB
        # host range concurrently with in-flight NCCL/compute corrupted
        # training numerics on this cluster (steps 2+ NaN with all tee
        # copies disabled; M4 bisect). One-time cost at the first logged
        # iteration's finalize.
        self._backend.full_sync()
        self._pool = self._backend.alloc_pool(path, self._pool_nbytes)
        # Register-then-unlink (gemini pattern): the container's mapping keeps
        # the anonymous pool alive through SIGKILL of this process.
        self._engine.register_log_pool(path, self._pool_nbytes)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        logger.info(
            "[moevement] upstream log ring: %d iterations x %.2f MB = %.2f MB",
            self._capacity,
            self._slot_bytes / (1024 * 1024),
            self._pool_nbytes / (1024 * 1024),
        )

    # -- replay path --------------------------------------------------------

    def arm_replay(
        self, logs: dict | None, required_iterations=None
    ) -> bool:
        """logs: {(iteration, mb, stage, direction): [tensor, ...]} — the
        per-key payload lists in wire order (load_persisted_logs builds this
        from the container dump).

        ``required_iterations`` is the replay window's step range. An
        iteration is committed only after *all* of its tees are
        event-confirmed, and the replayed schedule performs exactly the recv
        set the captured one did (same topology, same gas, same stage
        assignment), so "every required iteration is present" is equivalent
        to "every boundary recv of the replay resolves". Returns that bit —
        the manager all-gathers it and feeds the world vector back through
        ``set_log_fed_ranks``.
        """
        self._replay_logs = logs if logs else None
        self._required_iterations = (
            tuple(sorted({int(i) for i in required_iterations}))
            if required_iterations is not None
            else ()
        )
        have = {key[0] for key in self._replay_logs} if self._replay_logs else set()
        self._missing_iterations = [
            i for i in self._required_iterations if i not in have
        ]
        self._self_sufficient = (
            self._replay_logs is not None and not self._missing_iterations
        )
        # Until the world announces its decision (single-process tests), this
        # rank feeds from its own logs iff it has them; no wire skipping.
        self._log_fed = self._self_sufficient
        self._log_fed_ranks = None
        self.fill_hits = 0
        self.fill_misses = 0
        return self._self_sufficient

    def set_log_fed_ranks(self, ranks) -> None:
        """World-agreed set of GLOBAL ranks replaying from their own logs.

        A p2p transfer happens iff its RECEIVER is not in this set. Both ends
        read this same vector — the receiver to skip its recv, the sender to
        drop the matching send op — so no transfer is ever half-skipped.
        """
        self._log_fed_ranks = frozenset(int(r) for r in ranks)
        self._log_fed = self._rank in self._log_fed_ranks
        if self._log_fed and not self._self_sufficient:
            raise RuntimeError(
                f"[moevement] rank {self._rank} was announced log-fed but its "
                f"own log store is incomplete (missing iterations "
                f"{self._missing_iterations}) — the world vote and the local "
                f"store disagree"
            )

    def end_replay(self) -> None:
        self._replay_logs = None
        self._replaying = False
        self._log_fed = False
        self._log_fed_ranks = None
        self._required_iterations = ()
        self._missing_iterations = []

    def fetch(
        self, iteration: int, mb: int, stage: int, direction: str
    ) -> list[torch.Tensor] | None:
        if self._replay_logs is None:
            return None
        return self._replay_logs.get((iteration, mb, stage, direction))

    def _resolve_fills(self, recv_infos, mb_index: int, direction: str):
        """All-or-nothing resolution of every _RecvInfo's logged tensor.
        Returns the per-info tensor list, or None to fall back to live
        exchange (this rank is not log-fed, missing entries, count/shape/dtype
        mismatch)."""
        if not self._log_fed or self._replay_logs is None:
            return None
        infos = [info for info in recv_infos if isinstance(info, _RecvInfo)]
        if not infos or self._iteration is None:
            return None
        entries_by_source: dict[int, list[torch.Tensor]] = {}
        needed: dict[int, int] = {}
        for info in infos:
            needed[info.source] = needed.get(info.source, 0) + 1
        for source, count in needed.items():
            got = self.fetch(self._iteration, mb_index, source, direction)
            if got is None or len(got) != count:
                return None
            entries_by_source[source] = list(got)
        cursor = {source: 0 for source in entries_by_source}
        resolved = []
        for info in infos:
            logged = entries_by_source[info.source][cursor[info.source]]
            cursor[info.source] += 1
            if (
                tuple(logged.shape) != tuple(info.buffer.shape)
                or logged.dtype != info.buffer.dtype
            ):
                return None
            resolved.append((info, logged))
        return resolved

    @torch.no_grad()
    def fill_recv_infos(self, recv_infos, mb_index: int, direction: str) -> bool:
        """Wire-recv override: populate the preallocated recv buffers in
        place (exactly what the posted irecv would have written) and report
        success; the caller then returns an empty op list.

        ``no_grad`` is load-bearing, not defensive: forward recv buffers are
        leaves with ``requires_grad_(True)`` (stage.py:1158-1162), so the
        in-place copy — the same write the irecv performs — is illegal under
        grad mode. (Never hit before receive-side keying: the sender-keyed
        override could not fire on a cross-rank boundary at all.)
        """
        if not self.wire_fill_enabled:
            return False
        resolved = self._resolve_fills(recv_infos, mb_index, direction)
        if resolved is None:
            self.fill_misses += 1
            if self._log_fed and self._log_fed_ranks is not None:
                # The sender already dropped the matching isend on the
                # strength of this rank's announced log-fed bit, so a silent
                # live-p2p fallback here would hang the pipeline. Fail loudly
                # instead. Unreachable while arm_replay's completeness gate
                # holds (the replayed recv set equals the captured one).
                raise RuntimeError(
                    f"[moevement] log-fed replay: rank {self._rank} could not "
                    f"resolve a {direction} boundary recv at iteration "
                    f"{self._iteration} mb {mb_index} from its own logs, but "
                    f"the world was told it would — the peer has already "
                    f"skipped the matching send"
                )
            return False
        for info, logged in resolved:
            info.buffer.copy_(logged.to(info.buffer.device))
        self.fill_hits += 1
        return True

    def fill_local_fwd_input(self, recv_infos, mb_index: int) -> bool:
        """Same-rank boundary override, forward direction: mirrors
        set_local_fwd_input's semantics (fresh leaf with requires_grad)."""
        resolved = self._resolve_fills(recv_infos, mb_index, DIR_FWD)
        if resolved is None:
            self.fill_misses += 1
            return False
        for info, logged in resolved:
            info.buffer = (
                logged.to(info.buffer.device).detach().requires_grad_(True)
            )
        self.fill_hits += 1
        return True

    def fill_local_bwd_input(self, recv_infos, mb_index: int) -> bool:
        """Same-rank boundary override, backward direction: mirrors
        set_local_bwd_input (plain tensor hand-off)."""
        resolved = self._resolve_fills(recv_infos, mb_index, DIR_BWD)
        if resolved is None:
            self.fill_misses += 1
            return False
        for info, logged in resolved:
            info.buffer = logged.to(info.buffer.device)
        self.fill_hits += 1
        return True


def load_persisted_logs(
    mem_fs_folder: str, rank: int
) -> dict[tuple[int, int, int, str], list[torch.Tensor]]:
    """Rebuild this rank's persisted upstream-log store from the container
    dump (rank_{r}_moevement_logs.pt): zero-copy typed views over the dumped
    pool bytes, keyed (iteration, mb, PRODUCING stage, direction) with
    elem-ordered payload lists. Unreadable/absent dumps degrade to an empty
    store — this rank is then not self-sufficient and replays every boundary
    over live p2p (the fatal-fault case: mem_fs was wiped with the logs)."""
    path = os.path.join(mem_fs_folder, f"rank_{rank}_moevement_logs.pt")
    if not os.path.exists(path):
        return {}
    try:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        pool_bytes = saved["_pool_bytes"]
        iters = saved["_iters"]
    except Exception as e:
        logger.warning(
            "[moevement] unreadable upstream-log dump %s: %s", path, e
        )
        return {}
    storage = pool_bytes.untyped_storage()
    keyed: dict[tuple[int, int, int, str], list[tuple[int, torch.Tensor]]] = {}
    for _step, header in iters.items():
        for meta in header:
            dtype = getattr(torch, meta["dtype"])
            elem_size = torch.empty(0, dtype=dtype).element_size()
            view = torch.empty(0, dtype=dtype)
            view.set_(
                source=storage,
                storage_offset=meta["offset"] // elem_size,
                size=tuple(meta["shape"]),
            )
            key = (
                meta["iteration"],
                meta["mb"],
                meta["stage"],
                meta["direction"],
            )
            keyed.setdefault(key, []).append((meta["elem"], view))
    store: dict[tuple[int, int, int, str], list[torch.Tensor]] = {}
    for key, entries in keyed.items():
        entries.sort(key=lambda pair: pair[0])
        # Each view holds a reference to the dumped pool storage, so the
        # bytes stay alive for as long as any entry is used.
        store[key] = [tensor for _elem, tensor in entries]
    return store
