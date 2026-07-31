"""MoEvement upstream logging over torch.distributed.pipelining (plan §3.5,
§9-M4).

Each rank logs what its pipeline stages *send*: every chunk's forward output
as it becomes the SEND_F payload, and every chunk's computed input-gradients
as they become the SEND_B payload. Entries are keyed
``(iteration, true micro-batch index, GLOBAL virtual-stage id, direction)``
— never a pipeline buffer id, fixing the reference's ``buffer_id`` aliasing
defect outright (MoEvement upstream_logging.py keys alias whenever
micro_batches > num_pipe_buffers; moe_upstream report §4).

Override surface (spiked against torch 2.11.0.dev20260120+cu128):

- ``_PipelineScheduleRuntime._step_microbatches`` (schedules.py:2031)
  dispatches every action with the TRUE ``action.microbatch_index``.
- fwd send payload == ``stage.fwd_cache[mb][0]`` (stage.py:446), populated at
  the end of ``forward_one_chunk`` (:720) -> tee there.
- bwd send payload == ``stage.bwd_cache[mb]`` filtered through
  ``grad_send_info`` (stage.py:488-509), populated at the end of
  ``backward_one_chunk`` (:833); ``get_bwd_send_ops`` POPS the cache, so the
  tee must run inside the ``backward_one_chunk`` override (SEND_B is a later,
  separate action).
- replay recv override: ``get_fwd_recv_ops``/``get_bwd_recv_ops`` fill
  ``args_recv_info``/``grad_recv_info`` buffers from logs and return ``[]``
  (``_batch_p2p([])`` and ``_wait_batch_p2p([])`` are no-ops,
  schedules.py:464/498). Same-rank adjacent stages exchange via
  ``set_local_fwd_input``/``set_local_bwd_input`` instead of wire ops
  (``_add_send_recv._has_comms`` skips them, schedules.py:1219-1228) -> those
  are overridden too.

M4 scope: logs are per-rank local (reloaded from this rank's own container
dump). A boundary's log lives on the SENDER's rank, so recv-override can only
fire where the sender stage is local to this rank (same-rank adjacencies —
V-style schedules or single-rank multi-stage pipelines). Under loop-mapped
interleaved schedules every adjacent boundary is cross-rank, and replay falls
back to live p2p — correct under whole-cluster replay. Cross-rank log
exchange is the M5+ extension point.

Storage: one additional shm ring pool per rank (retention 2*w_sparse
iterations), registered with the SAME SnapshotContainer under state key
"logs" with its own ``log_i{step}`` commit-key stream; commits ride the
engine's existing committer thread and are event-confirmed. Tee copies are
async D2H enqueued on the COMPUTE stream (a deliberate deviation from the
plan's side-stream design — see log_send: side-stream tees of in-flight
schedule payloads corrupted training numerics on the A100-PCIe cluster).
The pool is sized from the first logged iteration (payload shapes are
schedule-static, so per-iteration log bytes are constant); iteration 1
stages into per-tensor pinned host buffers and is copied into the pool once
allocated behind a full device sync.
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

# Directions: what the SENDER logs. A receiver stage s resolves its forward
# input from (iter, mb, s-1, DIR_FWD) and its grad recv from
# (iter, mb, s+1, DIR_BWD).
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
    """PipelineStage with send-payload tees and log-fed replay overrides.

    Copies only — never mutates payloads or reorders schedule ops. Inert
    (plain PipelineStage behavior) until ``upstream_logger`` is attached.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.upstream_logger: UpstreamLogger | None = None
        _tee_stages.append(weakref.ref(self))

    # -- capture tees -------------------------------------------------------

    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args,
        kwargs=None,
        save_forward_output: bool = True,
    ):
        output = super().forward_one_chunk(
            fwd_chunk_id, args, kwargs, save_forward_output
        )
        lg = self.upstream_logger
        if lg is not None and not self.is_last:
            # fwd_cache[mb][0] is exactly what get_fwd_send_ops ships (and
            # what set_local_fwd_input hands a same-rank next stage).
            output_tuple, _ = self.fwd_cache[fwd_chunk_id]
            lg.log_send(DIR_FWD, self.stage_index, fwd_chunk_id, output_tuple)
        return output

    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward: bool = False,
    ):
        super().backward_one_chunk(
            bwd_chunk_id,
            loss=loss,
            full_backward=full_backward,
            last_backward=last_backward,
        )
        lg = self.upstream_logger
        if (
            lg is not None
            and self.has_backward
            and not self.is_first
            and bwd_chunk_id in self.bwd_cache
        ):
            if self.grad_send_info is None:
                # Same lazy construction get_bwd_send_ops performs.
                self.grad_send_info = self._create_grad_send_info(
                    self.args_recv_info[0]
                )
            grads_input = self.bwd_cache[bwd_chunk_id]
            # The wire subset: mirrors get_bwd_send_ops' tensor+destination
            # filter, so the log holds exactly the bytes that hit the wire.
            payload = tuple(
                g
                for g, dst in zip(grads_input, self.grad_send_info)
                if isinstance(g, torch.Tensor) and dst is not None
            )
            lg.log_send(DIR_BWD, self.stage_index, bwd_chunk_id, payload)

    # -- log-fed replay overrides ------------------------------------------

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
    """Per-rank upstream-send log over a container-attached shm ring pool.

    Iteration protocol (driven by the checkpoint manager):
      begin_iteration(step)          — evict the ring slot being reused
      log_send(...) x N              — tees, async D2H (compute-stream-ordered)
      finish_iteration(step)         — record event, enqueue commit

    Replay protocol:
      arm_replay(load_persisted_logs(...)), then begin_iteration(step,
      replaying=True) per replayed step; the tee stages consult
      fill_recv_infos / fill_local_*; end_replay() when the window is done.
      Replay iterations never re-log (reference behavior).
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
        # log_send for why the plan's side-stream D2H was abandoned).
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
        self.fill_hits = 0  # replay-override engagements (tests/verification)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def replay_active(self) -> bool:
        return self._replaying and self._replay_logs is not None

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

    @torch.no_grad()
    def log_send(
        self,
        direction: str,
        stage_index: int,
        mb_index: int,
        tensors,
    ) -> None:
        if not self._active:
            return
        payload = [t for t in tensors if isinstance(t, torch.Tensor)]
        if not payload:
            return
        # Copies are enqueued on the CURRENT (compute) stream, non-blocking
        # into pinned shm — stream-ordered with both the payload's producer
        # and every later consumer, so no cross-stream race surface exists
        # by construction. DEVIATION from plan §3.5's "logger's own side
        # stream": on this cluster (A100 PCIe, driver 580.82.07), teeing the
        # in-flight schedule payloads from a dedicated side stream corrupted
        # training numerics (NaN from step 1; M4 bisect) even though the
        # side stream wait_stream'd the producer — the same pattern the
        # snapshot engine uses safely OUTSIDE the schedule. Cost: the D2H
        # serializes into the compute stream (~2-3 ms/step at gptoss-pp2's
        # 58 MB/iter) instead of overlapping; still no CPU sync.
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

    def arm_replay(self, logs: dict | None) -> None:
        """logs: {(iteration, mb, stage, direction): [tensor, ...]} — the
        per-key payload lists in wire order (load_persisted_logs builds
        this from the container dump)."""
        self._replay_logs = logs if logs else None

    def end_replay(self) -> None:
        self._replay_logs = None
        self._replaying = False

    def fetch(
        self, iteration: int, mb: int, stage: int, direction: str
    ) -> list[torch.Tensor] | None:
        if self._replay_logs is None:
            return None
        return self._replay_logs.get((iteration, mb, stage, direction))

    def _resolve_fills(self, recv_infos, mb_index: int, direction: str):
        """All-or-nothing resolution of every _RecvInfo's logged tensor.
        Returns the per-info tensor list, or None to fall back to live
        exchange (missing entries, count/shape/dtype mismatch)."""
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

    def fill_recv_infos(self, recv_infos, mb_index: int, direction: str) -> bool:
        """Wire-recv override: populate the preallocated recv buffers in
        place (exactly what the posted irecv would have written) and report
        success; the caller then returns an empty op list."""
        resolved = self._resolve_fills(recv_infos, mb_index, direction)
        if resolved is None:
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
    pool bytes, keyed (iteration, mb, stage, direction) with elem-ordered
    payload lists. Unreadable/absent dumps degrade to an empty store (replay
    then falls back to live p2p everywhere)."""
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
