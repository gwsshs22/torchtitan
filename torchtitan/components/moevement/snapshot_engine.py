"""MoEvement sparse-snapshot capture engine (plan §3.4).

Captures each iteration's scheduled operator subset from GPU into
container-attached shm pools: a dedicated side CUDA stream ``wait_stream``s
compute, packs per-dtype flats through GPU staging, and issues one async D2H
per dtype-run — no CPU sync on the hot path (reference
sparse_snapshot.py:373-434). A CUDA event recorded after each iteration's
D2H batch is (a) waited GPU-side by the NEXT step's pre-optimizer hook
(manager ``maybe_wait_for_staging``) so only ``optimizer.step`` — the next
writer of the source tensors — orders after the drain, and (b)
``synchronize``d by a background committer thread before that iteration is
committed to the container, so only event-confirmed bytes are ever committed
(gemini's HANDOFF-§5 durability lesson, applied per iteration).

Storage is a CURR/PREV double buffer of two window pools plus one shared
kill-survivable SnapshotContainer. Commits are append-style: every confirmed
iteration re-commits the current window's key with grown metadata, and
``finalize_window`` re-commits once more with the dataloader ring entry and
train-state scalars. The window-boundary barrier (plan §3.4) delays the
overwrite of window W-1's pool until W's final commit is confirmed, so once
the first window has finalized, a SIGKILL at any point leaves at least one
complete committed window; before that, recovery falls back to a fresh start
(documented property).
"""

import json
import logging
import os
import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from torchtitan.components.moevement.operators import Operator
from torchtitan.components.moevement.scheduler import (
    CheckpointSchedule,
    SparseCheckpointScheduler,
)
from torchtitan.components.snapshot import pool_shm
from torchtitan.components.snapshot.container import DumpPolicy, SnapshotContainer

logger = logging.getLogger(__name__)

# Alignment of every dtype-run's pool offset: keeps typed views over the pool
# element-aligned for any dtype.
_ALIGNMENT = 64
# Per-iteration allowance on top of scheduler.max_window_bytes(): per-iter RNG
# blobs ride commit metadata (not the pool), but the pool is sized as if they
# did — the slack doubles as cover for the dtype-run alignment padding, which
# max_window_bytes does not price.
_PER_ITER_ALLOWANCE = 16 * 1024
_HEADROOM = 1.05
# Slots in the pinned clip-norm ring (plan §3.3). A slot is reused only
# after the committer has consumed it; the committer is at most one window
# behind (the boundary barrier), and a window is at most `total_ops`
# iterations, so this bounds every configuration we run with ~4x margin.
_CLIP_NORM_RING = 1024
# Bound on the once-per-window boundary barrier; normally ~0 (the finalized
# window's last D2H event fired during the previous iteration).
_FINAL_COMMIT_TIMEOUT_S = 300.0


def _pool_name(index: int) -> str:
    return f"pool{index}"


def _local_slice(tensor: torch.Tensor, expert_idx: int | None) -> torch.Tensor:
    """The snapshot/restore unit: the (EP/FSDP-)local shard, optionally one
    expert's dim-0 slice of it. Shared by capture (engine) and in-place
    restore (conversion.apply_iteration) so both address the same bytes."""
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    local = local.detach()
    return local if expert_idx is None else local[expert_idx]


def window_used_nbytes(meta: dict[str, Any]) -> int:
    """Bytes of the window's pool actually referenced by its committed
    headers (max offset+nbytes across every iteration's tensors). This is
    the prefix replication ships AND the prefix the drain dumps — the rest
    of the pool is unused headroom. (Defined here rather than in
    conversion.py because MoevementDumpPolicy runs inside the container
    subprocess; conversion re-exports it for its existing importers.)"""
    used = 0
    for entry in meta.get("iters") or []:
        for op_meta in (entry.get("header") or {}).values():
            for tmeta in op_meta["tensors"].values():
                used = max(used, tmeta["offset"] + tmeta["nbytes"])
    return used


def find_adam_state(optimizers, param: torch.Tensor) -> dict:
    """Resolve the Adam state entry (exp_avg / exp_avg_sq / step) for
    ``param`` by reference.

    ``optimizers`` is any iterable of torch Optimizers — OptimizersContainer
    iterates its inner per-model-part optimizers, whose ``state`` dicts are
    keyed by the param objects themselves; the moments are (D)Tensors shaped
    like the param. Shared by the capture engine and by
    conversion.apply_iteration so capture and restore resolve moments
    identically.
    """
    for opt in optimizers:
        state = opt.state.get(param)
        if state and "exp_avg" in state:
            return state
    raise RuntimeError(
        "[moevement] no Adam state found for a scheduled parameter — "
        "capture must run post-optimizer-step (restore must materialize "
        "the optimizer state first)"
    )


class MoevementDumpPolicy(DumpPolicy):
    """Runs inside the container subprocess (picklable: str/int state only).

    Ledger: {window_key: metadata}. window_key = f"w{window_start}" — the
    window's first captured step, chosen over 'curr'/'prev' labels so a
    dumped file identifies its window without consulting metadata (the
    double-buffer position is an engine implementation detail; the "pool"
    field in the metadata records which registered pool holds the bytes).
    States: {pool_name ("pool0"/"pool1"): StateView}. Commits are
    append-style — the same key is re-committed with growing metadata — so
    on_commit imposes no pairing constraints.
    """

    def __init__(self, checkpoint_dir: str, rank: int):
        self.checkpoint_dir = checkpoint_dir
        self.rank = rank

    def on_commit(self, states, ledger, commit_key) -> None:
        assert ledger[commit_key].get("pool") in states, (
            f"commit {commit_key} references unregistered pool "
            f"{ledger[commit_key].get('pool')!r}"
        )

    def dump(self, states, ledger) -> None:
        window_items = {
            key: meta for key, meta in ledger.items()
            if meta.get("kind") != "log"
        }
        log_items = {
            key: meta for key, meta in ledger.items()
            if meta.get("kind") == "log"
        }

        # Dump order under mem_fs space pressure (ENOSPC mid-drain is
        # survivable and OBSERVED — the M7 deepseek FTFT drains on the
        # 126 GB-/dev/shm host fit only ~1-2 window files per rank). The
        # order is RECOVERY VALUE, newest first — what recovery actually
        # consumes must be written before anything it merely might use:
        #
        #   0. newest finalized OWN window — the window every non-faulty
        #      rank restores (load() picks max(local), find_committed_windows
        #      picks the newest). Window boundaries are world-synchronized
        #      (all ranks finalize the same window at the same step), so
        #      under uniform pressure every rank keeps the SAME newest start
        #      and the vote's MIN-of-newest is present everywhere.
        #   1. newest finalized REMOTE copy — the only thing a
        #      pair-of-faulty rank can serve to its faulty pair
        #      (serve_remote_window reads rank_{r}_moevement_r_w{S}.pt).
        #      Without it vote_contribution has no servable window and must
        #      vote FRESH-START: OBSERVED at M7 deepseek FTFT fault #3,
        #      where mew1 ranks dumped local=[129,133] but remote=[] because
        #      the old "all own windows first, oldest first" order spent the
        #      space on the OLDEST own windows and never reached the copies.
        #   2/3. older finalized own / remote — slack that only matters if
        #      the newest window turns out unusable, or if a remote copy
        #      lags its owner's newest by one window (the vote then agrees
        #      on the older start and the older own window is needed too).
        #      Newest-first within each class.
        #   4. never-finalized windows LAST — _build_bundle rejects them, so
        #      they can never be restored and must never starve a restorable
        #      window of tmpfs space.
        #
        # The "not faulty but has no usable dump of the agreed window"
        # RuntimeError in MoevementCheckpointManager.load() stays the loud
        # backstop for any residual skew this ordering cannot cover.
        finalized = [
            (int(meta.get("window_start", -1)), meta.get("kind") == "remote")
            for meta in window_items.values()
            if meta.get("final")
        ]
        own_ws = {ws for ws, is_remote in finalized if not is_remote}
        remote_ws = {ws for ws, is_remote in finalized if is_remote}
        # Recovery agrees on ONE window world-wide, and a rank whose pair is
        # faulty must both RESTORE that window from its own dump and SERVE it
        # from its remote copy. So the first bytes to protect under drain
        # pressure are the newest window for which BOTH exist — keeping
        # newest-own and newest-remote of DIFFERENT windows can leave a rank
        # unable to satisfy any agreement (the vote then correctly, but
        # avoidably, forces a fresh start).
        matched = max(own_ws & remote_ws, default=None)
        newest_own = max(own_ws, default=None)
        newest_remote = max(remote_ws, default=None)

        def _priority(item) -> tuple[int, int, int]:
            meta = item[1]
            if not meta.get("final"):
                return (5, 0, 0)
            ws = int(meta.get("window_start", -1))
            is_remote = meta.get("kind") == "remote"
            # Third element: within one window, own before its remote copy,
            # so the order never depends on ledger insertion order.
            tie = 1 if is_remote else 0
            if matched is not None and ws == matched:
                return (0, 0, tie)
            if is_remote:
                return (3, -ws, tie) if ws == newest_remote else (4, -ws, tie)
            return (2, -ws, tie) if ws == newest_own else (4, -ws, tie)

        dumped: dict[str, Any] = {}
        for key, meta in sorted(window_items.items(), key=_priority):
            path = os.path.join(
                self.checkpoint_dir, f"rank_{self.rank}_moevement_{key}.pt"
            )
            try:
                view = states[meta["pool"]]
                # Only the used prefix (max header offset+nbytes) is ever
                # referenced by restore/serve; the clone detaches the slice
                # from the pool storage so torch.save writes used bytes,
                # not the pool's full allocation (headroom tail — and for a
                # mid-window kill, the whole unused remainder).
                used = window_used_nbytes(meta)
                pool_bytes = view.pool_bytes()[:used].clone()
                torch.save({"_pool_bytes": pool_bytes, "_meta": meta}, path)
            except Exception:
                # A failed window must cost only ITSELF: drop the partial
                # file (frees space for the remaining keys) and keep
                # dumping — the index below lists exactly the survivors.
                # Aborting here used to lose the WHOLE rank's recovery:
                # the index is written last, so one ENOSPC made even
                # fully-written .pt files invisible to load_usable_windows
                # (M7 deepseek FTFT world-wide fresh-start).
                logger.exception(
                    "Failed to dump window %s — dropping the partial file "
                    "and continuing with the remaining windows", key,
                )
                try:
                    os.unlink(path)
                except OSError:
                    pass
                continue
            dumped[key] = meta
            logger.info(
                "Dumped window %s (kind=%s, window_start=%s, %d iters, "
                "owner=%s, %d used bytes)",
                key, meta.get("kind", "window"), meta.get("window_start"),
                len(meta.get("iters", [])), meta.get("owner", self.rank),
                pool_bytes.numel(),
            )
        if log_items:
            # All upstream-log commits share the single "logs" ring pool:
            # dump its bytes ONCE with the per-iteration headers. Logs are
            # an optimization (replay falls back to live p2p), so a failure
            # here must not abort the index below either.
            logs_path = os.path.join(
                self.checkpoint_dir, f"rank_{self.rank}_moevement_logs.pt"
            )
            try:
                view = states[next(iter(log_items.values()))["pool"]]
                iters = {
                    meta["iteration"]: meta["header"]
                    for meta in log_items.values()
                }
                torch.save(
                    {"_pool_bytes": view.pool_bytes(), "_iters": iters},
                    logs_path,
                )
                logger.info(
                    "Dumped upstream-log ring: %d committed iteration(s) %s",
                    len(iters), sorted(iters),
                )
            except Exception:
                logger.exception(
                    "Failed to dump the upstream-log ring — dropping the "
                    "partial file and continuing",
                )
                try:
                    os.unlink(logs_path)
                except OSError:
                    pass
        # The metadata index is written LAST and lists ONLY the windows
        # whose .pt landed completely: a crash mid-dump leaves no (or a
        # stale) index, so recovery never sees a key whose .pt is partial.
        # Log keys are deliberately NOT indexed (the index drives window
        # discovery in find_committed_windows; logs are discovered by their
        # own dump file).
        index = {
            key: {
                "window_start": meta["window_start"],
                "iters": [entry["step"] for entry in meta["iters"]],
                # "window" = this rank's own window; "remote" = a replicated
                # copy of the pair rank's window (plan §3.7). Recovery
                # discovery filters on this (load_usable_windows).
                "kind": meta.get("kind", "window"),
            }
            for key, meta in dumped.items()
        }
        try:
            with open(
                os.path.join(
                    self.checkpoint_dir,
                    f"rank_{self.rank}_moevement_metadata.json",
                ),
                "w",
            ) as f:
                json.dump(index, f)
            logger.info("Wrote moevement metadata index: %s", index)
        except Exception:
            logger.exception(
                "Failed to write the moevement metadata index — this "
                "rank's dumps will NOT be discovered by recovery",
            )
        if len(dumped) != len(window_items):
            logger.error(
                "Dumped only %d/%d committed windows; lost: %s",
                len(dumped), len(window_items),
                sorted(set(window_items) - set(dumped)),
            )


class _CpuEvent:
    """Immediately-fired event for the CPU transfer backend."""

    def synchronize(self) -> None:
        pass

    def query(self) -> bool:
        return True


class CudaTransferBackend:
    """Dedicated side-stream D2H into pinned shm pools (the reference's
    transfer scheduling, retargeted at container-attached pools)."""

    def __init__(self, device: torch.device | None = None):
        self.device = (
            device
            if device is not None
            else torch.device("cuda", torch.cuda.current_device())
        )
        self.stream = torch.cuda.Stream(self.device)

    def alloc_pool(self, path: str, nbytes: int) -> torch.UntypedStorage:
        storage, timings = pool_shm.alloc_and_pin(path, nbytes)
        logger.info(
            "[moevement] pool %s: %.2fMB alloc=%.1fms prefault=%.1fms pin=%.1fms",
            os.path.basename(path), nbytes / (1024 * 1024),
            timings["alloc"], timings["prefault"], timings["pin"],
        )
        return storage

    def capture_context(self):
        # The source tensors are written on the compute stream; order the
        # side stream after everything submitted so far, then run the packs
        # and D2H copies on the side stream.
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        return torch.cuda.stream(self.stream)

    def copy_run(self, sources: list[torch.Tensor], dst_view: torch.Tensor):
        # GPU staging flat so the whole run hits PCIe as ONE async D2H.
        staging = torch.empty(
            dst_view.numel(), dtype=dst_view.dtype, device=self.device
        )
        offset = 0
        for src in sources:
            n = src.numel()
            staging[offset : offset + n].copy_(src.reshape(-1))
            offset += n
        dst_view.copy_(staging, non_blocking=True)
        # Returned so the committer holds the staging alive until the D2H
        # event confirms the drain.
        return staging

    def record_event(self):
        event = torch.cuda.Event()
        event.record(self.stream)
        return event

    def wait_event_on_current_stream(self, event) -> None:
        torch.cuda.current_stream(self.device).wait_event(event)

    def alloc_host(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        """Pinned host staging (upstream logger's pre-pool iteration)."""
        return torch.empty(numel, dtype=dtype, pin_memory=True)

    def record_event_on_current_stream(self):
        """Event on the CURRENT stream (upstream-logger tee copies are
        compute-stream-ordered; see upstream_logger.log_recv)."""
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        return event

    def full_sync(self) -> None:
        """Device-wide quiesce fence (mid-training pool pinning)."""
        torch.cuda.synchronize(self.device)

    def capture_rng(self) -> dict[str, torch.Tensor]:
        return {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(self.device),
        }


class CpuTransferBackend:
    """Synchronous seam for CPU-only tests: same call surface as
    CudaTransferBackend, immediate copies, always-fired events, unpinned
    pools."""

    device = torch.device("cpu")

    def alloc_pool(self, path: str, nbytes: int) -> torch.UntypedStorage:
        if os.path.lexists(path):
            os.unlink(path)
        return torch.UntypedStorage.from_file(path, shared=True, nbytes=nbytes)

    def capture_context(self):
        return nullcontext()

    def copy_run(self, sources: list[torch.Tensor], dst_view: torch.Tensor):
        offset = 0
        for src in sources:
            n = src.numel()
            dst_view[offset : offset + n].copy_(src.reshape(-1))
            offset += n
        return None

    def record_event(self):
        return _CpuEvent()

    def wait_event_on_current_stream(self, event) -> None:
        pass

    def alloc_host(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(numel, dtype=dtype)

    def record_event_on_current_stream(self):
        return _CpuEvent()

    def full_sync(self) -> None:
        pass

    def capture_rng(self) -> dict[str, torch.Tensor]:
        return {"torch_cpu": torch.get_rng_state()}


# ---------------------------------------------------------------------------
# B_PCIe profiling (Algorithm 1's bandwidth input)
# ---------------------------------------------------------------------------
#
# The paper PROFILES the device->host bandwidth that its window-sizing
# budget is denominated in; the reference left it a config constant
# (MoEvementConfig.pcie_bandwidth_bytes_per_sec, whose own docstring says
# "re-tune ... if measured"). Measuring it costs ~130 ms once per process and
# is strictly better than a guess: it folds in the real link width and the
# host's pinned-page behavior.
#
# CAVEAT, measured on this cluster (docs/moevement_port_plan.md §9-M9):
# quiescent, all 8 GPUs report 24.2-24.3 GiB/s (48.4 GiB/s aggregate per
# host — the two links are independent), but run INSIDE lazy_init the same
# probe returns 14.7-24.1 GiB/s depending on what other init work is in
# flight. The reading is therefore a snapshot of init-time conditions, not a
# constant. That is safe by direction: a low reading proposes a LARGER
# w_sparse, and the manager pins the world all_reduce(MAX) of the proposals,
# so the most pessimistic rank sets the cadence.

_D2H_PROBE_NBYTES = 256 * 1024 * 1024
_D2H_PROBE_REPS = 5
# Per-process cache: the probe is a property of the box, not of the caller.
_measured_d2h_gbs: float | None = None


def measure_d2h_bandwidth_gbs(
    backend=None,
    nbytes: int = _D2H_PROBE_NBYTES,
    reps: int = _D2H_PROBE_REPS,
    use_cache: bool = True,
) -> float:
    """Profile effective device->host bandwidth in GiB/s (Algorithm 1's
    B_PCIe input).

    Runs the ENGINE's own transfer path rather than a synthetic best case:
    the side ("capture") stream ordered after compute by capture_context(),
    a pinned host destination of exactly the kind alloc_host() hands out,
    and one flat async D2H per rep — so the number reflects what the
    snapshot engine can actually sustain. One warm rep is discarded (first
    touch of the pinned buffer, stream/context setup), then ``reps`` timed
    reps; the MEDIAN is returned so a single scheduling hiccup cannot skew
    the budget. Both probe buffers are freed before returning.

    Cached per process (``use_cache``) — every rank measures its own device
    once, at init, off the training critical path.
    """
    global _measured_d2h_gbs
    if use_cache and _measured_d2h_gbs is not None:
        return _measured_d2h_gbs
    if backend is None:
        backend = (
            CudaTransferBackend()
            if torch.cuda.is_available()
            else CpuTransferBackend()
        )
    src = torch.empty(nbytes, dtype=torch.uint8, device=backend.device)
    dst = backend.alloc_host(nbytes, torch.uint8)
    samples: list[float] = []
    try:
        for rep in range(reps + 1):
            t0 = time.perf_counter()
            with backend.capture_context():
                dst.copy_(src, non_blocking=True)
                event = backend.record_event()
            event.synchronize()
            elapsed = time.perf_counter() - t0
            if rep:  # rep 0 is the warm-up
                samples.append(elapsed)
    finally:
        # Free the probe buffers before training allocates anything: the
        # device staging goes back to the caching allocator and the pinned
        # host page-locked range back to the driver.
        del src, dst
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    samples.sort()
    median = max(samples[len(samples) // 2], 1e-9)
    gbs = nbytes / median / (1024**3)
    if use_cache:
        _measured_d2h_gbs = gbs
    return gbs


@dataclass
class _IterRecord:
    window_key: str
    pool: str
    step: int
    event: Any
    header: dict[str, Any]
    rng: dict[str, torch.Tensor]
    staging: list = field(default_factory=list)
    # Slot in the engine's persistent pinned clip-norm ring holding this
    # iteration's pre-clip total gradient norm (frozen-skip replay anchor,
    # plan §3.3), plus the source dtype so the replay rebuilds a
    # bit-identical clip coefficient. Read by the committer AFTER
    # event.synchronize(), so the trainer never syncs. None when the
    # feature is off.
    clip_norm_slot: int | None = None
    clip_norm_dtype: str | None = None


@dataclass
class _FinalRecord:
    window_key: str
    ring: dict[str, Any]
    schedule: Any


@dataclass
class _LogRecord:
    """One upstream-log iteration: committed as `log_i{step}` against the
    shared "logs" pool once its D2H event confirms."""

    step: int
    event: Any
    header: list
    holds: list = field(default_factory=list)


_STOP = object()


def _log_key(step: int) -> str:
    return f"log_i{step}"


class MoevementSnapshotEngine:
    """Owns the CURR/PREV window pools + the shared container, and runs the
    background committer implementing the append-commit + window-boundary
    barrier protocol (see module docstring)."""

    def __init__(
        self,
        scheduler: SparseCheckpointScheduler,
        mem_fs_folder: str,
        log_dir: str,
        rank: int,
        backend=None,
        container=None,
    ):
        self._rank = rank
        self._backend = backend if backend is not None else CudaTransferBackend()
        os.makedirs(mem_fs_folder, exist_ok=True)
        self.pool_nbytes = self._compute_pool_nbytes(scheduler)
        self._container = (
            container
            if container is not None
            else SnapshotContainer(
                MoevementDumpPolicy(mem_fs_folder, rank), log_dir, rank
            )
        )
        # Container calls come from both the trainer thread (invalidate at
        # the window boundary) and the committer thread (commits); the
        # container's request/ack queue pairing is not reentrant, so every
        # call is serialized under this lock.
        self._container_lock = threading.Lock()

        # DEFERRED pool allocation (M7 structural fix): lazy_init constructs
        # this engine BEFORE load() has read + swept the previous attempt's
        # dump generation, and allocating pools while up to ~4 dump files
        # per rank still sit on tmpfs overflowed the 126 GB-/dev/shm host
        # under deepseek (SIGBUS at prefault, mid-vote). ensure_storage()
        # allocates the CURR/PREV pools only after the manager's load() has
        # materialized+swept the dumps (or at the first capture / at
        # standby-promotion init), so peak tmpfs is max(dumps, pools) +
        # bundle-in-RAM, never dumps + pools.
        self._mem_fs_folder = mem_fs_folder
        self._pools: list[torch.UntypedStorage] = []
        # Persistent pinned ring for the per-iteration clip-norm anchors
        # (frozen-skip, plan §3.3): allocated ONCE behind ensure_storage's
        # full-sync fence, never per step (the M4 caveat forbids unfenced
        # mid-training host pinning, and a per-step pinned alloc would be
        # pure overhead). float64 so any source dtype round-trips exactly.
        self._clip_norm_ring: torch.Tensor | None = None
        self._clip_norm_cursor = 0

        self._curr_pool = 0
        # Committed window key whose bytes a pool still holds (None while the
        # pool is being filled / holds nothing durable).
        self._pool_held_key: list[str | None] = [None, None]
        self._window_key: str | None = None
        self._pool_offset = 0
        self._last_final_key: str | None = None
        self._final_events: dict[str, threading.Event] = {}
        # Steps whose log commit has landed (committer adds, trainer-thread
        # eviction consumes; GIL-atomic set ops).
        self._committed_log_steps: set[int] = set()
        self._pending_event = None
        self._moment_cache: dict[int, dict] = {}
        # Window replicator (plan §3.7): attached by the manager after
        # construction; the committer forwards every final-committed window
        # to it and the boundary barrier waits for its sends (see
        # attach_replicator / _begin_window).
        self._replicator = None
        self._committer_error: BaseException | None = None
        self._queue: queue.Queue = queue.Queue()
        self._committer = threading.Thread(
            target=self._committer_main,
            name=f"moevement-committer-r{rank}",
            daemon=True,
        )
        self._committer.start()
        self._closed = False

    @staticmethod
    def _compute_pool_nbytes(scheduler: SparseCheckpointScheduler) -> int:
        # max_window_bytes is the monotone regen-proof bound (plan §8-R4);
        # the per-iter allowance is priced at the worst-case window length
        # (w_sparse <= total_ops since num_active >= 1, or the override).
        base = scheduler.max_window_bytes()
        # The pinned cadence is either the config override or the manager's
        # world-aligned Algorithm-1 decision (pinned BEFORE this engine is
        # constructed); 0 only while the cadence is genuinely free, where the
        # worst case is w_sparse == total_ops.
        pinned = scheduler.pinned_w_sparse()
        if pinned > 0:
            max_iters = pinned
        else:
            max_iters = max(1, len(scheduler.ops))
        nbytes = int((base + max_iters * _PER_ITER_ALLOWANCE) * _HEADROOM)
        return (nbytes + 4095) // 4096 * 4096

    def ensure_storage(self) -> None:
        """Idempotent deferred allocation of the two window pools + their
        container registration (M7 structural fix; see __init__).

        Trainer thread only. Callers: the manager — inside load() AFTER the
        dump generation has been fully read into process RAM and swept, at
        the first save() for runs that never load, and at standby-promotion
        init (notify_rmp_restored) — plus capture_iteration /
        recommit_window as engine-level backstops. The synchronous
        REGISTER->unlink kill-survivability protocol is unchanged, merely
        later. full_sync() first: cudaHostRegister outside a quiesced device
        corrupted numerics on this cluster (M4 hardware caveat) — on the
        load()/init-time paths the fence is a cheap no-op.
        """
        if self._pools:
            return
        self._backend.full_sync()
        if self._clip_norm_ring is None:
            self._clip_norm_ring = self._backend.alloc_host(
                _CLIP_NORM_RING, torch.float64
            )
            self._clip_norm_ring.zero_()
        for idx in range(2):
            # gemini's register->unlink pattern (in_mem_state.py): PID in the
            # name so active and standby groups on one host never collide;
            # the container attaches inside the synchronous register, after
            # which the unlinked pool is an anonymous kernel-refcounted
            # mapping that survives SIGKILL of this process.
            path = os.path.join(
                self._mem_fs_folder,
                f"moevement_pool_rank{self._rank}_v{idx}_pid{os.getpid()}",
            )
            storage = self._backend.alloc_pool(path, self.pool_nbytes)
            with self._container_lock:
                self._container.register(
                    _pool_name(idx), None, (path, self.pool_nbytes)
                )
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            self._pools.append(storage)

    # ------------------------------------------------------------------
    # Capture path (trainer thread)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def capture_iteration(
        self,
        step: int,
        slot: CheckpointSchedule,
        operators_by_name: dict[str, Operator],
        optimizers,
        clip_total_norm: torch.Tensor | None = None,
    ):
        """Capture one iteration's scheduled subset; returns the D2H event.

        ``clip_total_norm`` (when frozen-skip capture is on) is the
        iteration's pre-clip total gradient norm; it rides the same capture
        stream + event as the weight bytes, so reading it costs the trainer
        no synchronization — the committer picks it up from pinned host
        memory once the event has fired.
        """
        self._raise_committer_error()
        self.ensure_storage()
        if self._window_key is None:
            self._begin_window(step)
        header: dict[str, Any] = {}
        staging: list = []
        clip_norm_slot: int | None = None
        clip_norm_dtype: str | None = None
        with self._backend.capture_context():
            for name in slot.active:
                entries = self._active_entries(operators_by_name[name], optimizers)
                self._write_runs(name, True, entries, header, staging)
            for name in slot.frozen:
                entries = self._frozen_entries(operators_by_name[name])
                self._write_runs(name, False, entries, header, staging)
            if clip_total_norm is not None:
                source = clip_total_norm.detach().reshape(1)
                clip_norm_slot = self._clip_norm_cursor % _CLIP_NORM_RING
                self._clip_norm_cursor += 1
                clip_norm_dtype = str(source.dtype).removeprefix("torch.")
                self._clip_norm_ring[
                    clip_norm_slot : clip_norm_slot + 1
                ].copy_(source, non_blocking=True)
                # The async D2H reads `source` on the CAPTURE stream while the
                # allocator only tracks the compute stream that produced it —
                # dropping the last Python reference here would let a later
                # compute-stream allocation reuse the block before the copy
                # runs (observed: a replayed step's pinned norm read back as
                # 0.0, so the clip coefficient silently became 1.0). Hold it
                # in `staging`, which the committer clears only after
                # event.synchronize() — the same contract the GPU staging
                # flats use.
                staging.append(source)
        event = self._backend.record_event()
        rng = self._backend.capture_rng()
        self._pending_event = event
        self._queue.put(
            _IterRecord(
                window_key=self._window_key,
                pool=_pool_name(self._curr_pool),
                step=step,
                event=event,
                header=header,
                rng=rng,
                staging=staging,
                clip_norm_slot=clip_norm_slot,
                clip_norm_dtype=clip_norm_dtype,
            )
        )
        return event

    def attach_replicator(self, replicator) -> None:
        """Arm window replication (plan §3.7). Must be called before training
        steps (the committer reads the attribute without a lock)."""
        self._replicator = replicator

    def pool_storage(self, idx: int) -> torch.UntypedStorage:
        assert self._pools, (
            "[moevement] window pools not allocated yet — ensure_storage() "
            "must run before any pool access"
        )
        return self._pools[idx]

    def register_extra_pool(self, state_key: str, path: str, nbytes: int) -> None:
        """Register an additional shm pool (e.g. the replication 'remote'
        slots) with the shared container."""
        with self._container_lock:
            self._container.register(state_key, None, (path, nbytes))

    def commit_key(self, key: str, meta: dict) -> None:
        """Container commit passthrough for non-committer threads (the
        replication receiver); serialized on the container lock."""
        with self._container_lock:
            self._container.commit(key, meta)

    def invalidate_key(self, key: str) -> None:
        with self._container_lock:
            self._container.invalidate(key)

    @torch.no_grad()
    def recommit_window(
        self, meta: dict[str, Any], pool_bytes: torch.Tensor, used_nbytes: int
    ) -> str:
        """Re-commit a just-restored window's bytes + metadata into the
        fresh pools/container (M6 hardening, plan §9-M5 carry-forward (a)).

        Called by the manager's load() right after a successful restore
        (local dump or peer fetch), BEFORE any capture: copies the used
        pool prefix into pool0 (header offsets are absolute pool offsets,
        preserved verbatim), commits the normalized metadata under the
        window's own key, and marks pool0 held-with-fired-final so the
        normal double-buffer protocol takes over — the first fresh window
        fills pool1, and only the SECOND fresh window's begin barrier
        invalidates the restored copy (i.e. once a fresh finalized window
        is durable). A second fault before then therefore dumps and
        recovers this window again instead of fresh-starting. Trainer
        thread only; the committer is idle at this point.
        """
        self._raise_committer_error()
        self.ensure_storage()
        assert self._window_key is None and self._pool_held_key == [None, None], (
            "recommit_window must run on a fresh engine, before any capture"
        )
        window_start = int(meta["window_start"])
        key = f"w{window_start}"
        if used_nbytes > self.pool_nbytes:
            raise RuntimeError(
                f"[moevement] restored window {key} uses {used_nbytes}B but "
                f"the fresh pool holds {self.pool_nbytes}B — schedule/pool "
                f"sizing changed across the restart"
            )
        dst_idx = 0
        if used_nbytes > 0:
            dst = torch.empty(0, dtype=torch.uint8)
            dst.set_(
                source=self._pools[dst_idx],
                storage_offset=0,
                size=(used_nbytes,),
            )
            src = pool_bytes.view(torch.uint8).reshape(-1)[:used_nbytes]
            dst.copy_(src)
        meta = dict(meta)
        meta["pool"] = _pool_name(dst_idx)
        # A peer-fetched bundle carries the pair's remote-copy tagging;
        # normalized so the index lists it as this rank's OWN window.
        meta.pop("kind", None)
        meta.pop("owner", None)
        with self._container_lock:
            self._container.commit(key, meta)
        evt = threading.Event()
        evt.set()  # already durable — the begin barrier must not wait
        self._final_events[key] = evt
        self._pool_held_key[dst_idx] = key
        self._last_final_key = key
        self._curr_pool = 1 - dst_idx
        self._window_key = None
        self._pool_offset = 0
        if self._replicator is not None:
            # Refill the pair's remote slot too (both sides' remote pools
            # are fresh after the relaunch): one-shot ship through the
            # normal sender thread / receiver protocol.
            self._replicator.replicate_window(key, dst_idx, meta, used_nbytes)
        logger.info(
            "[moevement] re-committed restored window %s (%.2f MB) into "
            "fresh pool%d%s",
            key, used_nbytes / (1024 * 1024), dst_idx,
            "; re-replicating to pair" if self._replicator is not None else "",
        )
        return key

    def _begin_window(self, step: int) -> None:
        held = self._pool_held_key[self._curr_pool]
        if held is not None:
            # WINDOW-BOUNDARY BARRIER (plan §3.4): this pool still holds
            # finalized window W-1 and is about to be overwritten by W+1.
            # Invalidate-before-write is mandatory (torn-write safety), but
            # invalidating before W's final commit is durable would leave a
            # SIGKILL in that interval with no complete window anywhere. So
            # block — normally ~0 — until the committer confirms W's final
            # commit, and only then drop W-1.
            assert self._last_final_key is not None
            # Poll in short ticks: a dying committer may never set an event
            # inserted concurrently with its error path — re-checking the
            # error each tick surfaces that in ~1s instead of stalling for
            # the full timeout.
            evt = self._final_events[self._last_final_key]
            deadline = time.monotonic() + _FINAL_COMMIT_TIMEOUT_S
            while not evt.wait(1.0):
                self._raise_committer_error()
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"[moevement] final commit of {self._last_final_key} "
                        f"did not land within {_FINAL_COMMIT_TIMEOUT_S}s"
                    )
            self._raise_committer_error()
            if self._replicator is not None:
                # REPLICATION BARRIER (plan §3.7): this pool's bytes are the
                # send source of window `held`; overwriting them mid-send
                # would ship torn bytes to the pair. Normally ~0 — the send
                # ran during the previous window's steps.
                self._replicator.wait_sent(held, _FINAL_COMMIT_TIMEOUT_S)
            with self._container_lock:
                self._container.invalidate(held)
            self._final_events.pop(held, None)
            self._pool_held_key[self._curr_pool] = None
        self._window_key = f"w{step}"
        self._pool_offset = 0

    def _write_runs(
        self,
        op_name: str,
        is_active: bool,
        entries: list[tuple[str, torch.Tensor]],
        header: dict[str, Any],
        staging: list,
    ) -> None:
        tensors: dict[str, Any] = {}
        by_dtype: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
        for key, tensor in entries:
            by_dtype.setdefault(tensor.dtype, []).append((key, tensor))
        for dtype, items in by_dtype.items():
            elem_size = torch.empty(0, dtype=dtype).element_size()
            offset = (
                (self._pool_offset + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
            )
            total = sum(t.numel() for _, t in items)
            nbytes = total * elem_size
            if offset + nbytes > self.pool_nbytes:
                raise RuntimeError(
                    f"[moevement] window pool overflow at op {op_name}: "
                    f"offset {offset} + {nbytes}B > pool {self.pool_nbytes}B "
                    f"— schedule exceeded the max_window_bytes bound"
                )
            dst = torch.empty(0, dtype=dtype)
            dst.set_(
                source=self._pools[self._curr_pool],
                storage_offset=offset // elem_size,
                size=(total,),
            )
            buf = self._backend.copy_run([t for _, t in items], dst)
            if buf is not None:
                staging.append(buf)
            run_offset = offset
            for key, tensor in items:
                n = tensor.numel()
                tensors[key] = {
                    "offset": run_offset,
                    "nbytes": n * elem_size,
                    "dtype": str(dtype).removeprefix("torch."),
                    "shape": list(tensor.shape),
                }
                run_offset += n * elem_size
            self._pool_offset = offset + nbytes
        header[op_name] = {"is_active": is_active, "tensors": tensors}

    def _find_state(self, optimizers, param: torch.Tensor) -> dict:
        """Cached find_adam_state (Adam never replaces its state tensors
        after creation, so caching the state entry per param is safe; the
        manager clears the cache across a load())."""
        key = id(param)
        hit = self._moment_cache.get(key)
        if hit is not None:
            return hit
        state = find_adam_state(optimizers, param)
        self._moment_cache[key] = state
        return state

    def _active_entries(
        self, op: Operator, optimizers
    ) -> list[tuple[str, torch.Tensor]]:
        entries: list[tuple[str, torch.Tensor]] = []
        for fqn, param, expert_idx in op.param_entries:
            entries.append((f"params.{fqn}", _local_slice(param, expert_idx)))
            state = self._find_state(optimizers, param)
            entries.append(
                (
                    f"optimizer.{fqn}.exp_avg",
                    _local_slice(state["exp_avg"], expert_idx),
                )
            )
            entries.append(
                (
                    f"optimizer.{fqn}.exp_avg_sq",
                    _local_slice(state["exp_avg_sq"], expert_idx),
                )
            )
            # Adam per-param 'step' (bias-correction count): required for
            # exact adamw replay on activation. A scalar, never
            # expert-sliced — captured with every op that touches the param
            # (redundant for grouped expert weights, 4 B per entry). Rides
            # the normal pool path so no CPU sync is needed even when the
            # optimizer keeps 'step' on device (fused/capturable Adam).
            step_t = state.get("step")
            if torch.is_tensor(step_t):
                entries.append((f"optimizer.{fqn}.step", _local_slice(step_t, None)))
        for fqn, buf in op.buffer_entries:
            # fp32-consumed buffers (expert_bias) at full fp32 in both slots.
            entries.append((f"buffers.{fqn}", _local_slice(buf, None)))
        return entries

    def _frozen_entries(self, op: Operator) -> list[tuple[str, torch.Tensor]]:
        entries: list[tuple[str, torch.Tensor]] = []
        for fqn, param, expert_idx in op.param_entries:
            shard = _local_slice(param, expert_idx)
            # bf16 compute-weight capture; the cast kernel runs on the side
            # stream (inside capture_context), ordered before the D2H event.
            entries.append((f"compute_weights.{fqn}", shard.to(torch.bfloat16)))
        for fqn, buf in op.buffer_entries:
            entries.append((f"buffers.{fqn}", _local_slice(buf, None)))
        return entries

    def wait_pending_capture(self) -> None:
        """Queue a GPU-side wait on the last capture's D2H event (no CPU
        sync) — called pre-optimizer-step by the manager."""
        if self._pending_event is not None:
            self._backend.wait_event_on_current_stream(self._pending_event)
            self._pending_event = None

    # ------------------------------------------------------------------
    # Upstream-log pool (plan §3.5): same container, "logs" state key,
    # its own log_i{step} commit-key stream through the same committer.
    # ------------------------------------------------------------------

    def register_log_pool(self, path: str, nbytes: int) -> None:
        with self._container_lock:
            self._container.register("logs", None, (path, nbytes))

    def enqueue_log_iteration(
        self, step: int, event, header: list, holds: list
    ) -> None:
        self._raise_committer_error()
        self._queue.put(
            _LogRecord(step=step, event=event, header=header, holds=holds)
        )

    def invalidate_log_step(self, step: int) -> None:
        """Drop a log iteration's ledger entry before its ring slot is
        overwritten (torn-write safety). If its commit is still queued —
        possible only in a narrow first-eviction window, since eviction
        happens `capacity` iterations after the commit was enqueued — wait
        for it to land first: invalidating must never race a later commit of
        the same key, or the dump could attribute overwritten bytes to it."""
        deadline = time.monotonic() + _FINAL_COMMIT_TIMEOUT_S
        while step not in self._committed_log_steps:
            self._raise_committer_error()
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"[moevement] log commit of step {step} did not land "
                    f"within {_FINAL_COMMIT_TIMEOUT_S}s before its ring-slot "
                    f"eviction"
                )
            time.sleep(0.001)
        with self._container_lock:
            self._container.invalidate(_log_key(step))
        self._committed_log_steps.discard(step)

    def finalize_window(
        self,
        dataloader_state,
        train_state_scalars,
        lr_scheduler_state=None,
        schedule_snapshot=None,
    ) -> None:
        """Enqueue the window's final commit and swap CURR/PREV. Does not
        block: the boundary barrier in the next window's first capture waits
        for this commit before overwriting older state.

        dataloader_state/train_state_scalars/lr_scheduler_state must be the
        state as of the WINDOW'S START (before its first step): replaying
        this window's own iterations needs the start position, and the
        previous window — the other place that state could live — is
        invalidated as soon as the next window begins. The manager defers
        each boundary's captured state to the NEXT finalize call to satisfy
        this."""
        self._raise_committer_error()
        assert self._window_key is not None, (
            "finalize_window with no captured iteration in the window"
        )
        key = self._window_key
        self._final_events[key] = threading.Event()
        self._queue.put(
            _FinalRecord(
                window_key=key,
                ring={
                    "dataloader": dataloader_state,
                    "train_state": train_state_scalars,
                    "lr_scheduler": lr_scheduler_state,
                },
                schedule=schedule_snapshot,
            )
        )
        self._pool_held_key[self._curr_pool] = key
        self._last_final_key = key
        self._curr_pool = 1 - self._curr_pool
        self._window_key = None

    # ------------------------------------------------------------------
    # Committer (background thread)
    # ------------------------------------------------------------------

    def _committer_main(self) -> None:
        metadata: dict[str, dict[str, Any]] = {}
        while True:
            record = self._queue.get()
            if record is _STOP:
                break
            try:
                if isinstance(record, _IterRecord):
                    # Only event-confirmed bytes are ever committed.
                    record.event.synchronize()
                    record.staging.clear()
                    meta = metadata.setdefault(
                        record.window_key,
                        {
                            "window_start": record.step,
                            "pool": record.pool,
                            "iters": [],
                        },
                    )
                    iter_meta = {
                        "step": record.step,
                        "header": record.header,
                        "rng": record.rng,
                    }
                    if record.clip_norm_slot is not None:
                        # Event-confirmed above, so the pinned copy has
                        # landed; float() of a float32 is exact in float64,
                        # and the dtype rides along so the replay rebuilds
                        # a bit-identical clip coefficient.
                        iter_meta["clip_norm"] = float(
                            self._clip_norm_ring[record.clip_norm_slot]
                        )
                        iter_meta["clip_norm_dtype"] = record.clip_norm_dtype
                    meta["iters"].append(iter_meta)
                    with self._container_lock:
                        self._container.commit(record.window_key, meta)
                    if self._replicator is not None:
                        # Stream this iteration's event-confirmed pool span
                        # to the pair (plan §3.7; per-iteration cadence keeps
                        # the pair's copy within ~ms of our final commit).
                        pool_idx = int(record.pool.removeprefix("pool"))
                        self._replicator.enqueue_iter(
                            record.window_key, pool_idx, record.header
                        )
                elif isinstance(record, _LogRecord):
                    # Event-confirmed like iteration commits; its own
                    # commit-key stream against the shared "logs" pool.
                    record.event.synchronize()
                    record.holds.clear()
                    with self._container_lock:
                        self._container.commit(
                            _log_key(record.step),
                            {
                                "kind": "log",
                                "pool": "logs",
                                "iteration": record.step,
                                "header": record.header,
                            },
                        )
                    self._committed_log_steps.add(record.step)
                else:
                    meta = metadata.pop(record.window_key)
                    meta["final"] = True
                    meta["ring"] = record.ring
                    if record.schedule is not None:
                        meta["schedule"] = record.schedule
                    with self._container_lock:
                        self._container.commit(record.window_key, meta)
                    if self._replicator is not None:
                        # Ship the window's metadata blob (its pool spans
                        # already streamed per iteration). Enqueued AFTER the
                        # durable commit and BEFORE the final event: the
                        # boundary barrier then observes the send-tracking
                        # entry whenever it runs. put() may block on the
                        # bounded queue — pure backpressure.
                        self._replicator.enqueue_final(
                            record.window_key, meta
                        )
                    self._final_events[record.window_key].set()
            except BaseException as e:  # surface to the trainer thread
                self._committer_error = e
                logger.exception("[moevement] committer failed")
                # Unblock any barrier waiter (snapshot the values: the
                # trainer thread mutates the dict).
                for evt in list(self._final_events.values()):
                    evt.set()
                break

    def _raise_committer_error(self) -> None:
        if self._committer_error is not None:
            raise RuntimeError(
                "[moevement] snapshot committer failed"
            ) from self._committer_error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(_STOP)
        self._committer.join(timeout=_FINAL_COMMIT_TIMEOUT_S)
        with self._container_lock:
            self._container.close()
        if self._committer_error is not None:
            logger.error(
                "[moevement] committer had failed before close: %r",
                self._committer_error,
            )
