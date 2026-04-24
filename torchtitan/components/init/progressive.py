"""Progressive standby init — shared-memory advance signal.

At end of each training iteration, the active rank writes its current
`cudaMemGetInfo` free-MiB (int32) into `/dev/shm/leto_progressive_rank_{R}`.
The paired standby rank polls that file between init tasks, decides locally
whether the next task fits its memory budget, and all-reduces that decision
(MIN) across standby ranks over a CPU-only gloo PG. Tasks only advance when
all ranks agree.
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import struct
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


_PREFIX_ENV = "LETO_PROGRESSIVE_SHM_PREFIX"
_INT32 = struct.Struct("<i")
_mmap_cache: dict[int, mmap.mmap] = {}


def _shm_path(rank: int) -> str:
    prefix = os.environ.get(_PREFIX_ENV)
    if not prefix:
        raise RuntimeError(
            f"{_PREFIX_ENV} not set — worker controller did not initialize "
            f"progressive shm (is enable_standby on?)"
        )
    return f"{prefix}{rank}"


def _get_mmap(rank: int) -> mmap.mmap:
    mm = _mmap_cache.get(rank)
    if mm is not None:
        return mm
    path = _shm_path(rank)
    fd = os.open(path, os.O_RDWR)
    try:
        mm = mmap.mmap(fd, 4, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)
    _mmap_cache[rank] = mm
    return mm


# ---------------------------------------------------------------------------
# Active side
# ---------------------------------------------------------------------------

def write_free_mb(rank: int, free_mb: int) -> None:
    """Publish current free-GPU MiB to this rank's shm file."""
    mm = _get_mmap(rank)
    _INT32.pack_into(mm, 0, int(free_mb))


def get_free_mb() -> int:
    """Current GPU free memory in MiB via torch.cuda.mem_get_info."""
    free_b, _ = torch.cuda.mem_get_info()
    return int(free_b // (1024 * 1024))


_signal_thread: Optional[threading.Thread] = None
_signal_thread_lock = threading.Lock()


def start_signal_thread(rank: int, device: int, interval_s: float = 1.0) -> None:
    """Start a daemon thread that publishes free-GPU MiB to this rank's shm
    file every `interval_s` seconds. Idempotent: a second call is a no-op.

    Shares the process's primary CUDA context — no extra context is created
    for the thread. `device` (= LOCAL_RANK) is set thread-locally so
    mem_get_info queries the right device.
    """
    global _signal_thread
    with _signal_thread_lock:
        if _signal_thread is not None:
            return

        def _loop():
            # Per-thread current device. Primary context for `device` already
            # exists (created by main thread during activate_cuda_device), so
            # this is a thread-local pointer flip — zero GPU allocation.
            torch.cuda.set_device(device)
            while True:
                try:
                    write_free_mb(rank, get_free_mb())
                except Exception:
                    logger.exception("[progressive] signal thread error")
                time.sleep(interval_s)

        _signal_thread = threading.Thread(
            target=_loop, daemon=True, name="leto-progressive-signal"
        )
        _signal_thread.start()
        logger.info(
            f"[progressive] rank={rank} device={device} started signal thread "
            f"(interval={interval_s}s)"
        )


# ---------------------------------------------------------------------------
# Standby side
# ---------------------------------------------------------------------------

def _read_int32(rank: int) -> int:
    mm = _get_mmap(rank)
    return _INT32.unpack_from(mm, 0)[0]


def _consume_signal(rank: int) -> None:
    mm = _get_mmap(rank)
    _INT32.pack_into(mm, 0, -1)


StatusCheck = Callable[[], Optional[int]]


def wait_for_signal(
    rank: int,
    poll_interval_s: float,
    status_check: Optional[StatusCheck] = None,
    status_interval_s: float = 1.0,
) -> Tuple[str, int]:
    """Block until shm[rank] >= 0, then consume and return ("signal", free_mb).

    If status_check is provided, it is invoked every status_interval_s while
    waiting; a truthy return aborts the wait and returns ("status", code).
    """
    next_status = time.monotonic() + status_interval_s
    while True:
        v = _read_int32(rank)
        if v >= 0:
            _consume_signal(rank)
            return ("signal", v)
        if status_check is not None:
            now = time.monotonic()
            if now >= next_status:
                code = status_check()
                if code:
                    return ("status", int(code))
                next_status = now + status_interval_s
        time.sleep(poll_interval_s)


def try_advance(
    gloo_pg,
    delta_mb: float,
    safety_mb: float,
    threshold_mb: float,
    poll_interval_s: float,
    status_check: Optional[StatusCheck] = None,
) -> Tuple[str, Optional[int]]:
    """Decide, across all standby ranks, whether the next task can advance.

    Each rank reads its own delta_mb from its profile. Tasks with
    `delta_mb < threshold_mb` skip the shm wait (treated as CPU-only).
    Otherwise the rank waits for a fresh advance signal, checks locally
    whether `free_mb - delta_mb - safety_mb >= 0`, then joins a MIN
    all-reduce. The caller should loop on "retry".

    Returns:
      ("advance", None) — all ranks agree; run the task
      ("retry",   None) — at least one rank said no; poll again
      ("status",  code) — status_check tripped (ACTIVATE / TERMINATE)
    """
    rank = int(os.environ.get("RANK", "0"))

    if delta_mb < threshold_mb:
        local_ok = 1
        logger.info(f"[progressive] rank={rank} delta_mb={delta_mb:.1f} < threshold, skip wait")
    else:
        logger.info(f"[progressive] rank={rank} waiting for signal (delta_mb={delta_mb:.1f})")
        kind, val = wait_for_signal(rank, poll_interval_s, status_check)
        if kind == "status":
            logger.info(f"[progressive] rank={rank} status={val} during wait")
            return ("status", val)
        free_mb = val
        local_ok = 1 if (free_mb - delta_mb - safety_mb) >= 0 else 0
        logger.info(
            f"[progressive] rank={rank} free_mb={free_mb} - delta={delta_mb:.1f} - "
            f"safety={safety_mb:.1f} → local_ok={local_ok}"
        )

    logger.info(f"[progressive] rank={rank} entering all_reduce local_ok={local_ok}")
    t = torch.tensor([local_ok], dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=gloo_pg)
    result = "advance" if t.item() == 1 else "retry"
    logger.info(f"[progressive] rank={rank} all_reduce done → {result}")
    return (result, None)


# ---------------------------------------------------------------------------
# Schedule / profile loading
# ---------------------------------------------------------------------------

def load_solution(dump_folder: str, mode: str) -> list[str]:
    """Return task names in the solver's scheduled order."""
    path = Path(dump_folder) / "init_profile" / mode / "solution.json"
    if not path.exists():
        raise FileNotFoundError(
            f"progressive_init requires {path}. Run the profiling job first "
            f"(leto.profile_init=true) to generate it."
        )
    data = json.loads(path.read_text())
    return [t["name"] for t in data["tasks"]]


def load_rank_deltas(dump_folder: str, mode: str, rank: int) -> dict[str, float]:
    """Return {task_name: delta_mb} from this rank's profile."""
    path = Path(dump_folder) / "init_profile" / mode / f"rank_{rank}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"progressive_init requires {path}. Run the profiling job first "
            f"(leto.profile_init=true) to generate it."
        )
    data = json.loads(path.read_text())
    out: dict[str, float] = {}
    for entry in data["tasks"]:
        dm = entry.get("delta_mb")
        out[entry["task"]] = 0.0 if dm is None else float(dm)
    return out
