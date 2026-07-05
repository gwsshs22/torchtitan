"""Leto standby memory reservation for torchtitan training processes.

`install_reservation_broker(ledger_path, margin_mb, kill_callback)` builds
the `leto_free_mem_callback` C++ extension on first call (lazy + memoized),
installs the RecordingAllocator (a pass-through wrapper that records each
in-flight allocation's size), mmaps the per-rank shm ledger shared with the
co-located standby, and starts the broker thread.

The CUDACachingAllocator's FreeMemoryCallback (Execute) fires on every
cache miss: it takes one NVML read, publishes `used_estimate`
(= physical used excl. standby + in-flight bytes + margin) for the broker's
grant decisions, and reclaims (kills) the standby when even releasing the
active's whole cache could not keep the pending allocations out of the
standby's reservation. The broker and the callback share only three plain
atomics — no locks touch the allocation path.
"""

import atexit
import fcntl
import importlib.util
import os
import shutil
import sys
import threading
from typing import Callable, Optional

from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_load_lock = threading.Lock()
_module = None  # type: Optional[object]
_atexit_registered = False


def _ensure_ninja_on_path() -> None:
    """torch.utils.cpp_extension.load shells out to `ninja`. Under torchrun
    the spawned processes don't inherit a PATH that includes the venv's
    bin/, so prepend it if ninja lives there."""
    if shutil.which("ninja") is not None:
        return
    venv_bin = os.path.join(sys.prefix, "bin")
    if os.path.isfile(os.path.join(venv_bin, "ninja")):
        os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"


def _try_load_prebuilt(build_dir: str, src_path: str) -> Optional[object]:
    """Fast path: dlopen the existing .so directly, bypassing
    cpp_extension.load. Critical because cpp_extension's FileBaton uses
    O_EXCL file-existence as its mutex and is *not* released on process
    death — a SIGKILL'd builder leaves a corpse lock that wedges every
    future rank in FileBaton.wait() forever."""
    so_path = os.path.join(build_dir, "leto_free_mem_callback.so")
    try:
        if not os.path.isfile(so_path):
            return None
        if os.path.getmtime(so_path) < os.path.getmtime(src_path):
            return None  # source modified since last build → must rebuild
    except OSError:
        return None
    spec = importlib.util.spec_from_file_location(
        "leto_free_mem_callback", so_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_with_fcntl_lock(build_dir: str, src_path: str) -> object:
    """Cross-process serialized build using fcntl.flock — kernel-released
    on process death, unlike cpp_extension's FileBaton which leaves a
    corpse file that wedges all future ranks. We hold our own flock for
    the entire critical section: re-check fast path (someone else may
    have built while we waited), nuke any stale FileBaton corpse, then
    call cpp_extension.load. Inside the flock we are the only rank
    touching the build dir, so cpp_extension's FileBaton sees no
    contention."""
    coord_lock_path = os.path.join(build_dir, "leto_build.lock")
    os.makedirs(build_dir, exist_ok=True)
    fd = os.open(coord_lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        prebuilt = _try_load_prebuilt(build_dir, src_path)
        if prebuilt is not None:
            return prebuilt
        # Nuke any stale FileBaton corpse (from a SIGKILL'd prior build).
        # Safe here because our flock ensures no other rank is racing.
        try:
            os.remove(os.path.join(build_dir, "lock"))
        except FileNotFoundError:
            pass
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        return load(
            name="leto_free_mem_callback",
            sources=[src_path],
            extra_include_paths=[os.path.join(cuda_home, "include")],
            extra_ldflags=[
                "-lc10",
                "-lc10_cuda",
                "-lcudart",
                "-lnvidia-ml",
                f"-L{os.path.join(cuda_home, 'lib64')}",
                f"-L{os.path.join(cuda_home, 'lib64', 'stubs')}",
            ],
            is_python_module=True,
            verbose=False,
        )
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _load_module():
    global _module
    if _module is not None:
        return _module
    with _load_lock:
        if _module is not None:
            return _module
        _ensure_ninja_on_path()

        from torch.utils.cpp_extension import _get_build_directory
        build_dir = _get_build_directory(
            "leto_free_mem_callback", verbose=False)
        src_path = os.path.join(_HERE, "leto_free_mem_callback.cpp")

        prebuilt = _try_load_prebuilt(build_dir, src_path)
        if prebuilt is not None:
            _module = prebuilt
            return _module

        _module = _build_with_fcntl_lock(build_dir, src_path)
        return _module


def install_recording_allocator() -> bool:
    """Install the transparent RecordingAllocator as the current CUDA
    allocator so the FreeMemoryCallback can read the in-flight allocation
    size. Idempotent; safe to call at any point in the process. Returns
    False if no CUDA backend allocator exists yet."""
    return bool(_load_module().install_recording_allocator())


def is_recording_allocator_installed() -> bool:
    if _module is None:
        return False
    return bool(_module.is_recording_allocator_installed())


def get_num_kill_standby_called() -> int:
    """Number of times the C++ FreeMemoryCallback has invoked the
    Python kill callback since process start (or since
    reset_num_kill_standby_called)."""
    if _module is None:
        return 0
    return int(_module.get_num_kill_standby_called())


def reset_num_kill_standby_called() -> None:
    if _module is None:
        return
    _module.reset_num_kill_standby_called()


# ---------------------------------------------------------------------------
# Reservation broker (active side)
# ---------------------------------------------------------------------------

_reservation_atexit_registered = False


def install_reservation_broker(
    ledger_path: str,
    margin_mb: int,
    kill_callback: Optional[Callable[[], tuple]] = None,
    grant_only: bool = False,
) -> None:
    """Active-side setup for standby memory reservation.

    Installs the RecordingAllocator (which records in-flight allocation
    sizes — no locks), mmaps the per-rank shm ledger, and starts the C++
    broker thread. The broker grants/denies reservations posted by the
    co-located standby against the callback-published `used_estimate`,
    keeping `used_estimate + reserved <= capacity` (margin folded into
    the estimate).

    Args:
        ledger_path: per-rank shm file shared with the standby
            (e.g. f"{LETO_PROGRESSIVE_SHM_PREFIX}{rank}").
        margin_mb: global safety headroom (MiB) the broker keeps free.
        kill_callback: zero-arg callable returning (memory_freed, killed_pid),
            invoked by the reservation-aware FreeMemoryCallback to *reclaim*
            (kill) the standby. Pass None to run the broker in grant-only mode
            (the standby still comes up via grants, but the active never
            reclaims it) -- this is the "no safeguard" control.
    """
    global _reservation_atexit_registered
    m = _load_module()
    if not m.install_recording_allocator():
        raise RuntimeError(
            "install_reservation_broker: no CUDA backend allocator yet "
            "(call after torch.cuda is initialized)"
        )
    if not m.attach_reservation_ledger(ledger_path):
        raise RuntimeError(
            f"install_reservation_broker: attach_reservation_ledger("
            f"{ledger_path}) failed"
        )
    m.set_reservation_margin_mb(int(margin_mb))
    # Ablation (leto.progressive_protocol=grant_only): GRANT commits
    # directly against the estimate, no reservation phase.
    m.set_grant_only(bool(grant_only))
    if kill_callback is not None:
        m.set_kill_callback(kill_callback)
    if not m.start_broker(int(margin_mb)):
        raise RuntimeError("install_reservation_broker: start_broker failed")
    if not _reservation_atexit_registered:
        atexit.register(m.stop_broker)
        atexit.register(m.clear_kill_callback)
        _reservation_atexit_registered = True


def reset_granted() -> None:
    """Void the cumulative grant (e.g. after the standby is killed/activated).
    A fresh standby instance also resets it automatically via its epoch."""
    if _module is None:
        return
    _module.reset_granted()


def is_broker_running() -> bool:
    if _module is None:
        return False
    return bool(_module.is_broker_running())


def get_granted_mb() -> int:
    if _module is None:
        return 0
    return int(_module.get_granted_mb())


def get_assert_b_violations() -> int:
    """Number of times the broker observed standby_actual > granted (+tol):
    a protocol violation (a task over-allocated past its profiled delta)."""
    if _module is None:
        return 0
    return int(_module.get_assert_b_violations())


def ledger_constants() -> dict:
    """Ledger byte-layout constants (from the C++ struct). Used by the standby
    to drive the shm handshake and to assert the Python layout matches."""
    m = _load_module()
    return {
        "LEDGER_NBYTES": int(m.LEDGER_NBYTES),
        "LEDGER_MAGIC": int(m.LEDGER_MAGIC),
        "VERDICT_PENDING": int(m.VERDICT_PENDING),
        "VERDICT_DENY": int(m.VERDICT_DENY),
        "VERDICT_GRANT": int(m.VERDICT_GRANT),
        "OFF_MAGIC": int(m.OFF_MAGIC),
        "OFF_STANDBY_PID": int(m.OFF_STANDBY_PID),
        "OFF_STANDBY_EPOCH": int(m.OFF_STANDBY_EPOCH),
        "OFF_REQ_SEQ": int(m.OFF_REQ_SEQ),
        "OFF_REQ_BYTES": int(m.OFF_REQ_BYTES),
        "OFF_RESP_SEQ": int(m.OFF_RESP_SEQ),
        "OFF_RESP_VERDICT": int(m.OFF_RESP_VERDICT),
        "OFF_GRANTED": int(m.OFF_GRANTED),
        "OFF_EFFECTIVE_FREE": int(m.OFF_EFFECTIVE_FREE),
        "OFF_STANDBY_ACTUAL": int(m.OFF_STANDBY_ACTUAL),
        "OFF_REQ_TYPE": int(m.OFF_REQ_TYPE),
        "REQ_RESERVE": int(m.REQ_RESERVE),
        "REQ_GRANT": int(m.REQ_GRANT),
        "REQ_ROLLBACK": int(m.REQ_ROLLBACK),
    }
