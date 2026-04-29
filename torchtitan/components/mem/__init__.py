"""Leto OOM-safeguard installation for torchtitan training processes.

`install_oom_safeguard(threshold_mb, kill_callback)` builds the
`leto_free_mem_callback` C++ extension on first call (lazy + memoized),
registers a Python kill_callback that gets invoked synchronously by the
CUDACachingAllocator's FreeMemoryCallback whenever free GPU MiB falls
below `threshold_mb` during a cache-miss expansion, and registers an
atexit hook that clears the callback before interpreter teardown so the
static `py::object` in the C++ module never decrefs after Python is
finalized.
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


def install_oom_safeguard(
    threshold_mb: int, kill_callback: Callable[[], bool]
) -> None:
    """Install the OOM safeguard. No-op if threshold_mb <= 0.

    Args:
        threshold_mb: Free GPU MiB below which the kill_callback fires.
        kill_callback: Zero-arg callable returning True if memory was freed
            (so PyTorch retries the allocation), False otherwise.
    """
    global _atexit_registered
    if threshold_mb <= 0:
        return
    m = _load_module()
    m.set_threshold_mb(int(threshold_mb))
    m.set_kill_callback(kill_callback)
    if not _atexit_registered:
        atexit.register(m.clear_kill_callback)
        _atexit_registered = True


def set_threshold_mb(mb: int) -> None:
    """Update the OOM-safeguard threshold. Pass 0 to disable firing
    without unregistering the callback (e.g. once the first kill has
    been observed and we want subsequent allocations to surface OOM
    cleanly)."""
    if _module is None:
        return
    _module.set_threshold_mb(int(mb))


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
