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


def _load_module():
    global _module
    if _module is not None:
        return _module
    with _load_lock:
        if _module is not None:
            return _module
        _ensure_ninja_on_path()
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        _module = load(
            name="leto_free_mem_callback",
            sources=[os.path.join(_HERE, "leto_free_mem_callback.cpp")],
            extra_include_paths=[os.path.join(cuda_home, "include")],
            extra_ldflags=[
                "-lc10_cuda",
                "-lcudart",
                f"-L{os.path.join(cuda_home, 'lib64')}",
            ],
            is_python_module=True,
            verbose=False,
        )
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
