"""Self-managed shared-memory pool for gemini's checkpoint state.

Replaces ``torch.UntypedStorage._new_using_filename_cpu`` (which ``posix_fallocate``s
every page single-threaded — the dominant cost of pool init) with:

  * ``torch.UntypedStorage.from_file(path, shared=True)`` — maps a tmpfs-backed file
    (``gemini_mem_fs_folder`` lives on ``/dev/shm``) with ``ALLOCATOR_MAPPED_SHARED``:
    a sparse ``ftruncate`` (no ``fallocate``), and NO ``torch_shm_manager`` daemon;
  * a bounded parallel ``memset`` that faults the pages across N threads
    (N = ``pinned_num_register_threads``, default 8 — the same knob that bounds
    torch's own pinned-allocator registration, so a standby's pinning can't starve
    the active's iteration);
  * a single ``cudaHostRegister`` over the whole region (page pre-faulting keeps the
    CUDA global lock held only briefly).

Cross-process sharing is by **path**: the training process shares ``(path, nbytes)``
to the snapshot_container, which ``attach``es by path. Because the container attaches
synchronously (the ``register`` RPC blocks until it has mmap'd the pool), the training
process ``unlink``s the path immediately after — the mapping then becomes an anonymous,
kernel-refcounted region that is freed automatically when the training process and the
container both exit, even on SIGKILL (matching ``torch_shm_manager`` cleanup without a
daemon). A crash in the brief create->register window leaves a named file that
``gemini_mem_fs_folder`` cleanup sweeps on next start / fatal restart.
"""

import ctypes
import ctypes.util
import os
import threading

import torch
from torch.cuda._pin_memory_utils import pin_memory

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
_libc.memset.restype = ctypes.c_void_p

_DEFAULT_PIN_THREADS = 8
_MIN_PARALLEL_BYTES = 1 << 20  # below this the fan-out overhead isn't worth it


def pin_num_threads(default: int = _DEFAULT_PIN_THREADS) -> int:
    """Threads to fan the page prefault across. Read from
    ``pinned_num_register_threads`` in ``PYTORCH_ALLOC_CONF`` /
    ``PYTORCH_CUDA_ALLOC_CONF`` so one knob bounds both torch's own pinned-allocator
    registration and gemini's pool pinning."""
    for var in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        for kv in os.environ.get(var, "").split(","):
            key, sep, val = kv.partition(":")
            if sep and key.strip() == "pinned_num_register_threads":
                try:
                    return max(1, int(val.strip()))
                except ValueError:
                    return default
    return default


def _parallel_prefault(ptr: int, nbytes: int, num_threads: int) -> None:
    """Fault in (allocate) every page by writing zeros, fanned across ``num_threads``
    threads. ``ctypes`` releases the GIL for the ``memset`` foreign call, so the threads
    fault pages in parallel. Writing zeros is safe: the pool is a freshly-created sparse
    file (all zero) prefaulted before any checkpoint data is copied in."""
    if num_threads <= 1 or nbytes < _MIN_PARALLEL_BYTES:
        _libc.memset(ptr, 0, nbytes)
        return
    threads = []
    for i in range(num_threads):
        start = (nbytes * i) // num_threads
        end = (nbytes * (i + 1)) // num_threads
        if end > start:
            threads.append(
                threading.Thread(target=_libc.memset, args=(ptr + start, 0, end - start))
            )
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def alloc_and_pin(path: str, nbytes: int, num_threads: int | None = None):
    """Create a fresh tmpfs-backed shm pool at ``path`` (sparse — no ``posix_fallocate``),
    parallel-prefault its pages, and pin it with a single ``cudaHostRegister``.

    Returns ``(storage, timings)`` where ``storage`` is the ``UntypedStorage`` that owns
    the mapping and ``timings`` is ``{"alloc", "prefault", "pin", "threads"}`` (ms).
    The caller shares ``path`` to the container and unlinks it after the container
    attaches (see ``attach``)."""
    import time

    if num_threads is None:
        num_threads = pin_num_threads()

    # Drop any stale file at this deterministic path (e.g. a leftover from a crashed
    # generation). A live holder keeps its own inode alive across this unlink.
    if os.path.lexists(path):
        os.unlink(path)

    # Fail fast on insufficient tmpfs space. Unlike posix_fallocate (which the old
    # _new_using_filename_cpu used and which fails with a catchable ENOSPC), a sparse
    # file's write-fault SIGBUSes when the backing tmpfs is full — an uncatchable hard
    # crash. Checking free space up front restores graceful failure. (A small race
    # remains if another process fills the tmpfs between the check and the prefault; the
    # broker margins and this being a one-shot init make that negligible.)
    st = os.statvfs(os.path.dirname(path) or ".")
    avail = st.f_bavail * st.f_frsize
    if avail < nbytes:
        raise OSError(
            f"insufficient shm space for gemini pool: need {nbytes / 2**30:.1f}GiB, "
            f"{avail / 2**30:.1f}GiB free on {os.path.dirname(path)}. With enable_standby "
            f"the active and standby groups each allocate the full pool set; reduce model "
            f"size or free /dev/shm."
        )

    t0 = time.perf_counter()
    storage = torch.UntypedStorage.from_file(path, shared=True, nbytes=nbytes)
    t1 = time.perf_counter()
    _parallel_prefault(storage.data_ptr(), nbytes, num_threads)
    t2 = time.perf_counter()
    pin_memory(storage.data_ptr(), storage.nbytes())
    t3 = time.perf_counter()

    timings = {
        "alloc": (t1 - t0) * 1000.0,
        "prefault": (t2 - t1) * 1000.0,
        "pin": (t3 - t2) * 1000.0,
        "threads": num_threads,
    }
    return storage, timings


def attach(path: str, nbytes: int):
    """Attach (map) an existing shm pool by path — no prefault, no pin. Used by the
    snapshot_container subprocess to read the pool for checkpoint dumps."""
    return torch.UntypedStorage.from_file(path, shared=True, nbytes=nbytes)
