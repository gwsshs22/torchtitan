"""Self-managed shared-memory pool for kill-survivable checkpoint state
(shared by the gemini and moevement in-memory checkpointing methods).

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
import fcntl
import os
import threading
from contextlib import contextmanager

import torch
from torch.cuda._pin_memory_utils import pin_memory

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
_libc.memset.restype = ctypes.c_void_p

_DEFAULT_PIN_THREADS = 8
_MIN_PARALLEL_BYTES = 1 << 20  # below this the fan-out overhead isn't worth it

_ALLOC_LOCK_BASENAME = ".leto_pool_alloc.lock"
_DEFAULT_MARGIN_GB = 1.0


def _margin_bytes() -> int:
    """Safety margin the free-space guard reserves ON TOP of the requested
    pool size: headroom for NCCL's own shm segments, statvfs staleness, and
    small concurrent writers the guard cannot see. Configurable via
    ``LETO_SHM_ALLOC_MARGIN_GB`` (float GiB, default 1.0)."""
    raw = os.environ.get("LETO_SHM_ALLOC_MARGIN_GB", "")
    try:
        gb = float(raw) if raw else _DEFAULT_MARGIN_GB
    except ValueError:
        gb = _DEFAULT_MARGIN_GB
    return int(gb * 2**30)


@contextmanager
def host_alloc_lock(dirpath: str):
    """Serialize pool alloc+prefault across every process on this host.

    Without it, N concurrently-allocating local ranks each pass the statvfs
    margin check against the SAME free space and one of them SIGBUSes on
    prefault — the uncatchable sparse-file failure documented below (observed
    in the M7 deepseek FTFT matrix). fcntl.flock is kernel-released on any
    death, so a SIGKILL'd holder can never wedge later ranks (same pattern as
    the extension-build lock in components/mem/__init__.py)."""
    lock_path = os.path.join(dirpath or ".", _ALLOC_LOCK_BASENAME)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing the fd releases the flock


def _check_free_space(path: str, nbytes: int) -> None:
    """Fail fast — with a catchable OSError naming the shortfall — on
    insufficient tmpfs space. Unlike posix_fallocate (which the old
    _new_using_filename_cpu used and which fails with a catchable ENOSPC), a
    sparse file's write-fault SIGBUSes when the backing tmpfs is full — an
    uncatchable hard crash. Callers hold ``host_alloc_lock`` across
    check+prefault, so the check is race-free against other local ranks."""
    parent = os.path.dirname(path) or "."
    st = os.statvfs(parent)
    avail = st.f_bavail * st.f_frsize
    margin = _margin_bytes()
    need = nbytes + margin
    if avail < need:
        raise OSError(
            f"insufficient shm space for pool {path}: need "
            f"{nbytes / 2**30:.2f} GiB + {margin / 2**30:.2f} GiB safety "
            f"margin (LETO_SHM_ALLOC_MARGIN_GB) but only "
            f"{avail / 2**30:.2f} GiB free on {parent} — short "
            f"{(need - avail) / 2**30:.2f} GiB. Refusing to create a sparse "
            f"pool that would SIGBUS on prefault. Free /dev/shm (stale "
            f"checkpoint dumps, a concurrent standby's pool set) or reduce "
            f"model size."
        )


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

    t0 = time.perf_counter()
    # check + prefault run under the per-host flock so concurrent local
    # ranks' allocations serialize: each rank's statvfs sees the previous
    # rank's pages already faulted, making the margin check race-free.
    with host_alloc_lock(os.path.dirname(path)):
        # Drop any stale file at this deterministic path (e.g. a leftover from
        # a crashed generation). A live holder keeps its own inode alive
        # across this unlink.
        if os.path.lexists(path):
            os.unlink(path)
        _check_free_space(path, nbytes)
        storage = torch.UntypedStorage.from_file(path, shared=True, nbytes=nbytes)
        t1 = time.perf_counter()
        _parallel_prefault(storage.data_ptr(), nbytes, num_threads)
        t2 = time.perf_counter()
    # Pinning consumes no tmpfs pages — outside the lock to keep hold time
    # (and other ranks' wait) bounded by the prefault alone.
    pin_memory(storage.data_ptr(), storage.nbytes())
    t3 = time.perf_counter()

    timings = {
        "alloc": (t1 - t0) * 1000.0,
        "prefault": (t2 - t1) * 1000.0,
        "pin": (t3 - t2) * 1000.0,
        "threads": num_threads,
    }
    return storage, timings


def alloc_unpinned(path: str, nbytes: int, num_threads: int | None = None):
    """Create + prefault a tmpfs-backed shm pool WITHOUT cudaHostRegister.

    For pools that only ever see CPU writers/readers (the moevement
    replication 'remote' slots: gloo recv in, container dump out) — no CUDA
    DMA touches them, so pinning buys nothing, and skipping it sidesteps the
    M4 hardware caveat entirely (mid-training cudaHostRegister corrupted
    numerics on this cluster; these pools are also allocated at init).
    Returns the ``UntypedStorage``; same create->register->unlink sharing
    protocol as ``alloc_and_pin``."""
    if num_threads is None:
        num_threads = pin_num_threads()
    with host_alloc_lock(os.path.dirname(path)):
        if os.path.lexists(path):
            os.unlink(path)
        _check_free_space(path, nbytes)
        storage = torch.UntypedStorage.from_file(path, shared=True, nbytes=nbytes)
        _parallel_prefault(storage.data_ptr(), nbytes, num_threads)
    return storage


def attach(path: str, nbytes: int):
    """Attach (map) an existing shm pool by path — no prefault, no pin. Used by the
    snapshot_container subprocess to read the pool for checkpoint dumps."""
    return torch.UntypedStorage.from_file(path, shared=True, nbytes=nbytes)
