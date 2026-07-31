"""Compatibility shim: pool_shm moved to torchtitan.components.snapshot."""

from torchtitan.components.snapshot.pool_shm import (  # noqa: F401
    alloc_and_pin,
    attach,
    pin_num_threads,
    _parallel_prefault,
    _DEFAULT_PIN_THREADS,
    _MIN_PARALLEL_BYTES,
)
