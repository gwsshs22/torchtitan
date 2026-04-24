"""Smoke test for leto_free_mem_callback.

Builds leto_free_mem_callback.cpp as a shared lib via torch.utils.cpp_extension,
loads it so its static initializer registers the callback into
c10::FreeCudaMemoryCallbacksRegistry, then triggers CUDA allocations that
cache-miss so PyTorch calls our callback and prints "[leto] free=..." lines.
"""

import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
CUDA_HOME = os.environ["CUDA_HOME"]

# is_python_module=False builds a plain .so and dlopens it (triggering our
# static REGISTER_FREE_MEMORY_CALLBACK) without requiring a PyInit entry point.
load(
    name="leto_free_mem_callback",
    sources=[os.path.join(HERE, "leto_free_mem_callback.cpp")],
    extra_include_paths=[os.path.join(CUDA_HOME, "include")],
    extra_ldflags=[
        "-lc10_cuda",
        "-lcudart",
        f"-L{os.path.join(CUDA_HOME, 'lib64')}",
    ],
    is_python_module=False,
    verbose=True,
)

assert torch.cuda.is_available(), "need a CUDA device to exercise the callback"
torch.cuda.init()

# Each of the allocations below should be a cache miss (first-time size on a
# fresh pool / fresh stream), which triggers trigger_free_memory_callbacks and
# therefore our Execute(). Expect one "[leto] free=..." line per alloc.
print("=== alloc 1: small, fresh cache ===", flush=True)
a = torch.empty(1024, device="cuda")


print("=== alloc 3: larger new size ===", flush=True)
c = torch.empty(64 * 1024 * 1024, device="cuda")  # 256 MiB float32
del c

print("=== alloc 2: new size ===", flush=True)
b = torch.empty(16 * 1024 * 1024, device="cuda")  # 64 MiB float32


# Keep tensors alive until here so their frees happen after the prints.
del a, b
torch.cuda.empty_cache()
print("=== done ===", flush=True)
