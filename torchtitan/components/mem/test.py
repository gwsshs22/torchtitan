"""Smoke test for the leto OOM-safeguard install path.

Imports torchtitan.components.mem.install_oom_safeguard, registers a
counter-incrementing Python callback with a high threshold (so it always
fires), and triggers cache-miss allocations to verify the C++ -> Python
bridge fires and the per-allocation cycle works end-to-end.

Run with:
    PATH=$VENV/bin:$PATH CUDA_HOME=/usr/local/cuda \
        python -m torchtitan.components.mem.test
"""

import torch

from torchtitan.components.mem import install_oom_safeguard

counter = {"count": 0}


def my_callback() -> bool:
    counter["count"] += 1
    print(f"[python callback] fired, count={counter['count']}", flush=True)
    return True


install_oom_safeguard(threshold_mb=999_999_999, kill_callback=my_callback)

assert torch.cuda.is_available(), "need a CUDA device to exercise the callback"
torch.cuda.init()

print("=== alloc 1: small, fresh cache ===", flush=True)
a = torch.empty(1024, device="cuda")

print("=== alloc 3: larger new size ===", flush=True)
c = torch.empty(64 * 1024 * 1024, device="cuda")  # 256 MiB float32
del c

print("=== alloc 2: new size ===", flush=True)
b = torch.empty(16 * 1024 * 1024, device="cuda")  # 64 MiB float32

del a, b
torch.cuda.empty_cache()
print(f"=== done; python callback fired {counter['count']} times ===", flush=True)
