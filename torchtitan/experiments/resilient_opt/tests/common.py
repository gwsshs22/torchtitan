"""Shared utilities for resilient optimizer tests.

Provides optimizer creation from dumped info, state snapshot/restore,
mock RMP client, and a minimal optimizer wrapper.
"""

import argparse
import ctypes
import json

import torch

_cudart = ctypes.CDLL("libcudart.so")
from torch.distributed._tensor import DTensor


DTYPE_MAP = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
}


def get_local(tensor):
    """Get underlying local tensor from DTensor, or return as-is."""
    if isinstance(tensor, DTensor):
        return tensor._local_tensor
    return tensor


def create_optimizer_from_info(
    info: dict, device: str = "cuda:0", seed: int = 42,
) -> tuple[list[torch.nn.Parameter], torch.optim.Optimizer]:
    """Recreate optimizer with realistic param shapes from dumped info."""
    torch.manual_seed(seed)

    opt_info = info["optimizers"][0]
    opt_cls = getattr(torch.optim, opt_info["class"])
    param_groups = []
    all_params = []

    for pg_info in opt_info["param_groups"]:
        group_params = []
        for p_info in pg_info["params"]:
            param = torch.nn.Parameter(
                torch.randn(
                    p_info["shape"],
                    dtype=DTYPE_MAP[p_info["dtype"]],
                    device=device,
                ),
            )
            group_params.append(param)
            all_params.append(param)

        pg_dict = {"params": group_params}
        for key in ["lr", "eps", "weight_decay", "fused", "foreach"]:
            if pg_info.get(key) is not None:
                pg_dict[key] = pg_info[key]
        if pg_info.get("betas") is not None:
            pg_dict["betas"] = tuple(pg_info["betas"])
        param_groups.append(pg_dict)

    optimizer = opt_cls(param_groups)

    # Warm up optimizer state (exp_avg, exp_avg_sq)
    for p in all_params:
        p.grad = torch.randn_like(p)
    optimizer.step()
    optimizer.zero_grad()

    return all_params, optimizer


def populate_grads(params, seed):
    """Fill gradients with deterministic random values."""
    gen = torch.Generator(device=params[0].device)
    gen.manual_seed(seed)
    for p in params:
        p.grad = torch.randn(
            p.shape, dtype=p.dtype, device=p.device, generator=gen
        )


def snapshot(params, optimizer):
    """Clone params and optimizer state."""
    param_snap = [p.data.clone() for p in params]
    optim_snap = {}
    for p in params:
        if p in optimizer.state:
            optim_snap[id(p)] = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in optimizer.state[p].items()
            }
    return param_snap, optim_snap


def restore(params, optimizer, param_snap, optim_snap):
    """Restore params and optimizer state from snapshot."""
    for p, snap in zip(params, param_snap):
        p.data.copy_(snap)
    for p in params:
        if id(p) in optim_snap:
            for k, v in optim_snap[id(p)].items():
                if isinstance(v, torch.Tensor):
                    optimizer.state[p][k].copy_(v)


class MockRmpClient:
    """Lightweight mock for RmpClient — caches by key, matching real RMP semantics."""

    def __init__(self):
        self._gpu_tensors: dict[str, torch.Tensor] = {}
        self._cpu_memory: dict[str, torch.Tensor] = {}

    def get_or_allocate_tensors(self, tensor_specs):
        tensors = {}
        allocated = False
        for spec in tensor_specs:
            if spec.name not in self._gpu_tensors:
                self._gpu_tensors[spec.name] = torch.empty(
                    spec.shape, dtype=spec.dtype, device=f"cuda:{spec.device}"
                )
                allocated = True
            tensors[spec.name] = self._gpu_tensors[spec.name]
        return tensors, allocated

    def get_or_allocate_cpu_memory(self, key, num_bytes):
        if key in self._cpu_memory:
            storage = self._cpu_memory[key].untyped_storage()
            # Unpin so caller can re-pin (simulates new process attaching)
            _cudart.cudaHostUnregister(ctypes.c_void_p(storage.data_ptr()))
            return storage, False
        tensor = torch.empty(num_bytes, dtype=torch.uint8, device="cpu")
        self._cpu_memory[key] = tensor
        return tensor.untyped_storage(), True

    def close(self):
        for tensor in self._cpu_memory.values():
            storage = tensor.untyped_storage()
            _cudart.cudaHostUnregister(ctypes.c_void_p(storage.data_ptr()))
        self._cpu_memory.clear()
        self._gpu_tensors.clear()

class OptimizerList:
    """Minimal wrapper so ResilientOptimizer can iterate over optimizers."""

    def __init__(self, optimizer):
        self._opt = optimizer

    def __iter__(self):
        return iter([self._opt])

    def step(self, *args, **kwargs):
        self._opt.step(*args, **kwargs)


def load_optimizer_info(path: str) -> dict:
    """Load optimizer info JSON and print summary."""
    with open(path) as f:
        info = json.load(f)

    opt = info["optimizers"][0]
    total = sum(
        sum(torch.tensor(p["shape"]).prod().item() for p in pg["params"])
        for pg in opt["param_groups"]
    )
    print(
        f"Optimizer: {opt['class']}, {total:,} elements, "
        f"fused={opt['defaults'].get('fused')}"
    )
    return info


def make_parser(description: str) -> argparse.ArgumentParser:
    """Create base argument parser with common flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--optimizer-info", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser
