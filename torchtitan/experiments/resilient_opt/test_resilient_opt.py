"""Test environment for resilient optimizer.

Recreates a realistic optimizer setup from optimizer_info.json (dumped from
real training), simulates a mid-step fault, and verifies snapshot+replay
recovery produces the correct result.

Usage:
    python -m torchtitan.experiments.resilient_opt.test_resilient_opt \
        --optimizer-info /path/to/optimizer_info.json
"""

import argparse
import json
import sys

import torch

from torchtitan.experiments.resilient_opt.resilient_opt import ResilientOptimizer
from torchtitan.experiments.resilient_opt.resilient_opt_chunked import ResilientOptimizer as ResilientOptimizerChunked
from torchtitan.experiments.resilient_opt.resilient_opt_cpu_snapshot import ResilientOptimizerCpuSnapshot
from torchtitan.experiments.resilient_opt.resilient_opt_pipeline import ResilientOptimizerV2
from torchtitan.experiments.resilient_opt.resilient_opt_shadow import ResilientOptimizerShadow

def _get_local(tensor):
    from torch.distributed._tensor import DTensor
    if isinstance(tensor, DTensor):
        return tensor._local_tensor
    return tensor

DTYPE_MAP = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
}


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
                torch.randn(p_info["shape"], dtype=DTYPE_MAP[p_info["dtype"]], device=device),
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
        p.grad = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)


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


def test_fault_recovery(params, optimizer):
    """Simulate mid-step fault and verify snapshot+replay recovery."""
    # 1. Snapshot pre-step state
    pre_params, pre_optim = snapshot(params, optimizer)

    # 2. Populate grads and save them
    populate_grads(params, seed=200)
    saved_grads = [p.grad.clone() for p in params]

    # 3. Run the correct step (ground truth)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        optimizer.step()
        optimizer.zero_grad()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
    prof.export_chrome_trace("optimizer_step_trace.json")
    correct_params = [p.data.clone() for p in params]

    # 4. Restore to pre-step, re-populate grads, do step again but corrupt
    restore(params, optimizer, pre_params, pre_optim)
    populate_grads(params, seed=200)
    optimizer.step()

    # Simulate partial fault: revert first half of params to pre-step values
    n_corrupt = len(params) // 2
    for i in range(n_corrupt):
        params[i].data.copy_(pre_params[i])
        if id(params[i]) in pre_optim:
            for k, v in pre_optim[id(params[i])].items():
                if isinstance(v, torch.Tensor):
                    optimizer.state[params[i]][k].copy_(v)

    # Verify state is corrupted
    corrupted = any(
        not torch.equal(correct_params[i], params[i].data)
        for i in range(len(params))
    )
    print(f"  State corrupted after fault: {corrupted}")
    assert corrupted, "Fault simulation failed to corrupt state"

    # 5. Recovery: restore from snapshot + replay with saved grads
    restore(params, optimizer, pre_params, pre_optim)
    for p, g in zip(params, saved_grads):
        p.grad = g
    optimizer.step()
    optimizer.zero_grad()

    # 6. Verify recovery matches ground truth
    all_match = True
    for i, (correct, p) in enumerate(zip(correct_params, params)):
        if not torch.equal(correct, p.data):
            diff = (correct - p.data).abs().max().item()
            print(f"  Param {i}: MISMATCH max_diff={diff:.6e}")
            all_match = False

    print(f"  Recovery matches ground truth: {all_match}")
    return all_match


class MockRmpClient:
    """Lightweight mock for RmpClient — allocates regular CUDA/CPU tensors."""

    def get_or_allocate_tensors(self, tensor_specs):
        tensors = {}
        for spec in tensor_specs:
            tensors[spec.name] = torch.empty(
                spec.shape, dtype=spec.dtype, device=f"cuda:{spec.device}"
            )
        return tensors, True  # allocated=True

    def get_or_allocate_cpu_memory(self, key, num_bytes):
        storage = torch.UntypedStorage(num_bytes, device="cpu")
        return storage, True  # allocated=True


class _OptimizerList:
    """Minimal wrapper so ResilientOptimizer can iterate over optimizers."""

    def __init__(self, optimizer):
        self._opt = optimizer

    def __iter__(self):
        return iter([self._opt])

    def step(self, *args, **kwargs):
        self._opt.step(*args, **kwargs)


def test_resilient_optimizer(params, optimizer, device):
    """Test ResilientOptimizer step correctness.

    1. Run optimizer.step() to get ground truth.
    2. Restore, run resilient.step(), verify it matches.
    """
    pre_params, pre_optim = snapshot(params, optimizer)
    populate_grads(params, seed=300)
    saved_grads = [p.grad.clone() for p in params]

    # Ground truth
    optimizer.step()
    correct_params = [p.data.clone() for p in params]
    correct_exp_avg = [optimizer.state[p]["exp_avg"].clone() for p in params]
    correct_exp_avg_sq = [optimizer.state[p]["exp_avg_sq"].clone() for p in params]
    correct_steps = [optimizer.state[p]["step"].clone() for p in params]

    # Resilient step
    restore(params, optimizer, pre_params, pre_optim)
    for p, g in zip(params, saved_grads):
        p.grad = g

    mock_client = MockRmpClient()
    opt_list = _OptimizerList(optimizer)
    resilient = ResilientOptimizer(
        opt_list, mock_client, torch.device(device),
        init_chunk_mb=4, max_chunk_mb=256,
    )
    resilient.step()

    step_match = True
    for i, p in enumerate(params):
        if not torch.equal(correct_params[i], p.data):
            diff = (correct_params[i] - p.data).abs().max().item()
            print(f"  [step] Param {i}: MISMATCH max_diff={diff:.6e}")
            step_match = False
        if not torch.equal(correct_exp_avg[i], optimizer.state[p]["exp_avg"]):
            diff = (correct_exp_avg[i] - optimizer.state[p]["exp_avg"]).abs().max().item()
            print(f"  [step] exp_avg {i}: MISMATCH max_diff={diff:.6e}")
            step_match = False
        if not torch.equal(correct_exp_avg_sq[i], optimizer.state[p]["exp_avg_sq"]):
            diff = (correct_exp_avg_sq[i] - optimizer.state[p]["exp_avg_sq"]).abs().max().item()
            print(f"  [step] exp_avg_sq {i}: MISMATCH max_diff={diff:.6e}")
            step_match = False
        if not torch.equal(correct_steps[i], optimizer.state[p]["step"]):
            print(f"  [step] step {i}: {correct_steps[i].item()} vs {optimizer.state[p]['step'].item()}")
            step_match = False
    print(f"  Chunked step matches optimizer.step(): {step_match}")
    return step_match


def benchmark(params, optimizer, device, chunk_sizes_mb, warmup=3, repeats=10):
    """Measure overhead of resilient step vs vanilla optimizer.step()."""
    import time

    # -- Baseline: vanilla optimizer.step() --------------------------------
    def run_vanilla():
        populate_grads(params, seed=999)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(warmup):
        run_vanilla()
    vanilla_times = [run_vanilla() for _ in range(repeats)]
    vanilla_ms = sum(vanilla_times) / len(vanilla_times) * 1000

    pre_p, pre_o = snapshot(params, optimizer)

    def bench(label, mem_label, make_resilient):
        restore(params, optimizer, pre_p, pre_o)
        populate_grads(params, seed=999)
        r = make_resilient()

        def run():
            restore(params, optimizer, pre_p, pre_o)
            populate_grads(params, seed=999)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r.step()
            torch.cuda.synchronize()
            return time.perf_counter() - t0

        for _ in range(warmup):
            run()
        times = [run() for _ in range(repeats)]
        avg_ms = sum(times) / len(times) * 1000
        overhead = (avg_ms - vanilla_ms) / vanilla_ms * 100
        print(f"  {label:<20s}  {mem_label:>12s}  {avg_ms:10.2f}  {overhead:>+9.1f}%")

    print(f"\n  {'Variant':<20s}  {'GPU Mem':>12s}  {'Time (ms)':>10}  {'Overhead':>10}")
    print("  " + "-" * 62)
    print(f"  {'baseline':<20s}  {'0 MB':>12s}  {vanilla_ms:10.2f}  {'0%':>10}")

    opt_list = _OptimizerList(optimizer)
    dev = torch.device(device)

    # -- Fixed-overhead baselines ------------------------------------------

    # CPU snapshot: 0 GPU memory overhead (uses pinned CPU)
    bench(
        "cpu-snapshot", "0 (CPU)",
        lambda: ResilientOptimizerCpuSnapshot(opt_list, dev),
    )

    # Shadow: 3x state GPU memory (full duplicate of param+exp_avg+exp_avg_sq)
    total_state_bytes = sum(
        _get_local(p).numel() * _get_local(p).element_size() * 3
        for p in [p for opt in optimizer.state for p in [opt]]
    )
    # Compute shadow size from actual params
    shadow_mb = sum(
        _get_local(p).numel() * _get_local(p).element_size()
        for opt in [optimizer]
        for p in opt.state
        for t in [p, opt.state[p]["exp_avg"], opt.state[p]["exp_avg_sq"]]
    ) / (1024 ** 2)
    bench(
        "shadow", f"{shadow_mb:.0f} MB",
        lambda: ResilientOptimizerShadow(opt_list, MockRmpClient(), dev),
    )

    # -- Exponential bootstrap (final): 0 GPU overhead -----------------------
    for init_mb in [4, 8]:
        for max_mb in [32, 64, 128, 256, 512, 1024]:
            label = f"exp-{init_mb}/{max_mb}"
            bench(
                label, f"0 (CPU {init_mb})",
                lambda i=init_mb, m=max_mb: ResilientOptimizer(
                    opt_list, MockRmpClient(), dev,
                    init_chunk_mb=i, max_chunk_mb=m,
                ),
            )

    # -- Chunked variants at different memory budgets ----------------------
    for total_mb in [32, 64, 128, 256, 512, 1024, 2048]:
        bench(
            f"chunked-{total_mb}MB", f"{total_mb} MB",
            lambda mb=total_mb: ResilientOptimizerChunked(
                opt_list, MockRmpClient(), dev,
                chunk_size_mb=mb, use_cuda_graph=False,
            ),
        )

    restore(params, optimizer, pre_p, pre_o)


def profile_resilient(params, optimizer, device, output_path):
    """Profile all resilient variants into one Chrome trace."""
    warmup = 2

    pre_p, pre_o = snapshot(params, optimizer)
    opt_list = _OptimizerList(optimizer)
    dev = torch.device(device)

    def reset_and_grad():
        restore(params, optimizer, pre_p, pre_o)
        populate_grads(params, seed=999)

    # Build all variants outside the profiler
    reset_and_grad()
    cpu_snap = ResilientOptimizerCpuSnapshot(opt_list, dev)
    reset_and_grad()
    shadow = ResilientOptimizerShadow(opt_list, MockRmpClient(), dev)
    reset_and_grad()
    exp_8_512 = ResilientOptimizer(opt_list, MockRmpClient(), dev, init_chunk_mb=8, max_chunk_mb=512)
    reset_and_grad()
    chunked_512 = ResilientOptimizerChunked(opt_list, MockRmpClient(), dev, chunk_size_mb=512, use_cuda_graph=False)

    variants = [
        ("baseline", lambda: optimizer.step()),
        ("cpu_snapshot", lambda: cpu_snap.step()),
        ("shadow", lambda: shadow.step()),
        ("exp_8_512", lambda: exp_8_512.step()),
        ("chunked_512MB", lambda: chunked_512.step()),
    ]

    # Warmup
    for _ in range(warmup):
        for _, fn in variants:
            reset_and_grad()
            fn()

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for name, fn in variants:
            reset_and_grad()
            torch.cuda.synchronize()
            with torch.profiler.record_function(name):
                fn()
                torch.cuda.synchronize()

    prof.export_chrome_trace(output_path)
    print(f"Trace exported to {output_path}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-info", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run overhead benchmark across chunk sizes",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Profile baseline + resilient (8/32/1024 MB) into a Chrome trace",
    )
    args = parser.parse_args()

    with open(args.optimizer_info) as f:
        info = json.load(f)

    opt = info["optimizers"][0]
    total = sum(
        sum(torch.tensor(p["shape"]).prod().item() for p in pg["params"])
        for pg in opt["param_groups"]
    )
    print(f"Optimizer: {opt['class']}, {total:,} elements, fused={opt['defaults'].get('fused')}")

    params, optimizer = create_optimizer_from_info(info, device=args.device)
    print(f"Created {len(params)} params on {args.device}")

    if args.profile:
        profile_resilient(
            params, optimizer, args.device,
            "resilient_optimizer_step_trace.json",
        )
    elif args.benchmark:
        chunk_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
        benchmark(params, optimizer, args.device, chunk_sizes)
    else:
        print("\n=== Manual Fault Recovery Test ===")
        passed_manual = test_fault_recovery(params, optimizer)

        # Re-create for resilient optimizer test (clean state)
        params, optimizer = create_optimizer_from_info(info, device=args.device)

        print("\n=== ResilientOptimizer Test ===")
        passed_resilient = test_resilient_optimizer(params, optimizer, args.device)

        passed = passed_manual and passed_resilient
        print(f"\nResult: {'PASS' if passed else 'FAIL'}")
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
