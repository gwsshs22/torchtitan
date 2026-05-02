"""Benchmark resilient optimizer variants.

Compares: baseline, cpu-snapshot, shadow, exponential bootstrap, chunked.

Usage:
    uv run python -m torchtitan.experiments.resilient_opt.tests.benchmark \
        --optimizer-info /path/to/optimizer_info.json
"""

import time

import torch

from torchtitan.experiments.resilient_opt.resilient_opt import ResilientOptimizer
from torchtitan.experiments.resilient_opt.resilient_opt_chunked import (
    ResilientOptimizer as ResilientOptimizerChunked,
)
from torchtitan.experiments.resilient_opt.resilient_opt_cpu_snapshot import (
    ResilientOptimizerCpuSnapshot,
)
from torchtitan.experiments.resilient_opt.resilient_opt_gpu_snapshot import (
    ResilientOptimizerGpuSnapshot,
)
from torchtitan.experiments.resilient_opt.tests.common import (
    MockRmpClient,
    OptimizerList,
    create_optimizer_from_info,
    get_local,
    load_optimizer_info,
    make_parser,
    populate_grads,
    restore,
    snapshot,
)


def benchmark(params, optimizer, device, warmup=3, repeats=10):
    """Measure overhead of all resilient optimizer variants.

    Two grad-population strategies are needed because they exercise
    different perf paths:

      * ``populate_grads`` (default for variants that don't preserve grad
        memory across steps): allocates a fresh tensor each call.  This
        is what the in-process tests have always done, and it's safe for
        the eager paths.

      * ``refill_grads`` (used for the cuda-graph variant only):
        preserves the grad tensor across steps and only refills its
        contents.  Production with ``RmpGradientAllocator`` keeps grad
        memory pointer-stable across steps — which is what makes a
        captured graph remain valid — and ``populate_grads``'s alloc-
        per-step would invalidate the captured pointers.  Refill-only
        gives a fair measurement of the captured-graph path.
    """

    def measure_gpu_time(setup_fn, step_fn, warmup_count, repeat_count):
        """Measure GPU time of step_fn using CUDA events.
        setup_fn runs before each step but is excluded from timing."""
        for _ in range(warmup_count):
            setup_fn()
            step_fn()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_count)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_count)]

        torch.cuda.synchronize()
        for i in range(repeat_count):
            setup_fn()
            starts[i].record()
            step_fn()
            ends[i].record()
        torch.cuda.synchronize()

        total = sum(s.elapsed_time(e) for s, e in zip(starts, ends))
        return total / repeat_count

    def refill_grads(params, gens):
        """Refill existing grad tensors in place (preserves data pointers)."""
        for p, gen in zip(params, gens):
            get_local(p.grad).normal_(generator=gen)

    # -- Baseline --------------------------------------------------------------
    def vanilla_setup():
        populate_grads(params, seed=999)

    vanilla_ms = measure_gpu_time(vanilla_setup, optimizer.step, warmup, repeats)

    pre_p, pre_o = snapshot(params, optimizer)

    def bench(label, mem_label, make_resilient, rmp_client=None,
              preserve_grad_ptr=False):
        restore(params, optimizer, pre_p, pre_o)
        populate_grads(params, seed=999)
        r = make_resilient()
        if hasattr(r, "bind"):
            r.bind()

        if preserve_grad_ptr:
            # Grad memory now exists from the populate_grads above; switch
            # to refill mode so step() always sees the same data pointers.
            gens = [
                torch.Generator(device=p.device).manual_seed(999 + i)
                for i, p in enumerate(params)
            ]
            def setup():
                refill_grads(params, gens)
        else:
            def setup():
                restore(params, optimizer, pre_p, pre_o)
                populate_grads(params, seed=999)

        avg_ms = measure_gpu_time(setup, r.step, warmup, repeats)
        overhead = (avg_ms - vanilla_ms) / vanilla_ms * 100
        print(
            f"  {label:<28s}  {mem_label:>12s}  "
            f"{avg_ms:10.2f}  {overhead:>+9.1f}%"
        )
        if rmp_client is not None:
            rmp_client.close()

    # -- Header ----------------------------------------------------------------
    print(
        f"\n  {'Variant':<20s}  {'GPU Mem':>12s}  "
        f"{'Time (ms)':>10}  {'Overhead':>10}"
    )
    print("  " + "-" * 62)
    print(f"  {'baseline':<20s}  {'0 MB':>12s}  {vanilla_ms:10.2f}  {'0%':>10}")

    opt_list = OptimizerList(optimizer)
    dev = torch.device(device)

    # -- CPU snapshot ----------------------------------------------------------
    bench(
        "cpu-snapshot",
        "0 (CPU)",
        lambda: ResilientOptimizerCpuSnapshot(opt_list, dev),
    )

    # -- Shadow ----------------------------------------------------------------
    shadow_mb = sum(
        get_local(p).numel() * get_local(p).element_size()
        for p in optimizer.state
        for t in [p, optimizer.state[p]["exp_avg"], optimizer.state[p]["exp_avg_sq"]]
    ) / (1024**2)
    rmp = MockRmpClient()
    bench(
        "gpu-snapshot",
        f"{shadow_mb:.0f} MB",
        lambda: ResilientOptimizerGpuSnapshot(opt_list, rmp, dev),
        rmp_client=rmp,
    )

    # -- Exponential bootstrap -------------------------------------------------
    # Each (init, max) is benchmarked twice: once eager (use_cuda_graph=False,
    # the historical baseline) and once with the captured-step graph
    # (use_cuda_graph=True, preserving grad pointer across steps so the
    # captured graph remains valid).
    for init_mb, max_mb in [(1, 256), (1, 512), (2, 256), (2, 512), (4, 256), (4, 512)]:
        cpu_mb = init_mb * 9
        rmp = MockRmpClient()
        bench(
            f"exp-{init_mb}/{max_mb}",
            f"0 (CPU {cpu_mb})",
            lambda i=init_mb, m=max_mb, r=rmp: ResilientOptimizer(
                opt_list, r, dev,
                init_chunk_size_mb=i, max_chunk_size_mb=m,
                use_cuda_graph=False,
            ),
            rmp_client=rmp,
        )
        rmp_g = MockRmpClient()
        bench(
            f"exp-{init_mb}/{max_mb} +cuda_graph",
            f"0 (CPU {cpu_mb})",
            lambda i=init_mb, m=max_mb, r=rmp_g: ResilientOptimizer(
                opt_list, r, dev,
                init_chunk_size_mb=i, max_chunk_size_mb=m,
                use_cuda_graph=True,
            ),
            rmp_client=rmp_g,
            preserve_grad_ptr=True,
        )

    # -- Chunked ---------------------------------------------------------------
    # chunk_size_mb = param bytes; GPU buffer = 3× that
    for cs_mb in [16, 32, 64, 128, 256, 512]:
        gpu_mb = cs_mb * 3
        rmp = MockRmpClient()
        bench(
            f"chunked-{cs_mb}MB",
            f"{gpu_mb} MB",
            lambda mb=cs_mb, r=rmp: ResilientOptimizerChunked(
                opt_list, r, dev,
                chunk_size_mb=mb, use_cuda_graph=False,
            ),
            rmp_client=rmp,
        )

    restore(params, optimizer, pre_p, pre_o)


def main():
    parser = make_parser("Resilient optimizer benchmark")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    info = load_optimizer_info(args.optimizer_info)
    params, optimizer = create_optimizer_from_info(info, device=args.device)
    print(f"Created {len(params)} params on {args.device}")

    benchmark(params, optimizer, args.device)


if __name__ == "__main__":
    main()
