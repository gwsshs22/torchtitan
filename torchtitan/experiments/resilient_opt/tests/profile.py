"""Profile resilient optimizer variants into a Chrome trace.

Profiles eager + cuda-graph variants of ResilientOptimizer alongside the
shadow / cpu-snapshot / chunked baselines, so the trace shows side-by-side
the per-step kernel launches under each strategy.

Each variant has its own ``setup_fn`` that runs *outside* the profiled
region (so it doesn't pollute the per-step kernel timings):

  * eager variants use ``populate_grads`` — re-allocates ``p.grad`` each
    call, matching the historical baseline.
  * cuda-graph variants use ``refill_grads`` — keeps each ``p.grad``'s
    underlying memory pointer stable and only refreshes the values, which
    is what FSDP2 + ``RmpGradientAllocator`` does in production.  Without
    this the captured graph's recorded data pointers would dangle on the
    next step.

Output: resilient_optimizer_step_trace.json (open in chrome://tracing or
Perfetto).

Usage:
    uv run python -m torchtitan.experiments.resilient_opt.tests.profile \
        --optimizer-info /path/to/optimizer_info.json
"""

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


def profile_resilient(params, optimizer, device, output_path, warmup=3):
    """Profile all resilient variants into one Chrome trace."""
    pre_p, pre_o = snapshot(params, optimizer)
    opt_list = OptimizerList(optimizer)
    dev = torch.device(device)

    # Build variants outside the profiler (separate RMP per variant for isolation)
    rmps = []

    def new_rmp():
        r = MockRmpClient()
        rmps.append(r)
        return r

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    # `realloc_setup` matches the historical behavior: every call drops
    # the old grad tensor and allocates a new one.  Used for variants
    # whose perf doesn't depend on grad pointer stability.
    def realloc_setup():
        restore(params, optimizer, pre_p, pre_o)
        populate_grads(params, seed=999)

    # `refill_setup` is the variant for cuda-graph captures: allocate
    # grads once (lazily on first call), then only re-fill the existing
    # tensors so their data pointers stay stable across steps.  Mirrors
    # production FSDP2 + RmpGradientAllocator semantics.
    refill_state = {"gens": None}
    def refill_setup():
        restore(params, optimizer, pre_p, pre_o)
        if refill_state["gens"] is None:
            populate_grads(params, seed=999)
            refill_state["gens"] = [
                torch.Generator(device=p.device).manual_seed(1000 + i)
                for i, p in enumerate(params)
            ]
            return
        for p, gen in zip(params, refill_state["gens"]):
            get_local(p.grad).normal_(generator=gen)

    # ------------------------------------------------------------------
    # Construct variants
    # ------------------------------------------------------------------
    # Each entry: (label, setup_fn, build_fn).  build_fn is called once
    # *after* setup_fn so the resilient optimizer's bind() / first-step
    # capture see correctly-shaped grads.
    realloc_setup()
    cpu_snap = ResilientOptimizerCpuSnapshot(opt_list, dev)
    realloc_setup()
    shadow = ResilientOptimizerGpuSnapshot(opt_list, new_rmp(), dev)

    # Eager exp variants — re-allocate grad each call.
    realloc_setup()
    exp_1_256_eager = ResilientOptimizer(
        opt_list, new_rmp(), dev,
        init_chunk_size_mb=1, max_chunk_size_mb=256,
        use_cuda_graph=False,
    )
    exp_1_256_eager.bind()
    realloc_setup()
    exp_2_256_eager = ResilientOptimizer(
        opt_list, new_rmp(), dev,
        init_chunk_size_mb=2, max_chunk_size_mb=256,
        use_cuda_graph=False,
    )
    exp_2_256_eager.bind()
    realloc_setup()
    exp_4_512_eager = ResilientOptimizer(
        opt_list, new_rmp(), dev,
        init_chunk_size_mb=4, max_chunk_size_mb=512,
        use_cuda_graph=False,
    )
    exp_4_512_eager.bind()

    # Cuda-graph exp variants — keep grad pointer stable.
    # Each builds its own refill_state segment so the variants don't
    # share lazy-init state.  We do this by wrapping refill_setup with
    # a fresh dict.
    def make_refill_setup():
        st = {"gens": None}
        def s():
            restore(params, optimizer, pre_p, pre_o)
            if st["gens"] is None:
                populate_grads(params, seed=999)
                st["gens"] = [
                    torch.Generator(device=p.device).manual_seed(1000 + i)
                    for i, p in enumerate(params)
                ]
                return
            for p, gen in zip(params, st["gens"]):
                get_local(p.grad).normal_(generator=gen)
        return s

    setup_1_256_g = make_refill_setup()
    setup_1_256_g()
    exp_1_256_graph = ResilientOptimizer(
        opt_list, new_rmp(), dev,
        init_chunk_size_mb=1, max_chunk_size_mb=256,
        use_cuda_graph=True,
    )
    exp_1_256_graph.bind()

    setup_2_256_g = make_refill_setup()
    setup_2_256_g()
    exp_2_256_graph = ResilientOptimizer(
        opt_list, new_rmp(), dev,
        init_chunk_size_mb=2, max_chunk_size_mb=256,
        use_cuda_graph=True,
    )
    exp_2_256_graph.bind()

    setup_4_512_g = make_refill_setup()
    setup_4_512_g()
    exp_4_512_graph = ResilientOptimizer(
        opt_list, new_rmp(), dev,
        init_chunk_size_mb=4, max_chunk_size_mb=512,
        use_cuda_graph=True,
    )
    exp_4_512_graph.bind()

    realloc_setup()
    chunked_32 = ResilientOptimizerChunked(
        opt_list, new_rmp(), dev, chunk_size_mb=32, use_cuda_graph=False,
    )
    realloc_setup()
    chunked_128 = ResilientOptimizerChunked(
        opt_list, new_rmp(), dev, chunk_size_mb=128, use_cuda_graph=False,
    )
    realloc_setup()
    chunked_512 = ResilientOptimizerChunked(
        opt_list, new_rmp(), dev, chunk_size_mb=512, use_cuda_graph=False,
    )

    # ------------------------------------------------------------------
    # Variant table.  For graph variants the setup is the per-variant
    # refill closure so the captured graph remains valid across both the
    # warmup runs (which trigger capture) and the profiled run (replay).
    # ------------------------------------------------------------------
    variants = [
        ("baseline", realloc_setup, lambda: optimizer.step()),
        ("cpu_snapshot", realloc_setup, lambda: cpu_snap.step()),
        ("shadow_3583MB_gpu", realloc_setup, lambda: shadow.step()),
        ("exp_1_256_9MB_cpu_eager", realloc_setup, lambda: exp_1_256_eager.step()),
        ("exp_1_256_9MB_cpu_graph", setup_1_256_g, lambda: exp_1_256_graph.step()),
        ("exp_2_256_18MB_cpu_eager", realloc_setup, lambda: exp_2_256_eager.step()),
        ("exp_2_256_18MB_cpu_graph", setup_2_256_g, lambda: exp_2_256_graph.step()),
        ("exp_4_512_36MB_cpu_eager", realloc_setup, lambda: exp_4_512_eager.step()),
        ("exp_4_512_36MB_cpu_graph", setup_4_512_g, lambda: exp_4_512_graph.step()),
        ("chunked_32_96MB_gpu", realloc_setup, lambda: chunked_32.step()),
        ("chunked_128_384MB_gpu", realloc_setup, lambda: chunked_128.step()),
        ("chunked_512_1536MB_gpu", realloc_setup, lambda: chunked_512.step()),
    ]

    # Warmup: triggers the cuda-graph capture for graph variants on their
    # *first* warmup pass; subsequent passes (and the profiled pass) replay.
    for _ in range(warmup):
        for _, setup, fn in variants:
            setup()
            fn()
    torch.cuda.synchronize()

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for name, setup, fn in variants:
            setup()
            torch.cuda.synchronize()
            with torch.profiler.record_function(name):
                fn()
                torch.cuda.synchronize()

    for r in rmps:
        r.close()
    prof.export_chrome_trace(output_path)
    print(f"Trace exported to {output_path}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))


def main():
    parser = make_parser("Profile resilient optimizer variants")
    parser.add_argument(
        "--output", type=str, default="resilient_optimizer_step_trace.json",
        help="Output path for Chrome trace JSON",
    )
    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    info = load_optimizer_info(args.optimizer_info)
    params, optimizer = create_optimizer_from_info(info, device=args.device)
    print(f"Created {len(params)} params on {args.device}")

    profile_resilient(params, optimizer, args.device, args.output)


if __name__ == "__main__":
    main()
