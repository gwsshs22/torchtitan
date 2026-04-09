"""Profile resilient optimizer variants into a Chrome trace.

Profiles: baseline, cpu-snapshot, shadow, exp-8/512, chunked-512MB.
Output: resilient_optimizer_step_trace.json (open in chrome://tracing or Perfetto).

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
    load_optimizer_info,
    make_parser,
    populate_grads,
    restore,
    snapshot,
)


def profile_resilient(params, optimizer, device, output_path, warmup=2):
    """Profile all resilient variants into one Chrome trace."""
    pre_p, pre_o = snapshot(params, optimizer)
    opt_list = OptimizerList(optimizer)
    dev = torch.device(device)

    def reset_and_grad():
        restore(params, optimizer, pre_p, pre_o)
        populate_grads(params, seed=999)

    # Build variants outside the profiler (separate RMP per variant for isolation)
    rmps = []

    def new_rmp():
        r = MockRmpClient()
        rmps.append(r)
        return r

    reset_and_grad()
    cpu_snap = ResilientOptimizerCpuSnapshot(opt_list, dev)
    reset_and_grad()
    shadow = ResilientOptimizerGpuSnapshot(opt_list, new_rmp(), dev)
    reset_and_grad()
    exp_1_256 = ResilientOptimizer(
        opt_list, new_rmp(), dev, init_chunk_size_mb=1, max_chunk_size_mb=256,
    )
    reset_and_grad()
    exp_2_256 = ResilientOptimizer(
        opt_list, new_rmp(), dev, init_chunk_size_mb=2, max_chunk_size_mb=256,
    )
    reset_and_grad()
    exp_4_512 = ResilientOptimizer(
        opt_list, new_rmp(), dev, init_chunk_size_mb=4, max_chunk_size_mb=512,
    )
    reset_and_grad()
    chunked_32 = ResilientOptimizerChunked(
        opt_list, new_rmp(), dev, chunk_size_mb=32, use_cuda_graph=False,
    )
    reset_and_grad()
    chunked_128 = ResilientOptimizerChunked(
        opt_list, new_rmp(), dev, chunk_size_mb=128, use_cuda_graph=False,
    )
    reset_and_grad()
    chunked_512 = ResilientOptimizerChunked(
        opt_list, new_rmp(), dev, chunk_size_mb=512, use_cuda_graph=False,
    )

    variants = [
        ("baseline", lambda: optimizer.step()),
        ("cpu_snapshot", lambda: cpu_snap.step()),
        ("shadow_3583MB_gpu", lambda: shadow.step()),
        ("exp_1_256_9MB_cpu", lambda: exp_1_256.step()),
        ("exp_2_256_18MB_cpu", lambda: exp_2_256.step()),
        ("exp_4_512_36MB_cpu", lambda: exp_4_512.step()),
        ("chunked_32_96MB_gpu", lambda: chunked_32.step()),
        ("chunked_128_384MB_gpu", lambda: chunked_128.step()),
        ("chunked_512_1536MB_gpu", lambda: chunked_512.step()),
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
