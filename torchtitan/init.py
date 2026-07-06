# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Configurable initialization sequences for Trainer.

Provides two init modes controlled by ``leto.init_mode``:
  - "reordered" (default): defers CUDA context until after standby poll,
    enabling standby processes to pre-initialize more work.
  - "baseline": original init order (pre-98d7b6f), CUDA context set immediately.

A future "dynamic" mode will use a task dependency graph with profiling.
"""

import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import random

import torch
import torch.distributed as dist
import torch.distributed.tensor._random as dtensor_random

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.ft import FTManager
from torchtitan.components.fsdp_collective_manager import FsdpCollectiveManager
from torchtitan.components.gemini.checkpoint import GeminiCheckpointManager
from torchtitan.components.loss import rescale_accumulated_loss
from torchtitan.components.metrics import (
    build_metrics_processor as _build_metrics_processor,
    ensure_pp_loss_visible,
)
from torchtitan.components.rmp_manager import RmpManager
from leto.rmp.flags import FLAG_KIND_GPU
from torchtitan.experiments.resilient_opt.resilient_opt import ResilientOptimizer
from torchtitan.experiments.resilient_opt.resilient_opt_cpu_snapshot import (
    AsyncCpuSnapshotOptimizer,
)
from torchtitan.components.skip_shape_infer import maybe_warmup_stages
from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import logger

# Optional leto integration
try:
    from leto.launch.worker_controller_client import (
        poll_standby_status,
        is_standby as leto_is_standby,
        STANDBY_ACTION_ACTIVATE,
        STANDBY_ACTION_TERMINATE,
    )
    _LETO_AVAILABLE = True
except ImportError:
    _LETO_AVAILABLE = False


@dataclass
class InitContext:
    """Mutable bag of state accumulated during initialization.

    Task functions populate fields; ``apply_to`` transfers them to Trainer.
    """
    job_config: JobConfig

    # distributed
    parallel_dims: ParallelDims | None = None
    device: torch.device | None = None

    # internal (not transferred to Trainer)
    _device_module: Any = None
    _device_type: str | None = None
    _global_rank: int | None = None
    _batch_degree: int | None = None
    _batch_rank: int | None = None
    _model: torch.nn.Module | None = None
    _model_args: Any = None
    _model_converters: Any = None
    _color: Any = None

    # components
    ft_manager: FTManager | None = None
    gc_handler: utils.GarbageCollection | None = None
    train_spec: train_spec_module.TrainSpec | None = None
    tokenizer: Any = None
    dataloader: Any = None
    model_parts: list[torch.nn.Module] | None = None
    model_args: Any = None
    loss_fn: Any = None
    gradient_accumulation_steps: int | None = None
    train_context: Any = None
    maybe_enable_amp: Any = None
    checkpointer: Any = None
    optimizers: Any = None
    lr_schedulers: Any = None
    metrics_processor: Any = None
    rmp_manager: RmpManager | None = None
    rmp_restored: bool = False
    buffer_device: str | None = None
    _init_device: str | None = None

    # PP state
    pp_schedule: Any = None
    pp_has_first_stage: bool = False
    pp_has_last_stage: bool = False

    # validator
    validator: Any = None

    # trainer states
    step: int = 0
    ntokens_seen: int = 0
    _prev_step_faulted: bool = False
    _restored_step: int = 0

    # resilient optimizer
    _resilient_opt: Any = None
    _cpu_snapshot_opt: Any = None

    # expert dist tracker
    _expert_dist_tracker: Any = None

    # collective manager (internal, not transferred)
    _collective_manager: Any = None
    is_standby: bool = False

    # standby-only CPU gloo PG used for cross-rank coordination during init
    standby_gloo_pg: Any = None

    def apply_to(self, trainer: Any) -> None:
        """Transfer all public fields to the trainer instance."""
        for f in dataclasses.fields(self):
            if f.name == "job_config":
                continue
            val = getattr(self, f.name)
            setattr(trainer, f.name, val)

    # Stateful protocol — allows checkpoint loading to restore step/ntokens/RNG
    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "ntokens_seen": self.ntokens_seen,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state": torch.cuda.get_rng_state(self.device),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]
        if "torch_rng_state" in state_dict:
            torch.set_rng_state(state_dict["torch_rng_state"])
        if "cuda_rng_state" in state_dict:
            torch.cuda.set_rng_state(state_dict["cuda_rng_state"], self.device)
        if "numpy_rng_state" in state_dict:
            np.random.set_state(state_dict["numpy_rng_state"])
        if "python_rng_state" in state_dict:
            random.setstate(state_dict["python_rng_state"])


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def init_distributed(ctx: InitContext) -> None:
    job_config = ctx.job_config
    world_size = dist_utils.init_distributed(
        job_config.comm,
        enable_cpu_backend=job_config.training.enable_cpu_offload,
        base_folder=job_config.job.dump_folder,
    )
    parallelism_config = job_config.parallelism
    ctx.parallel_dims = ParallelDims(
        dp_shard=parallelism_config.data_parallel_shard_degree,
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        cp=parallelism_config.context_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        ep=parallelism_config.expert_parallel_degree,
        etp=parallelism_config.expert_tensor_parallel_degree,
        world_size=world_size,
    )
    logger.info("Init distributed.")


def set_device_attr(ctx: InitContext) -> None:
    ctx._device_module, ctx._device_type = utils.device_module, utils.device_type
    ctx.device = torch.device(f"{ctx._device_type}:{int(os.environ['LOCAL_RANK'])}")


def compute_batch_info(ctx: InitContext) -> None:
    ctx._global_rank = int(os.environ["RANK"])
    ctx._batch_degree, ctx._batch_rank = ctx.parallel_dims.get_batch_info(
        ctx._global_rank
    )


def init_ft_manager(ctx: InitContext) -> None:
    # pyrefly: ignore [bad-argument-type]
    ctx.ft_manager = FTManager(ctx.job_config.fault_tolerance)
    ctx._batch_degree, ctx._batch_rank = ctx.ft_manager.get_dp_info(
        ctx._batch_degree, ctx._batch_rank
    )


def init_garbage_collection(ctx: InitContext) -> None:
    job_config = ctx.job_config
    ctx.gc_handler = utils.GarbageCollection(
        gc_freq=job_config.training.gc_freq, debug=job_config.training.gc_debug
    )


def load_train_spec(ctx: InitContext) -> None:
    ctx.train_spec = train_spec_module.get_train_spec(ctx.job_config.model.name)


def build_tokenizer_and_dataloader(ctx: InitContext) -> None:
    job_config = ctx.job_config
    ctx.tokenizer = (
        ctx.train_spec.build_tokenizer_fn(job_config)
        if ctx.train_spec.build_tokenizer_fn is not None
        else None
    )
    ctx.dataloader = ctx.train_spec.build_dataloader_fn(
        dp_world_size=ctx._batch_degree,
        dp_rank=ctx._batch_rank,
        tokenizer=ctx.tokenizer,
        job_config=job_config,
    )


def build_model_on_meta(ctx: InitContext) -> None:
    job_config = ctx.job_config
    parallel_dims = ctx.parallel_dims

    model_args = ctx.train_spec.model_args[job_config.model.flavor]
    model_args.update_from_config(job_config)

    vocab_parallel_divisor = 128
    if (
        hasattr(model_args, "vocab_size")
        and model_args.vocab_size % vocab_parallel_divisor != 0
    ):
        old_vocab_size = model_args.vocab_size
        model_args.vocab_size = (
            (old_vocab_size + vocab_parallel_divisor - 1) // vocab_parallel_divisor
        ) * vocab_parallel_divisor
        logger.info(
            f"Padded vocab_size from {old_vocab_size} to {model_args.vocab_size} "
            f"for even sharding (divisible by TP({parallel_dims.tp}) x FSDP({parallel_dims.dp_shard}))"
        )

    ctx._model_args = model_args
    ctx.model_args = model_args

    logger.info(
        f"Building {job_config.model.name} {job_config.model.flavor}"
        f"with {json.dumps(dataclasses.asdict(model_args), indent=2, ensure_ascii=False)}"
    )
    with (
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
    ):
        # pyrefly: ignore[bad-instantiation]
        ctx._model = ctx.train_spec.model_cls(model_args)


def apply_model_converters(ctx: InitContext) -> None:
    ctx._model_converters = build_model_converters(
        ctx.job_config, ctx.parallel_dims
    )
    ctx._model_converters.convert(ctx._model)


def build_metrics_processor(ctx: InitContext) -> None:
    job_config = ctx.job_config
    build_fn = (
        _build_metrics_processor
        if ctx.train_spec.build_metrics_processor_fn is None
        else ctx.train_spec.build_metrics_processor_fn
    )
    ctx.metrics_processor = build_fn(
        job_config,
        ctx.parallel_dims,
        ctx._model_args,
    )
    ctx._color = ctx.metrics_processor.color

    (
        model_param_count,
        ctx.metrics_processor.num_flops_per_token,
    ) = ctx._model_args.get_nparams_and_flops(
        ctx._model, job_config.training.seq_len
    )
    logger.info(
        f"{ctx._color.blue}Model {job_config.model.name} {job_config.model.flavor} "
        f"{ctx._color.red}size: {model_param_count:,} total parameters{ctx._color.reset}"
    )

def build_loss_fn_and_grad_accum(ctx: InitContext) -> None:
    job_config = ctx.job_config
    parallel_dims = ctx.parallel_dims

    ctx.loss_fn = ctx.train_spec.build_loss_fn(
        job_config, parallel_dims=parallel_dims, ft_manager=ctx.ft_manager
    )

    global_batch_size = job_config.training.global_batch_size
    if global_batch_size < 0:
        global_batch_size = job_config.training.local_batch_size * ctx._batch_degree
    assert global_batch_size > 0
    assert (
        global_batch_size % (job_config.training.local_batch_size * ctx._batch_degree)
        == 0
    ), (
        f"global batch size must be multiple of local batch size times "
        f"data-parallel degree ({global_batch_size} "
        f"% ({job_config.training.local_batch_size} * {ctx._batch_degree}) != 0)"
    )

    ctx.gradient_accumulation_steps = global_batch_size // (
        job_config.training.local_batch_size * ctx._batch_degree
    )
    assert ctx.gradient_accumulation_steps > 0
    ctx.loss_fn = rescale_accumulated_loss(
        ctx.loss_fn, ctx.gradient_accumulation_steps
    )


def build_train_context(ctx: InitContext) -> None:
    job_config = ctx.job_config
    loss_parallel_enabled = (
        ctx.parallel_dims.tp_enabled
        and not job_config.parallelism.disable_loss_parallel
    )
    ctx.train_context = dist_utils.get_train_context(loss_parallel_enabled)
    ctx.maybe_enable_amp = dist_utils.maybe_enable_amp(
        ctx.parallel_dims,
        job_config.training.mixed_precision_param,
        ctx._device_type,
    )


def init_gemini_checkpoint_partial(ctx: InitContext) -> None:
    """Reordered mode: create partial GeminiCheckpointManager before standby poll."""
    if ctx.job_config.checkpoint.use_gemini:
        ctx.checkpointer = GeminiCheckpointManager(
            dataloader=ctx.dataloader,
            states={"train_state": ctx},
            checkpoint_config=ctx.job_config.checkpoint,
            base_folder=ctx.job_config.job.dump_folder,
        )


def _maybe_standby_oom_test_alloc(ctx: InitContext) -> None:
    """OOM-safeguard E2E test hook (standby side). Make this standby rank
    reserve (through the active's broker) and HOLD a controlled amount of GPU
    memory after its progressive init, so the active's reclaim of the standby
    is the difference between OOM and success when the active's usage spikes
    (maybe_alloc_for_oom_test). Gated on leto.oom_test_standby_alloc_mb and
    progressive_init. The tensor is stashed on ctx so it survives parking."""
    cfg = ctx.job_config.leto
    alloc_mb = int(getattr(cfg, "oom_test_standby_alloc_mb", 0))
    if alloc_mb <= 0 or not cfg.progressive_init:
        return
    rank = int(os.environ.get("RANK", "0"))
    if int(cfg.progressive_reservation_margin_mb) <= 0:
        # No broker (grant-only OOM control): hold the memory ungated. Asking a
        # broker that isn't running would block the _reserve handshake forever.
        kind = "grant"
        logger.info(
            f"[oom_test] standby rank={rank} reservation disabled (margin=0); "
            f"holding {alloc_mb}MiB ungated"
        )
    else:
        # Reserve through the broker (raise our grant to cover current footprint
        # plus the test allocation). Only HOLD if the broker grants it -- a
        # relaunched standby after a reclaim has less free memory and would
        # OOM-loop if it allocated past what the broker can give it.
        try:
            from torchtitan.components.init.progressive import _reserve

            cur_mb = _get_gpu_mem_mb() or 0.0
            target_b = int((cur_mb + alloc_mb + 256) * 1024 * 1024)
            kind, _ = _reserve(rank, target_b, 0.01, None)
            logger.info(
                f"[oom_test] standby rank={rank} reserve(+{alloc_mb}MiB, "
                f"target~{int(cur_mb + alloc_mb)}MiB) -> {kind}"
            )
        except Exception:
            logger.warning("[oom_test] standby reservation failed", exc_info=True)
            kind = "deny"
    if kind != "grant":
        logger.info(
            f"[oom_test] standby rank={rank} reservation not granted; "
            f"skipping the {alloc_mb}MiB hold"
        )
        return
    n = (alloc_mb * 1024 * 1024) // 2  # bf16 = 2 bytes/elem
    ctx._oom_test_hold = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    ctx._oom_test_hold.fill_(0)
    torch.cuda.synchronize()
    free_b, _ = torch.cuda.mem_get_info()
    logger.info(
        f"[oom_test] standby rank={rank} holding {alloc_mb}MiB; "
        f"device free now {int(free_b // (1024 * 1024))}MiB"
    )


def maybe_wait_for_resuming(ctx: InitContext) -> None:
    """Standby poll — blocks until activated or terminated."""
    if not _LETO_AVAILABLE:
        return
    if not leto_is_standby():
        return

    _maybe_standby_oom_test_alloc(ctx)

    logger.info("Entering standby mode - polling for activation...")
    poll_interval = ctx.job_config.leto.standby_poll_interval
    while True:
        try:
            action = poll_standby_status()
        except Exception:
            logger.warning("Error while polling standby status", exc_info=True)
            # os._exit (not sys.exit) because @record on Trainer.__init__
            # swallows SystemExit(0); we need a hard exit to prevent the
            # process from continuing into train() with a half-initialized
            # Trainer.
            os._exit(0)
        if action == STANDBY_ACTION_ACTIVATE:
            logger.info("Standby activated - resuming initialization")
            ctx.standby_activated = True
            _release_occupy_holds(ctx)
            return
        elif action == STANDBY_ACTION_TERMINATE:
            logger.info("Standby terminated")
            os._exit(0)
        time.sleep(poll_interval)


def _standby_wait_for_allocation_flag(
    rmp_client,
    kind: str,
    poll_interval: float = 0.1,
) -> None:
    """Standby-side wait on an RMP allocation flag with controller fallback.

    Polls ``rmp_client.get_allocation_flag(kind)``; the wait ends when:
      * the flag reads >= 1 (active completed allocation), OR
      * ``poll_standby_status()`` returns STANDBY_ACTION_ACTIVATE
        (worker controller is promoting us — keep going so the caller
        can do the allocation itself).

    On STANDBY_ACTION_TERMINATE (or a poll error), logs and ``os._exit(0)``.

    No-op if leto isn't available or this isn't a standby.
    """
    if not (_LETO_AVAILABLE and leto_is_standby()):
        # Not standby; just check the flag once for parity.
        if rmp_client.get_allocation_flag(kind) < 1:
            logger.warning(
                f"[RMP] non-standby called wait for {kind} flag, but flag != 1"
            )
        return

    if rmp_client.get_allocation_flag(kind) >= 1:
        logger.info(f"[RMP] {kind} allocation flag already = 1, no wait")
        return

    logger.info(f"[RMP] waiting for {kind} allocation flag (poll={poll_interval}s)")
    start = time.monotonic()
    while True:
        if rmp_client.get_allocation_flag(kind) >= 1:
            elapsed = time.monotonic() - start
            logger.info(f"[RMP] {kind} allocation flag = 1, waited {elapsed:.2f}s")
            return

        try:
            action = poll_standby_status()
        except Exception:
            logger.error(
                f"Error polling standby status while waiting for {kind} flag",
                exc_info=True,
            )
            sys.exit(1)

        if action == STANDBY_ACTION_ACTIVATE:
            elapsed = time.monotonic() - start
            logger.info(
                f"[RMP] {kind} allocation flag wait released by ACTIVATE "
                f"after {elapsed:.2f}s"
            )
            return
        if action == STANDBY_ACTION_TERMINATE:
            elapsed = time.monotonic() - start
            logger.error(
                f"[RMP] standby TERMINATE while waiting for {kind} allocation "
                f"flag after {elapsed:.2f}s"
            )
            # os._exit (not sys.exit): @record on Trainer.__init__ swallows
            # SystemExit(0), leaving the Trainer half-initialized and
            # crashing in train().
            os._exit(0)

        time.sleep(poll_interval)


def activate_cuda_device(ctx: InitContext) -> None:
    """Set the CUDA device — this is the CUDA context boundary."""
    if ctx._device_module is None:
        # baseline mode: device_module not yet set
        ctx._device_module, ctx._device_type = utils.device_module, utils.device_type
        ctx.device = torch.device(f"{ctx._device_type}:{int(os.environ['LOCAL_RANK'])}")
    ctx._device_module.set_device(ctx.device)


def set_determinism(ctx: InitContext) -> None:
    dist_utils.set_determinism(
        ctx.parallel_dims,
        ctx.device,
        ctx.job_config.debug,
        distinct_seed_mesh_dims=["pp"],
    )


def compute_init_device(ctx: InitContext) -> None:
    job_config = ctx.job_config
    if job_config.checkpoint.create_seed_checkpoint:
        ctx._init_device = "cpu"
        ctx.buffer_device = None
    elif job_config.training.enable_cpu_offload:
        ctx._init_device = "cpu"
        ctx.buffer_device = ctx._device_type
    else:
        ctx._init_device = ctx._device_type
        ctx.buffer_device = None


def apply_parallelisms_and_init_weights(ctx: InitContext) -> None:
    job_config = ctx.job_config
    parallel_dims = ctx.parallel_dims
    model = ctx._model
    model_args = ctx._model_args
    init_device = ctx._init_device

    if parallel_dims.pp_enabled:
        if not ctx.train_spec.pipelining_fn:
            raise RuntimeError(
                f"Pipeline Parallel is enabled but {job_config.model.name} "
                f"does not support pipelining"
            )

        (
            ctx.pp_schedule,
            ctx.model_parts,
            ctx.pp_has_first_stage,
            ctx.pp_has_last_stage,
        ) = ctx.train_spec.pipelining_fn(
            model,
            parallel_dims,
            job_config,
            ctx.device,
            model_args,
            ctx.train_spec.parallelize_fn,
            ctx.loss_fn,
        )
        del ctx._model

        if not job_config.leto.enable_rmp_gpu:
            for m in ctx.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    # pyrefly: ignore [not-callable]
                    m.init_weights(buffer_device=ctx.buffer_device)
                m.train()

        # pyrefly: ignore [bad-argument-type]
        ensure_pp_loss_visible(parallel_dims, job_config, ctx._color)
    else:
        model = ctx.train_spec.parallelize_fn(model, parallel_dims, job_config)

        if not job_config.leto.enable_rmp_gpu:
            model.to_empty(device=init_device)
            with torch.no_grad():
                # pyrefly: ignore [not-callable]
                model.init_weights(buffer_device=ctx.buffer_device)
            model.train()

        ctx.model_parts = [model]
        ctx._model = None
        ctx.pp_has_last_stage = True
        ctx.pp_has_first_stage = True

    ctx.ft_manager.maybe_set_all_reduce_hook(ctx.model_parts)


def log_device_memory_stats(ctx: InitContext) -> None:
    device_memory_monitor = ctx.metrics_processor.device_memory_monitor
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{ctx._device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )


def build_optimizers(ctx: InitContext) -> None:
    job_config = ctx.job_config
    ctx.optimizers = ctx.train_spec.build_optimizers_fn(
        ctx.model_parts, job_config.optimizer, ctx.parallel_dims, ctx.ft_manager
    )
    lr_steps = (
        job_config.training.max_steps
        if job_config.training.max_steps > 0
        else job_config.training.steps
    )
    ctx.lr_schedulers = ctx.train_spec.build_lr_schedulers_fn(
        ctx.optimizers, job_config.lr_scheduler, lr_steps
    )


def init_trainer_states(ctx: InitContext) -> None:
    ctx.step = 0
    ctx.ntokens_seen = 0
    ctx._prev_step_faulted = False


def _maybe_register_rmp_oom_reclaim(ctx: InitContext) -> None:
    """Active side: route RMP-server CUDA OOM through the standby reclaim.

    RMP allocations (model/optim pools, gradient-persistence tensors) happen
    in the RMP server process, so an OOM there never reaches the active's
    CUDACachingAllocator FreeMemoryCallback. Registered here — before the
    first RMP allocation at init — rather than with the broker install at
    train start, because the gradient tensors are allocated during THIS init
    task; a standby holding grants (e.g. after a restart, or leftover from a
    prior attempt) could otherwise starve them and crash the active."""
    cfg = ctx.job_config.leto
    if ctx.is_standby or not (
        cfg.enable_standby
        and cfg.progressive_init
        and int(cfg.progressive_reservation_margin_mb) > 0
    ):
        return
    try:
        from leto.launch.worker_controller_client import (
            kill_standby_for_oom_safeguard,
        )
        from leto.rmp.client import set_oom_reclaim_hook
        from torchtitan.components.init.progressive import get_free_mb
        from torchtitan.components.mem import reset_granted
    except ImportError:
        return
    rank = int(os.environ.get("RANK", -1))

    def _reclaim() -> bool:
        freed, _pid = kill_standby_for_oom_safeguard(get_free_mb(), 0, rank=rank)
        if freed:
            reset_granted()
        return freed

    set_oom_reclaim_hook(_reclaim)
    logger.info(f"[RMP] OOM-reclaim hook registered on active rank={rank}")


def init_rmp_and_resilient_opt(ctx: InitContext) -> None:
    job_config = ctx.job_config

    _maybe_register_rmp_oom_reclaim(ctx)

    ctx.rmp_manager = RmpManager(
        leto_config=job_config.leto,
        model_parts=ctx.model_parts,
        optimizers=ctx.optimizers,
        states={"train_state": ctx},
        lr_schedulers=ctx.lr_schedulers,
        dataloader=ctx.dataloader,
        device=ctx.device,
    )
    if job_config.leto.enable_rmp_gpu and ctx.is_standby:
        _standby_wait_for_allocation_flag(
            ctx.rmp_manager.rmp_client, FLAG_KIND_GPU,
        )

    ctx.rmp_restored = ctx.rmp_manager.maybe_init(ctx.buffer_device)

    # Wrap optimizers with resilient optimizer if RMP GPU is enabled
    assert not (
        job_config.leto.enable_rmp_gpu and job_config.leto.enable_cpu_snapshot_opt
    ), "enable_rmp_gpu and enable_cpu_snapshot_opt are mutually exclusive"

    if job_config.leto.enable_rmp_gpu and not job_config.leto.disable_resilient_opt:
        ctx._resilient_opt = ResilientOptimizer(
            ctx.optimizers,
            ctx.rmp_manager.rmp_client,
            ctx.device,
            model_parts=ctx.model_parts,
            parallel_dims=ctx.parallel_dims,
        )
        if ctx.rmp_restored:
            ctx.rmp_manager.restore_param_gradients(ctx.model_parts)
        if job_config.leto.resilient_opt_fault_injection:
            ctx._resilient_opt.enable_fault_injection(
                job_config.leto.resilient_opt_fault_injection_prob,
            )
    elif job_config.leto.enable_cpu_snapshot_opt:
        ctx._cpu_snapshot_opt = AsyncCpuSnapshotOptimizer(
            ctx.optimizers, ctx.device,
        )


def register_model_converter_hooks(ctx: InitContext) -> None:
    ctx.optimizers.register_step_post_hook(
        lambda *args, **kwargs: ctx._model_converters.post_optimizer_hook(
            ctx.model_parts
        )
    )
    ctx.metrics_processor.optimizers = ctx.optimizers
    ctx.metrics_processor.model_parts = ctx.model_parts


def init_expert_dist_tracker(ctx: InitContext) -> None:
    from torchtitan.train import ExpertDistTracker

    ctx._expert_dist_tracker = None
    if ctx.job_config.metrics.save_expert_dist:
        ctx._expert_dist_tracker = ExpertDistTracker(
            ctx.model_parts[0], ctx.job_config.job.dump_folder, dist.get_rank()
        )


def init_collective_manager_and_checkpoint(ctx: InitContext) -> None:
    job_config = ctx.job_config

    collective_manager = FsdpCollectiveManager()

    if job_config.checkpoint.use_gemini:
        # Gemini is decoupled from RMP: its CPU checkpoint pool is
        # process-owned shm, so there's no RMP CPU allocation to wait on and
        # no rmp_manager to hand down.
        ctx.checkpointer.lazy_init(
            model_parts=ctx.model_parts,
            optimizers=ctx.optimizers,
            lr_schedulers=ctx.lr_schedulers,
            parallel_dims=ctx.parallel_dims,
            collective_manager=collective_manager,
        )
    else:
        ctx.checkpointer = CheckpointManager(
            dataloader=ctx.dataloader,
            model_parts=ctx.model_parts,
            optimizers=ctx.optimizers,
            lr_schedulers=ctx.lr_schedulers,
            states={"train_state": ctx},
            checkpoint_config=job_config.checkpoint,
            sd_adapter=(
                ctx.train_spec.state_dict_adapter(
                    ctx._model_args, job_config.model.hf_assets_path
                )
                if ctx.train_spec.state_dict_adapter
                else None
            ),
            base_folder=job_config.job.dump_folder,
            ft_manager=ctx.ft_manager,
        )

    ctx.rmp_manager.init_gradient_allocator(collective_manager, ctx.model_parts)
    collective_manager.attach(ctx.model_parts)
    ctx._collective_manager = collective_manager


def build_validator(ctx: InitContext) -> None:
    job_config = ctx.job_config
    if not job_config.validation.enable:
        return

    assert ctx.train_spec.build_validator_fn is not None
    parallel_dims = ctx.parallel_dims

    pp_schedule, pp_has_first_stage, pp_has_last_stage = (
        (
            ctx.pp_schedule,
            ctx.pp_has_first_stage,
            ctx.pp_has_last_stage,
        )
        if parallel_dims.pp_enabled
        else (None, None, None)
    )

    ctx.validator = ctx.train_spec.build_validator_fn(
        job_config=job_config,
        dp_world_size=ctx._batch_degree,
        dp_rank=ctx._batch_rank,
        tokenizer=ctx.tokenizer,
        parallel_dims=parallel_dims,
        loss_fn=ctx.loss_fn,
        validation_context=ctx.train_context,
        maybe_enable_amp=ctx.maybe_enable_amp,
        metrics_processor=ctx.metrics_processor,
        pp_schedule=pp_schedule,
        pp_has_first_stage=pp_has_first_stage,
        pp_has_last_stage=pp_has_last_stage,
    )


def _nccl_eager_init_enabled(ctx: InitContext) -> bool:
    if not (_LETO_AVAILABLE and leto_is_standby()):
        return False
    eager_list = ctx.job_config.leto.eager_init_list
    if len(eager_list) == 0 or "none" in eager_list:
        return False
    return "all" in eager_list or "nccl" in eager_list


def _eager_init_mesh(ctx: InitContext, mesh_name: str) -> None:
    """Warm up NCCL for a single 1D mesh with a dummy all-reduce."""
    if not _nccl_eager_init_enabled(ctx):
        return
    mesh = ctx.parallel_dims.get_optional_mesh(mesh_name)
    if mesh is None or mesh.size() <= 1:
        return
    pg = mesh.get_group()
    if pg is None:
        return
    if mesh_name == "ep":
        ep_size = mesh.size()
        warmup_input = torch.zeros(ep_size, device=ctx.device)
        warmup_output = torch.zeros(ep_size, device=ctx.device)
        split_sizes = [1] * ep_size
        dist.all_to_all_single(
            warmup_output,
            warmup_input,
            output_split_sizes=split_sizes,
            input_split_sizes=split_sizes,
            group=pg,
        )

    warmup_tensor = torch.zeros(1, device=ctx.device)
    dist.all_reduce(warmup_tensor, group=pg)
    torch.cuda.synchronize()
    logger.info(f"Eagerly initialized NCCL for '{mesh_name}' mesh")


def eager_init_nccl_fsdp(ctx: InitContext) -> None:
    _eager_init_mesh(ctx, "fsdp")


def eager_init_nccl_ep(ctx: InitContext) -> None:
    _eager_init_mesh(ctx, "ep")


def eager_init_nccl_tp(ctx: InitContext) -> None:
    _eager_init_mesh(ctx, "tp")


def eager_init_nccl_pp(ctx: InitContext) -> None:
    _eager_init_mesh(ctx, "pp")

def eager_init_nccl_loss(ctx: InitContext) -> None:
    _eager_init_mesh(ctx, "loss")

def warmup_stages(ctx: InitContext) -> None:
    # Stage warmup is a standby-side feature: it pre-warms allocator/kernels
    # so a PROMOTED standby's first step is fast. The run's initial active is
    # about to run real steps anyway — warming it up is ~13s of pure init
    # cost. Skip it there, mirroring _nccl_eager_init_enabled; profile_init
    # keeps today's behavior so the profiling environment is unchanged.
    if (
        _LETO_AVAILABLE
        and not ctx.is_standby
        and not ctx.job_config.leto.profile_init
    ):
        logger.info("[warmup] skipping stage warmup on non-standby active")
        return
    maybe_warmup_stages(
        ctx.model_parts,
        ctx.job_config,
        loss_fn=ctx.loss_fn,
        pp_has_last_stage=getattr(ctx, "pp_has_last_stage", True),
        parallel_dims=ctx.parallel_dims,
    )

# ---------------------------------------------------------------------------
# Initialization sequences
# ---------------------------------------------------------------------------

REORDERED_SEQUENCE: list[Callable[[InitContext], None]] = [
    set_device_attr,
    compute_batch_info,
    init_ft_manager,
    init_garbage_collection,
    load_train_spec,
    build_tokenizer_and_dataloader,
    build_model_on_meta,
    apply_model_converters,
    build_metrics_processor,
    build_loss_fn_and_grad_accum,
    build_train_context,
    init_gemini_checkpoint_partial,
    # --- CUDA boundary ---
    activate_cuda_device,
    set_determinism,
    compute_init_device,
    apply_parallelisms_and_init_weights,
    log_device_memory_stats,
    build_optimizers,
    init_trainer_states,
    init_rmp_and_resilient_opt,
    register_model_converter_hooks,
    init_expert_dist_tracker,
    init_collective_manager_and_checkpoint,
    build_validator,
    eager_init_nccl_fsdp,
    eager_init_nccl_ep,
    eager_init_nccl_tp,
    eager_init_nccl_pp,
    eager_init_nccl_loss,
    warmup_stages,
]

def _parse_occupy_mbs(job_config: JobConfig) -> list[int]:
    """Parse leto.standby_test_occupy_mbs ("100,500,2048") into [100, 500, 2048]."""
    raw = str(getattr(job_config.leto, "standby_test_occupy_mbs", "") or "").strip()
    if not raw:
        return []
    return [int(tok) for tok in raw.split(",") if tok.strip()]


def _make_occupy_task(idx: int, mb: int) -> Callable[[InitContext], None]:
    """Synthetic standby-only ballast task (OOM-safeguard testing).

    Allocates and HOLDS `mb` MiB of GPU memory on standby ranks; no-op on the
    active and on a standby that has already been activated (a promoted active
    must not carry the ballast — see maybe_wait_for_resuming, which releases
    any held ballast on activation)."""

    def _task(ctx: InitContext) -> None:
        if not ctx.is_standby or getattr(ctx, "standby_activated", False):
            return
        n = (mb * 1024 * 1024) // 2  # bf16 = 2 bytes/elem
        t = torch.empty(n, dtype=torch.bfloat16, device="cuda")
        t.fill_(0)
        torch.cuda.synchronize()
        holds = getattr(ctx, "_occupy_holds", None)
        if holds is None:
            holds = []
            ctx._occupy_holds = holds
        holds.append(t)
        logger.info(
            f"[occupy] standby rank={os.environ.get('RANK')} holding +{mb}MiB "
            f"(task standby_test_occupy_{idx})"
        )

    _task.__name__ = f"standby_test_occupy_{idx}"
    return _task


def _release_occupy_holds(ctx: InitContext) -> None:
    """Free any standby ballast (called on activation/promotion)."""
    holds = getattr(ctx, "_occupy_holds", None)
    if not holds:
        return
    total_mb = sum(t.numel() * t.element_size() for t in holds) // (1024 * 1024)
    ctx._occupy_holds = None
    del holds
    torch.cuda.empty_cache()
    logger.info(
        f"[occupy] rank={os.environ.get('RANK')} released {total_mb}MiB "
        f"ballast on activation"
    )


BASELINE_SEQUENCE: list[Callable[[InitContext], None]] = [
    activate_cuda_device,
    # --- CUDA context set immediately ---
    set_device_attr,
    compute_batch_info,
    init_ft_manager,
    init_garbage_collection,
    set_determinism,
    load_train_spec,
    build_tokenizer_and_dataloader,
    build_model_on_meta,
    apply_model_converters,
    build_metrics_processor,
    build_loss_fn_and_grad_accum,
    build_train_context,
    init_gemini_checkpoint_partial,
    compute_init_device,
    apply_parallelisms_and_init_weights,
    log_device_memory_stats,
    build_optimizers,
    init_trainer_states,
    init_rmp_and_resilient_opt,
    register_model_converter_hooks,
    init_expert_dist_tracker,
    init_collective_manager_and_checkpoint,
    build_validator,
    eager_init_nccl_fsdp,
    eager_init_nccl_ep,
    eager_init_nccl_tp,
    eager_init_nccl_pp,
    eager_init_nccl_loss,
    warmup_stages,
]


_NVML_STATE: dict | None = None


def _get_gpu_mem_mb() -> float | None:
    """Return GPU memory used (MiB) by the current PID via pynvml, or None.

    Finds this process's physical GPU by scanning every NVML device for the
    current PID, then caches that handle. This is robust to CUDA_VISIBLE_DEVICES
    / hostfile `devices=` remapping: NVML indexes *physical* GPUs and ignores
    CUDA visibility, so resolving the handle by LOCAL_RANK (as before) queried
    the wrong card whenever `devices=` didn't start at 0 -> the per-PID lookup
    matched nothing and returned 0. Matching by PID sidesteps the mapping
    entirely (mirrors the reservation broker, which resolves by PCI bus id).
    """
    global _NVML_STATE
    try:
        from pynvml import (
            nvmlDeviceGetComputeRunningProcesses_v3,
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlInit,
        )

        my_pid = os.getpid()
        if _NVML_STATE is None:
            nvmlInit()
            _NVML_STATE = {"handle": None, "count": nvmlDeviceGetCount()}

        # Once we've found our GPU, query only it; otherwise scan all (the CUDA
        # context may not exist yet -> not on any GPU -> a correct 0).
        cached = _NVML_STATE["handle"]
        handles = (
            [cached]
            if cached is not None
            else [nvmlDeviceGetHandleByIndex(i) for i in range(_NVML_STATE["count"])]
        )
        for h in handles:
            used = 0
            found = False
            for p in nvmlDeviceGetComputeRunningProcesses_v3(h):
                if p.pid == my_pid and p.usedGpuMemory is not None:
                    used += p.usedGpuMemory
                    found = True
            if found:
                _NVML_STATE["handle"] = h  # cache our GPU for subsequent calls
                return used / (1024 * 1024)
        return 0.0
    except Exception:
        return None


def _device_free_mb() -> float | None:
    """Physical device-free MiB of THIS rank's GPU via pynvml.

    Deliberately avoids torch.cuda (a mem_get_info would create a ~400MiB
    CUDA context on a standby that hasn't reached activate_cuda_device).
    NVML indexes physical GPUs, so resolve through CUDA_VISIBLE_DEVICES +
    LOCAL_RANK; falls back to LOCAL_RANK when unset. Returns None on any
    failure (diagnostic only)."""
    try:
        from pynvml import (
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlInit,
        )

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cvd.strip():
            visible = [int(d) for d in cvd.split(",") if d.strip()]
            phys = visible[local_rank] if local_rank < len(visible) else local_rank
        else:
            phys = local_rank
        nvmlInit()
        mem = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(phys))
        return mem.free / (1024 * 1024)
    except Exception:
        return None


def _run_progressive_sequence(
    ctx: InitContext,
    sequence: list[Callable[[InitContext], None]],
    mode: str,
) -> None:
    """Run the init sequence in solver-scheduled order, gated per-task on the
    active's advance signal. Only called on standby ranks when progressive_init
    is on."""
    from torchtitan.components.init.progressive import (
        load_rank_deltas,
        load_solution,
        standby_register,
        try_advance,
    )

    job_config = ctx.job_config
    dump_folder = job_config.job.dump_folder
    rank = int(os.environ.get("RANK", "0"))

    name_to_fn = {fn.__name__: fn for fn in sequence}
    solution_variant = str(job_config.leto.progressive_solution)
    logger.info(
        f"[progressive] rank={rank} loading solution "
        f"(variant={solution_variant}) + profile from "
        f"{dump_folder}/init_profile/{mode}"
    )
    ordered_names = load_solution(dump_folder, mode, solution_variant)
    deltas = load_rank_deltas(dump_folder, mode, rank)
    logger.info(f"[progressive] rank={rank} loaded: {len(ordered_names)} tasks")

    missing_fn = [n for n in ordered_names if n not in name_to_fn]
    if missing_fn:
        raise RuntimeError(
            f"solution.json references tasks not in {mode} sequence: {missing_fn}"
        )
    missing_name = [n for n in name_to_fn if n not in set(ordered_names)]
    if missing_name:
        raise RuntimeError(
            f"{mode} sequence has tasks not in solution.json: {missing_name}. "
            f"Re-run profiling to refresh the schedule."
        )

    threshold_mb = float(job_config.leto.progressive_zero_delta_threshold_mb)
    poll_s = float(job_config.leto.progressive_poll_interval_ms) / 1000.0

    def _status_check() -> int:
        try:
            return poll_standby_status()
        except Exception:
            logger.warning("Error polling standby status", exc_info=True)
            return STANDBY_ACTION_TERMINATE

    # margin == 0 means the active does not run the reservation broker (see
    # Trainer._maybe_install_oom_safeguard), so the shm reservation handshake
    # would block forever with no one to answer. Run the init tasks in the
    # reordered (solver-scheduled) order, ungated — the grant-only OOM control.
    # Still poll standby status between tasks so TERMINATE stays responsive.
    margin_mb = int(job_config.leto.progressive_reservation_margin_mb)
    if margin_mb <= 0:
        logger.info(
            f"[progressive] rank={rank} reservation disabled (margin=0); "
            f"running {len(ordered_names)} tasks reordered, ungated"
        )
        for name in ordered_names:
            if _status_check() == STANDBY_ACTION_TERMINATE:
                logger.info("[progressive] standby terminated")
                os._exit(0)
            logger.info(f"[progressive] task={name} → running (ungated)")
            name_to_fn[name](ctx)
        return

    # Identify this standby instance to the active's broker before driving the
    # shm reservation handshake. (The C++/Python ledger layout is verified by
    # test_reservation.py, not at runtime, so the standby stays pure-Python.)
    standby_register(rank)

    # DIAGNOSTIC (env-gated, default off) for the standby steady-overhead
    # investigation (awsexps/measure_mem/fix_overhead). LETO_STATE_NOPOLL: build
    # all CPU (0-delta) init tasks unconditionally — same resident state as a
    # margin-parked standby — then just SLEEP (no RESERVE / gloo / status poll).
    # Isolates "held state" from "the try_advance polling machinery". Result:
    # state-alone ≈ base; state + polling (any frequency) = +21ms. The run must
    # have no fault (the standby is never promoted here).
    if os.environ.get("LETO_STATE_NOPOLL"):
        for name in ordered_names:
            if deltas.get(name, 0.0) >= threshold_mb:
                break  # stop at the first GPU task (no CUDA context built)
            name_to_fn[name](ctx)
        logger.info(
            f"[progressive] rank={rank} STATE_NOPOLL: CPU state built, "
            f"sleeping (no reservation handshake / gloo / status poll)"
        )
        while True:
            time.sleep(30.0)

    activated = False
    cumulative_mb = 0.0  # running target footprint of reserved tasks so far
    for name in ordered_names:
        if not activated:
            delta_mb = deltas.get(name, 0.0)
            if delta_mb >= threshold_mb:
                cumulative_mb += delta_mb
            logger.info(
                f"[progressive] rank={rank} next_task={name} "
                f"delta_mb={delta_mb:.1f} cumulative_mb={cumulative_mb:.1f}"
            )
            while True:
                outcome, extra = try_advance(
                    ctx.standby_gloo_pg,
                    delta_mb,
                    cumulative_mb,
                    threshold_mb,
                    poll_s,
                    status_check=_status_check,
                    protocol=str(ctx.job_config.leto.progressive_protocol),
                )
                if outcome == "advance":
                    break
                if outcome == "retry":
                    # A denied reservation cannot succeed until the active's
                    # memory situation changes; pacing retries at 1s keeps
                    # the broker handshake + gloo unanimity traffic
                    # negligible (immediate retries ran continuously during
                    # the 2026-07 AWS runs). Promotion stays responsive: the
                    # next _reserve poll runs status_check within ~1s.
                    # DIAGNOSTIC (env-gated, default 1.0): LETO_PARK_RETRY_S
                    # overrides the parked retry pacing. Used to show the
                    # standby overhead is FREQUENCY-INDEPENDENT (10s retry gives
                    # the same +21ms as 1s — verified via DENY-line spacing).
                    # Longer = fewer wakeups but slower promotion.
                    time.sleep(float(os.environ.get("LETO_PARK_RETRY_S", "1.0")))
                    continue
                if outcome == "status":
                    if extra == STANDBY_ACTION_ACTIVATE:
                        logger.info(
                            f"[progressive] activated during task={name}; "
                            f"running remaining tasks unconditionally"
                        )
                        activated = True
                        # Mark promotion so synthetic ballast tasks
                        # (standby_test_occupy_*) skip allocating and any
                        # already-held ballast is released before training.
                        ctx.standby_activated = True
                        _release_occupy_holds(ctx)
                        break
                    if extra == STANDBY_ACTION_TERMINATE:
                        logger.info("[progressive] standby terminated")
                        # os._exit (not sys.exit): @record on
                        # Trainer.__init__ swallows SystemExit(0).
                        os._exit(0)
            logger.info(
                f"[progressive] task={name} delta_mb={delta_mb:.1f} → running"
            )
            # Diagnostic: physical device free at the instant the granted
            # task starts allocating (pynvml — no CUDA-context side effect).
            # A grant whose task then OOMs shows here as
            # device_free < delta_mb at t0, i.e. the reservation was
            # admitted but never physically backed (trim-lag).
            free0 = _device_free_mb()
            if free0 is not None:
                logger.info(
                    f"[progressive] rank={rank} task={name} pre-run "
                    f"device_free={free0:.0f}MiB (delta_mb={delta_mb:.1f})"
                )
        name_to_fn[name](ctx)
        if not activated:
            free1 = _device_free_mb()
            if free1 is not None:
                logger.info(
                    f"[progressive] rank={rank} task={name} post-run "
                    f"device_free={free1:.0f}MiB"
                )


def run_init_sequence(job_config: JobConfig) -> InitContext:
    """Execute the initialization sequence for the configured mode."""
    mode = job_config.leto.init_mode
    sequences = {
        "reordered": REORDERED_SEQUENCE,
        "baseline": BASELINE_SEQUENCE,
    }
    if mode not in sequences:
        raise ValueError(
            f"Unknown init_mode: {mode!r}. Expected one of {list(sequences.keys())}"
        )
    is_standby = _LETO_AVAILABLE and leto_is_standby()
    profile = job_config.leto.profile_init and is_standby
    if profile:
        time.sleep(10)

    # Synthetic standby ballast tasks (OOM-safeguard testing). Appended to the
    # end of the sequence so they run after real init; they no-op on the
    # active, so both groups execute the same task list (and the profile /
    # solver schedule stays consistent with the runtime sequence).
    occupy_mbs = _parse_occupy_mbs(job_config)
    sequence = sequences[mode] + [
        _make_occupy_task(i, mb) for i, mb in enumerate(occupy_mbs)
    ]

    profile_records: list[dict] = []
    t0 = time.monotonic()

    ctx = InitContext(job_config=job_config)
    ctx.is_standby = is_standby

    # init_distributed always runs first; not part of the schedulable sequence.
    init_distributed(ctx)

    # Standby ranks need a CPU-only gloo PG to vote on advance decisions
    # between init tasks.
    if job_config.leto.enable_standby and is_standby:
        logger.info(f"[progressive] rank={os.environ.get('RANK')} creating gloo PG...")
        ctx.standby_gloo_pg = dist.new_group(backend="gloo")
        logger.info(f"[progressive] rank={os.environ.get('RANK')} gloo PG ready")

    progressive = (
        job_config.leto.progressive_init
        and is_standby
        and not profile
    )

    if progressive:
        _run_progressive_sequence(ctx, sequence, mode)
    else:
        for task_fn in sequence:
            if profile:
                mem_before = _get_gpu_mem_mb()
                t_start = time.monotonic() - t0

            task_fn(ctx)

            if profile:
                t_end = time.monotonic() - t0
                mem_after = _get_gpu_mem_mb()
                delta_mb = (
                    round(max(0.0, mem_after - mem_before), 1)
                    if mem_before is not None and mem_after is not None
                    else None
                )
                profile_records.append({
                    "task": task_fn.__name__,
                    "t_start_s": round(t_start, 4),
                    "t_end_s": round(t_end, 4),
                    "duration_s": round(t_end - t_start, 4),
                    "gpu_mem_before_mb": round(mem_before, 1) if mem_before is not None else None,
                    "gpu_mem_after_mb": round(mem_after, 1) if mem_after is not None else None,
                    "delta_mb": delta_mb,
                })

    if profile and profile_records:
        rank = int(os.environ.get("RANK", "0"))
        profile_dir = os.path.join(job_config.job.dump_folder, "init_profile", mode)
        os.makedirs(profile_dir, exist_ok=True)
        path = os.path.join(profile_dir, f"rank_{rank}.json")
        import json
        with open(path, "w") as f:
            json.dump({
                "init_mode": mode,
                "total_s": round(time.monotonic() - t0, 4),
                "tasks": profile_records,
            }, f, indent=2)
        logger.info(f"Init profile written to {path}")

        if rank == 0:
            from torchtitan.components.init.solver import solve
            result = solve(path, time_limit=60)
            names = result["names"]
            durations = result["durations"]
            memories = result["memories"]

            def _dump_solution(filename: str, order: list[int], cost: float,
                               reordered: bool) -> None:
                solution_path = os.path.join(profile_dir, filename)
                with open(solution_path, "w") as f:
                    json.dump({
                        "mode": mode,
                        "reordered": reordered,
                        "solver_cost": cost,
                        "tasks": [
                            {
                                "name": names[i],
                                "duration_s": durations[i],
                            }
                            for i in order
                        ],
                    }, f, indent=2)
                logger.info(
                    f"Init schedule solution written to {solution_path} "
                    f"(cost={cost:.1f})"
                )

            # solver-optimized order
            _dump_solution("solution.json", result["order"], result["cost"],
                           reordered=True)
            # A/B control: progressive gating with the BASELINE ordering (the
            # profile's recorded execution order) — isolates the benefit of
            # the solver's reordering. Cost computed with the same
            # prefix-memory objective for comparison.
            no_cost, cum = 0.0, 0.0
            for i in result["original_order"]:
                cum += memories[i]
                no_cost += cum * durations[i]
            _dump_solution("solution_no_reordering.json",
                           result["original_order"], no_cost, reordered=False)

    maybe_wait_for_resuming(ctx)
    return ctx


def wait_for_standby_init_profile(
    job_config: JobConfig, timeout_s: float | None = None
) -> bool:
    """Block until the standby group has finished writing its init profile.

    Counterpart to the profiling block in ``run_init_sequence``. On a
    ``leto.profile_init`` + ``leto.enable_standby`` run the standby group
    profiles the init sequence and writes
    ``init_profile/<mode>/{rank_*.json,solution.json}`` (consumed later by a
    ``leto.progressive_init`` run). The *active* group must not run training on
    such a run — its activation/optimizer allocations would OOM on top of the
    standby's still-resident init memory — but it also must not exit the moment
    init finishes: that trips the master's shutdown, which SIGKILLs the standby
    (``worker_controller._kill_standby``) before it can finish profiling. So the
    active group calls this to stay alive until the profile is on disk, then
    shuts down cleanly. Both groups' init complete independently here: every
    cross-group RMP allocation flag the standby waits on is set during the
    active's own init, so the active need not run a training step.

    Returns True once the full profile (solution + one file per rank) is
    present, or False if ``timeout_s`` elapses first (logged as an error; the
    active still proceeds to a clean shutdown).
    """
    if timeout_s is None:
        timeout_s = job_config.leto.profile_init_wait_timeout_s
    mode = job_config.leto.init_mode
    profile_dir = os.path.join(job_config.job.dump_folder, "init_profile", mode)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    expected = [os.path.join(profile_dir, "solution.json")] + [
        os.path.join(profile_dir, f"rank_{r}.json") for r in range(world_size)
    ]

    def _ready(path: str) -> bool:
        # Non-empty guards against observing a file mid-write.
        try:
            return os.path.getsize(path) > 0
        except OSError:
            return False

    deadline = time.monotonic() + timeout_s
    announced = False
    while not all(_ready(p) for p in expected):
        if time.monotonic() >= deadline:
            missing = [p for p in expected if not _ready(p)]
            logger.error(
                f"profile_init: timed out after {timeout_s:.0f}s waiting for the "
                f"standby init profile under {profile_dir}; {len(missing)} "
                f"file(s) still missing (e.g. {missing[:3]}). A progressive_init "
                f"run consuming this profile will fail until it is regenerated."
            )
            return False
        if not announced:
            logger.info(
                f"profile_init: active group init complete; holding (no training) "
                f"until the standby finishes profiling {world_size} rank(s) under "
                f"{profile_dir}"
            )
            announced = True
        time.sleep(1.0)
    logger.info(
        "profile_init: standby init profile complete; terminating active group"
    )
    return True
