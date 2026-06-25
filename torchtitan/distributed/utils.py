# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import math
import os
from abc import abstractmethod
from collections.abc import Iterable
from datetime import timedelta
from typing import NamedTuple, Protocol

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
import torch.distributed.tensor._random
import torch.distributed.tensor.parallel
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from torchtitan.config import Comm as CommConfig, Debug as DebugConfig, TORCH_DTYPE_MAP
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_module, device_type


def _dist_reduce(
    x: torch.Tensor,
    reduceOp: str,
    mesh: DeviceMesh | None,
    extra_pg: dist.ProcessGroup | None,
) -> float:
    """Perform distributed reduction on a tensor.

    Args:
        x (torch.Tensor): Input tensor.
        reduceOp (str): Reduce operation to perform.
        mesh (DeviceMesh | None): Device mesh to use for reduction.
            If None, no reduction is performed but simply convert the tensor to a float.
        extra_pg (dist.ProcessGroup, optional): Extra process group to use for reduction.
            Defaults to None. If provided, this all_reduce will be called for the extra
            process group, and then the result will be all_reduced for the mesh.
    """
    if isinstance(x, DTensor):
        # functional collectives do not support DTensor inputs
        x = x.full_tensor()

    if extra_pg is not None:
        x = funcol.all_reduce(x, reduceOp=reduceOp, group=extra_pg)

    if mesh is None:
        return x.item()

    assert x.numel() == 1  # required by `.item()`
    return funcol.all_reduce(x, reduceOp=reduceOp, group=mesh).item()


# TODO: rename this to maybe_dist_max
def dist_max(
    x: torch.Tensor,
    mesh: DeviceMesh | None = None,
    extra_pg: dist.ProcessGroup | None = None,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.MAX.name, mesh=mesh, extra_pg=extra_pg
    )


def dist_sum(
    x: torch.Tensor,
    mesh: DeviceMesh | None = None,
    extra_pg: dist.ProcessGroup | None = None,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.SUM.name, mesh=mesh, extra_pg=extra_pg
    )


def dist_mean(
    x: torch.Tensor,
    mesh: DeviceMesh | None = None,
    extra_pg: dist.ProcessGroup | None = None,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.AVG.name, mesh=mesh, extra_pg=extra_pg
    )


def set_determinism(
    parallel_dims: ParallelDims,
    device: torch.device,
    debug_config: DebugConfig,
    distinct_seed_mesh_dims: list[str],
) -> None:
    """
    Set the same DTensor manual seed for all dimensions in world mesh, but only different seeds
    across dimensions denoted by `distinct_seed_mesh_dims`. An example use case is pipeline parallelism,
    where we want to have the same seed across SPMD groups, but different seeds across PP groups.

    Currently, does not set seeds for the CUDA RNG since TorchTitan always uses DTensor for SPMD parallelisms,
    and DTensor manages its own RNG tracker, but we could extend to support both if needed.

    Set Determinism flags for increased reproducibility with loss of performance.

    Args:
        world_mesh: Device mesh for distributed training
        device: Device to use
        debug_config: Debug config to use
        distinct_seed_mesh_dims: List of mesh dimension names to have distinct seeds across.
    """
    if debug_config.deterministic:
        logger.info("Deterministic algorithm enabled (expect perf degradation).")
        torch.use_deterministic_algorithms(True)
        torch.use_deterministic_algorithms(
            True, warn_only=debug_config.deterministic_warn_only
        )
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # env var for deterministic CuBLAS
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Ensure flex_attention is compiled without max-autotune. This is needed to ensure
        # reproducibility, since the autotune results may not be deterministic.
        from torch.nn.attention.flex_attention import flex_attention

        from torchtitan.models.attention import FlexAttentionWrapper

        FlexAttentionWrapper._compiled_flex_attn = torch.compile(flex_attention)

    seed = debug_config.seed
    if parallel_dims.world_size == 1:
        if seed is not None:
            torch.manual_seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
            logger.debug(f"Single-process job using seed: {seed}")
        return

    # to ensure we can control which ranks have same or different seeds, all ranks agree on a starting seed.
    # if user provides one, we use this. Otherwise rank 0 rolls the dice and everyone else uses that.
    if seed is None:
        # Extract the seed for torch's main generator on rank 0 and standardizes on using that to build
        # seeds for unique SPMD groups
        seed_tensor = torch.get_rng_state()[:8].to(device)
        torch.distributed.broadcast(seed_tensor, src=0)
        seed = seed_tensor.to("cpu").view(torch.uint64).item()
    assert isinstance(seed, int)

    # Set distinct seed for each rank in mesh dimensions, with dimension names provided by `distinct_seed_mesh_dims`
    # For PP + SPMD cases, we want to separate the world into the SPMD mesh and the PP mesh,
    # and choose a unique seed for each rank on the PP mesh.
    # We support multiple distinct dimensions by adding each distinct dimension's local rank to the seed.
    distinct_seed_meshes = [
        parallel_dims.get_optional_mesh(dim) for dim in distinct_seed_mesh_dims
    ]
    distinct_seed_meshes = [mesh for mesh in distinct_seed_meshes if mesh is not None]
    assert all(mesh is not None for mesh in distinct_seed_meshes)

    if distinct_seed_meshes:
        # Each dimension contributes: local_rank * (product of all previous dimension sizes)
        # This guarantees uniqueness like multi-dimensional array indexing
        seed_offset = 0
        cumulative_size = 1

        for distinct_mesh in distinct_seed_meshes:
            local_rank = distinct_mesh.get_local_rank()
            # Add contribution from this dimension
            seed_offset += local_rank * cumulative_size
            # Update cumulative size for next dimension
            cumulative_size *= distinct_mesh.size()

        seed += seed_offset
        seed %= 2**64

        logger.debug(
            f"Distinct dims {distinct_seed_mesh_dims}, Global rank {c10d.get_rank()} using seed: {seed}"
        )

    else:
        logger.debug(f"Global Rank {c10d.get_rank()} using seed: {seed}")

    # The native RNGs and python RNG may not be important, except for the 1-D PP case, but we seed them for consistency.
    torch.manual_seed(seed)
    # PYTHONHASHSEED can be a decimal number in the range [0, 2**32 - 1]
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)

    # As long as we are not in the 1-D (PP-only) case, we will have a seed to use for
    # all ranks of the SPMD mesh. If PP is also used, this seed is unique per PP rank.
    # TODO: remove the need of passing in a mesh once
    # torch.distributed.tensor._random.manual_seed doesn't require a mesh input.
    if parallel_dims.world_size > parallel_dims.pp:
        # We just need to pass the world_mesh as the device_id is the only information
        # this API uses.
        torch.distributed.tensor._random.manual_seed(seed, parallel_dims.world_mesh)


def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: list[torch.Tensor],
    cp_seq_dims: list[int],
    cp_no_restore_buffers: set[torch.Tensor],
    cp_rotate_method: str,
):
    try:
        from torch.distributed.tensor.experimental import context_parallel
        from torch.distributed.tensor.experimental._attention import set_rotate_method
    except ImportError as e:
        raise ValueError(
            f"PyTorch version {torch.__version__} does not include the experimental "
            "Context Parallel API. Please update to a newer version."
        ) from e

    set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


class TrainContext(Protocol):
    @abstractmethod
    def __call__(self) -> contextlib.AbstractContextManager[None]:
        pass


def get_train_context(enable_loss_parallel: bool) -> TrainContext:
    @contextlib.contextmanager
    def context():
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())

            yield

    return context


def maybe_enable_amp(
    parallel_dims: ParallelDims, mixed_precision_param: str, device_type: str
) -> contextlib.AbstractContextManager[None]:
    if parallel_dims.fsdp_enabled:
        # FSDP handles mixed precision internally
        logger.info("Mixed precision training is handled by fully_shard")
        return contextlib.nullcontext()
    else:
        if parallel_dims.tp_enabled or parallel_dims.pp_enabled:
            logger.warning(
                "Mixed precision training with TP or PP is only supported when FSDP/HSDP/CP is enabled."
            )
            logger.info("Mixed precision training is disabled")
            return contextlib.nullcontext()
        else:
            # the following code will only be executed for DDP or single-device training
            logger.info("Mixed precision training is handled by AMP")
            # pyrefly: ignore [bad-return]
            return torch.autocast(
                device_type,
                dtype=TORCH_DTYPE_MAP[mixed_precision_param],
            )


def init_fake_mode(world_size: int, comm_mode: str = "fake_backend"):
    """Initialize fake backend

    Args:
        world_size: The number of GPUs to simulate
        comm_mode: Communication mode ("fake_backend" or "local_tensor")

    Returns:
        The world size
    """
    torch.distributed.init_process_group(
        "fake",
        rank=0,
        world_size=world_size,
    )

    # If local_tensor mode is enabled, initialize LocalTensorMode context
    if comm_mode == "local_tensor":
        from torch.distributed import _local_tensor

        lm = _local_tensor.LocalTensorMode(world_size)
        lm.__enter__()


def init_distributed(
    comm_config: CommConfig,
    enable_cpu_backend: bool = False,
    base_folder: str = "",
    ranks: list[int] | None = None,
) -> int:
    if comm_config.mode in ("fake_backend", "local_tensor"):
        ngpu_str = os.environ.get("NGPU")
        if ngpu_str is None:
            raise ValueError(
                f"NGPU environment variable must be set when using comm_mode={comm_config.mode}"
            )
        try:
            world_size = int(ngpu_str)
        except ValueError as e:
            raise ValueError(
                f"NGPU environment variable must be a valid integer, got: {ngpu_str}"
            ) from e
        init_fake_mode(world_size, comm_config.mode)
        return world_size

    def _warn_overwrite_env(env, val):
        if env in os.environ:
            logger.warning(
                f"ENV[{env}] = {os.environ[env]} will be overridden to {val} based on job config"
            )
        os.environ[env] = val

    def _get_distributed_backend(enable_cpu_backend):
        backend = "nccl"
        if device_type in torch.distributed.Backend.default_device_backend_map:
            backend = torch.distributed.Backend.default_device_backend_map.get(
                device_type
            )
        if enable_cpu_backend:
            backend = f"{device_type}:{backend},cpu:gloo"
        return backend

    TRACE_BUFFER_SIZE = "TORCH_FR_BUFFER_SIZE"
    TRACE_FILE = "TORCH_FR_DUMP_TEMP_FILE"
    DUMP_ON_TIMEOUT = "TORCH_NCCL_DUMP_ON_TIMEOUT"
    ASYNC_ERROR_HANDLING = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
    SKIP_CLEANUP = "3"

    # FlightRecorder is incompatible with =1 mode where watchdog aborts work, must use =3 (skipcleanup)
    # to get flight recorder dumps. See https://github.com/pytorch/pytorch/issues/121055
    # This could be done only when flight recorder is enabled, but its nice to be consistent to avoid subtle
    # behavior differences
    _warn_overwrite_env(ASYNC_ERROR_HANDLING, SKIP_CLEANUP)

    # enable torch nccl flight recorder in the mode that would dump files if timeout is detected
    _warn_overwrite_env(TRACE_BUFFER_SIZE, str(comm_config.trace_buf_size))
    if comm_config.trace_buf_size > 0:
        # dump on timeout by default if trace buffer is enabled
        _warn_overwrite_env(DUMP_ON_TIMEOUT, "1")
        dump_dir = os.path.join(base_folder, comm_config.save_traces_folder)
        prefix = comm_config.save_traces_file_prefix
        os.makedirs(dump_dir, exist_ok=True)
        _warn_overwrite_env(TRACE_FILE, f"{dump_dir}/{prefix}")

    torch.distributed.init_process_group(
        backend=_get_distributed_backend(enable_cpu_backend),
        timeout=timedelta(seconds=comm_config.init_timeout_seconds),
        _ranks=ranks if ranks is not None else [],
    )

    return torch.distributed.get_world_size()


def set_pg_timeouts(
    timeout: timedelta,
    parallel_dims: ParallelDims,
):
    """
    Sets the timeout for all PGs in the provided mesh, and the default (world) group.

    Note: synchronizes via a barrier, before changing the timeouts. This is important, because
    otherwise you may face a race where the slow rank has not reached the timeout reduction point
    yet due to slow operations permitted under the old timeout value, but other faster ranks may
    start issuing collectives under the new shorter timeout and then immediately timeout.
    """
    logger.info(
        f"Synchronizing and adjusting timeout for all ProcessGroups to {timeout}"
    )
    # Ensure that all the ranks have reached the point of setting the new timeout-
    # otherwise, some ranks may issue collectives with the new/shorter timeout and
    # those may time out, before other ranks have finished with initialization done
    # under the old/slow timeout.
    torch.distributed.barrier(device_ids=[device_module.current_device()])
    device_module.synchronize()

    # None represents the 'default' PG, not part of the mesh
    groups: list[torch.distributed.ProcessGroup | None] = [
        mesh.get_group()
        for mesh in parallel_dims.get_all_one_dimensional_meshes().values()
    ] + [None]
    for group in groups:
        torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)


class ClipLocals(NamedTuple):
    """Pre-reduction local norm contributions for ``clip_grad_norm_``.

    ``locals`` holds the values to persist into RMP *before* the reduction
    collective(s); ``ep_params`` / ``non_ep_params`` are the EP split (None
    when ``ep_enabled`` was False) and are used only by the in-place stock
    scaling — the resilient path ignores them.

    EP layout: ``locals = [ep_local, non_ep_local]``
    Dense layout: ``locals = [total_local]``

    Each entry is the raw ``torch.nn.utils.get_total_norm(...)`` result
    (a 0-dim ``_NormPartial`` DTensor, or a non-DTensor 0-dim tensor when
    a rank has no grads in that group — e.g. PP+EP corner case).
    """

    locals: list[torch.Tensor]
    ep_params: list[torch.Tensor] | None
    non_ep_params: list[torch.Tensor] | None


def _apply_pp_norm_reduce(
    total_norm: torch.Tensor,
    norm_type: float,
    pp_mesh: DeviceMesh | None,
) -> torch.Tensor:
    """Optional PP all-reduce step shared by all reducers below."""
    if pp_mesh is None:
        return total_norm
    if math.isinf(norm_type):
        dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
    else:
        total_norm **= norm_type  # pyrefly: ignore[unsupported-operation]
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
        total_norm **= 1.0 / norm_type  # pyrefly: ignore[unsupported-operation]
    return total_norm


def _reduce_clip_from_locals_dense(
    total_local: torch.Tensor,
    max_norm: float,
    norm_type: float,
    pp_mesh: DeviceMesh | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce a single pre-reduction local to ``(total_norm, clip_coef)``.

    Mirrors the original non-EP tail of ``clip_grad_norm_``: ``full_tensor``
    (collectives over the param mesh, if a DTensor), optional PP all-reduce,
    same 1e-6 eps + clamp as ``torch.nn.utils.clip_grads_with_norm_``.
    Used by stock, resilient-normal, and resilient-recovery — bit-identical
    by construction (deterministic collective replay over identical inputs).
    """
    total_norm = total_local
    if isinstance(total_norm, DTensor):
        # Reach here if any non-PP parallelism is used. If only PP, total_norm
        # is already a local tensor and we skip straight to the PP all-reduce.
        total_norm = total_norm.full_tensor()
    total_norm = _apply_pp_norm_reduce(total_norm, norm_type, pp_mesh)
    clip_coef_clamped = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    return total_norm, clip_coef_clamped


def _reduce_clip_from_locals_ep(
    ep_local: torch.Tensor,
    non_ep_local: torch.Tensor,
    max_norm: float,
    norm_type: float,
    pp_mesh: DeviceMesh | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce ep + non-ep pre-reduction locals to ``(total_norm, clip_coef)``.

    Same operation sequence as the original ``_clip_grad_norm_with_ep`` tail:
    per-group ``full_tensor`` (collectives over the ep / non-ep param meshes),
    p-norm combine, optional PP all-reduce, clamp.
    """
    ep_norm = ep_local.full_tensor() if isinstance(ep_local, DTensor) else ep_local
    non_ep_norm = (
        non_ep_local.full_tensor()
        if isinstance(non_ep_local, DTensor)
        else non_ep_local
    )
    if math.isinf(norm_type):
        total_norm = torch.maximum(ep_norm, non_ep_norm)
    else:
        total_norm = (
            ep_norm**norm_type  # pyrefly: ignore[unsupported-operation]
            + non_ep_norm**norm_type
        )
        total_norm **= 1.0 / norm_type  # pyrefly: ignore[unsupported-operation]
    total_norm = _apply_pp_norm_reduce(total_norm, norm_type, pp_mesh)
    clip_coef_clamped = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    return total_norm, clip_coef_clamped


@torch.no_grad()
def clip_compute_locals(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    *,
    ep_enabled: bool,
) -> ClipLocals:
    """Compute the pre-reduction ``_NormPartial`` locals for the resilient
    clip path — the per-rank scalars that must be persisted to RMP *before*
    the reduction collective runs.

    No collectives are issued. The caller persists ``ClipLocals.locals``
    into RMP, then calls :func:`clip_reduce_from_locals` (which runs the
    actual collectives + clamp).

    Why split: the collective is the synchronization point. If any rank
    advances past it (counter bumps), then every rank entered the collective
    — which means every rank executed *this* function and its persist
    immediately after. So persistence is witnessed by collective completion:
    every rank's RMP holds the fresh ``locals`` for the resume step, and
    recovery can re-run the same reducer over them with no peer reads and
    no grads consulted (immune to advanced-rank mutated grads).
    """
    if ep_enabled:
        ep_params: list[torch.Tensor] = []
        non_ep_params: list[torch.Tensor] = []
        ep_grads: list[torch.Tensor] = []
        non_ep_grads: list[torch.Tensor] = []
        for p in parameters:
            if p.grad is None:
                continue
            assert isinstance(p, DTensor) and isinstance(p.grad, DTensor)
            # pyrefly: ignore[not-iterable]
            if "ep" in p.device_mesh.mesh_dim_names:
                ep_params.append(p)
                ep_grads.append(p.grad)
            else:
                non_ep_params.append(p)
                non_ep_grads.append(p.grad)
        ep_local = torch.nn.utils.get_total_norm(
            ep_grads, norm_type, error_if_nonfinite, foreach
        )
        non_ep_local = torch.nn.utils.get_total_norm(
            non_ep_grads, norm_type, error_if_nonfinite, foreach
        )
        return ClipLocals(
            locals=[ep_local, non_ep_local],
            ep_params=ep_params,
            non_ep_params=non_ep_params,
        )

    if isinstance(parameters, torch.Tensor):
        parameters_list: list[torch.Tensor] = [parameters]
    else:
        parameters_list = list(parameters)
    grads = [p.grad for p in parameters_list if p.grad is not None]
    total_local = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )
    return ClipLocals(locals=[total_local], ep_params=None, non_ep_params=None)


@torch.no_grad()
def clip_reduce_from_locals(
    locals_list: list[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    pp_mesh: DeviceMesh | None = None,
    *,
    ep_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce per-rank locals (live or recovery-reconstructed) to
    ``(total_norm, clip_coef_clamped)``.

    The single reducer shared by the stock clip path, the resilient normal
    path, and the resilient recovery path. Bit-identity rests on
    deterministic collective replay over identical inputs (mesh, group, op,
    and persisted locals are all the same at normal and recovery time).
    """
    if ep_enabled:
        assert len(locals_list) == 2
        return _reduce_clip_from_locals_ep(
            locals_list[0], locals_list[1], max_norm, norm_type, pp_mesh
        )
    assert len(locals_list) == 1
    return _reduce_clip_from_locals_dense(
        locals_list[0], max_norm, norm_type, pp_mesh
    )


@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
    ep_enabled: bool = False,
    compute_only: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: Pipeline Parallel device mesh. If not None, will reduce gradient norm across PP stages.
        ep_dense_params_mesh_ndim: Mesh ndim of the dense params when EP is used. If EP is not used,
            set it to ``None``.
        compute_only: if True, compute the total norm and the (clamped) clip
            coefficient but do **not** scale the gradients in place. Returns
            ``(total_norm, clip_coef_clamped)`` instead of ``total_norm``.
            Used by ResilientOptimizer so the clip scaling can be folded into
            its fault-recoverable chunked step (gradients are never mutated
            outside the replayable region). Default ``False`` keeps the
            original in-place behavior and single-tensor return.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
        If ``compute_only`` is True, returns
        ``(total_norm, clip_coef_clamped)`` and leaves gradients untouched.

    """
    if ep_enabled:
        return _clip_grad_norm_with_ep(
            parameters,
            max_norm,
            norm_type,
            error_if_nonfinite,
            foreach,
            pp_mesh,
            compute_only,
        )

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)

    # Split into compute-locals (no collective) -> reduce-from-locals
    # (collective + clamp). Same two-phase contract the resilient path
    # uses; keeps stock and resilient bit-identical by routing through
    # the same reducer over the same locals.
    clip_locals = clip_compute_locals(
        parameters, norm_type, error_if_nonfinite, foreach, ep_enabled=False
    )
    total_norm, clip_coef_clamped = clip_reduce_from_locals(
        clip_locals.locals, max_norm, norm_type, pp_mesh, ep_enabled=False
    )

    if compute_only:
        # Do NOT scale grads here — the resilient chunked step does
        # `grad * clip_coef -> scratch -> fused AdamW` inside its
        # fault-recoverable replay region.
        return total_norm, clip_coef_clamped

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


@torch.no_grad()
def _clip_grad_norm_with_ep(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float,
    error_if_nonfinite: bool,
    foreach: bool | None,
    pp_mesh: DeviceMesh | None,
    compute_only: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # Split into compute-locals (no collective) -> reduce-from-locals
    # (collective + combine + clamp). The compute-locals helper also
    # returns the ep / non_ep param split used by the in-place scaling
    # below — so we do the split exactly once.
    clip_locals = clip_compute_locals(
        parameters, norm_type, error_if_nonfinite, foreach, ep_enabled=True
    )
    total_norm, clip_coef_clamped = clip_reduce_from_locals(
        clip_locals.locals, max_norm, norm_type, pp_mesh, ep_enabled=True
    )

    if compute_only:
        # Resilient path: defer scaling into ResilientOptimizer's chunk step.
        return total_norm, clip_coef_clamped

    torch.nn.utils.clip_grads_with_norm_(clip_locals.ep_params, max_norm, total_norm, foreach)
    torch.nn.utils.clip_grads_with_norm_(clip_locals.non_ep_params, max_norm, total_norm, foreach)

    return total_norm
