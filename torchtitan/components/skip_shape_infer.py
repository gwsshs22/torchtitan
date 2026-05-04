# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Stage input/output recording and shape-loading utilities for pipeline parallel training.

This module provides functionality to:
1. Record stage inputs AND outputs (shapes, dtypes, devices) during the first training iteration
2. Load recorded shapes as meta tensors to pass to PipelineStage, bypassing PyTorch's
   runtime _shape_inference (which runs a full dummy forward pass across all ranks).
"""

import gc
import json
import os
import types
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from torchtitan.tools.logging import logger


def _placement_to_metadata(p: Any) -> dict[str, Any]:
    from torch.distributed.tensor import Partial, Replicate, Shard

    if isinstance(p, Shard):
        return {"type": "Shard", "dim": int(p.dim)}
    if isinstance(p, Replicate):
        return {"type": "Replicate"}
    if isinstance(p, Partial):
        return {"type": "Partial", "reduce_op": str(p.reduce_op)}
    return {"type": "Unknown"}


def _metadata_to_placement(d: dict[str, Any]) -> Any:
    from torch.distributed.tensor import Partial, Replicate, Shard

    kind = d.get("type")
    if kind == "Shard":
        return Shard(d["dim"])
    if kind == "Replicate":
        return Replicate()
    if kind == "Partial":
        return Partial(d.get("reduce_op", "sum"))
    raise ValueError(f"Unknown placement metadata: {d}")


def _tensor_to_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    """Convert a tensor to metadata dict with shape, dtype, and device.

    For DTensor inputs, additionally capture the local shape, mesh dim
    names, and placements so the reader can reconstruct a matching
    DTensor at warmup time. Without this, warmup inputs would be plain
    ``torch.Tensor`` even for graphs that real training calls with
    DTensors, which would cause AOTAutograd's ``subclass_inp_meta`` to
    mismatch and crash in ``runtime_unwrap_tensor_subclasses``.
    """
    from torch.distributed.tensor import DTensor

    if isinstance(tensor, DTensor):
        local = tensor.to_local()
        mesh = tensor.device_mesh
        mesh_dim_names = list(mesh.mesh_dim_names) if mesh.mesh_dim_names else []
        return {
            "type": "DTensor",
            "shape": list(tensor.shape),
            "local_shape": list(local.shape),
            "dtype": str(tensor.dtype),
            "device": str(local.device),
            "mesh_dim_names": mesh_dim_names,
            "placements": [
                _placement_to_metadata(p) for p in tensor.placements
            ],
        }
    return {
        "type": "Tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _metadata_to_meta_tensor(metadata: dict[str, Any]) -> torch.Tensor:
    """Create a meta-device tensor from recorded metadata (shape + dtype only)."""
    dtype_str = metadata["dtype"]
    dtype = getattr(torch, dtype_str.split(".")[-1])
    return torch.empty(metadata["shape"], dtype=dtype, device="meta")

def _remap_device(device_str: str) -> str:
    """Remap a recorded cuda device index to the current rank's local cuda
    device. Recording captured absolute local indices from the recording
    topology (e.g. ``cuda:2`` when the recording node had 4 GPUs). A replay
    run with a different #GPUs-per-node layout would crash with "invalid
    device ordinal" if we used the recorded index verbatim.
    """
    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
    return device_str


# Module-level reference to the current rank's ``ParallelDims`` during
# warmup. ``_metadata_to_tensor`` reads this when encountering a DTensor
# metadata entry so it can reconstruct a DTensor with the matching mesh.
_current_parallel_dims: Any = None


def _metadata_to_tensor(metadata: dict[str, Any]) -> torch.Tensor:
    """Create a zero tensor from metadata."""
    dtype_str = metadata["dtype"]
    # Parse dtype string like "torch.float32" to actual dtype
    dtype = getattr(torch, dtype_str.split(".")[-1])

    device_str = _remap_device(metadata["device"])

    if metadata.get("type") == "DTensor":
        return _metadata_to_dtensor(metadata, dtype, device_str)

    # Use zeros to support all dtypes including integer types (e.g. torch.int64)
    tensor = torch.zeros(metadata["shape"], dtype=dtype, device=device_str)
    # Enable gradient tracking for compile warmup
    if dtype.is_floating_point:
        tensor.requires_grad = True

    return tensor


def _metadata_to_dtensor(
    metadata: dict[str, Any],
    dtype: torch.dtype,
    device_str: str,
) -> torch.Tensor:
    """Reconstruct a DTensor from recorded metadata using the current rank's
    mesh. Falls back to a plain tensor if no matching mesh is available
    (e.g. parallel_dims not provided or dim names missing).
    """
    from torch.distributed.tensor import DTensor

    local_shape = metadata.get("local_shape", metadata["shape"])
    mesh_dim_names = tuple(metadata.get("mesh_dim_names", ()))
    placements = tuple(
        _metadata_to_placement(p) for p in metadata.get("placements", [])
    )

    local = torch.zeros(local_shape, dtype=dtype, device=device_str)
    if dtype.is_floating_point:
        local.requires_grad = True

    mesh = None
    if _current_parallel_dims is not None and mesh_dim_names:
        try:
            mesh = _current_parallel_dims.get_optional_mesh(list(mesh_dim_names))
        except Exception as e:
            logger.warning(
                f"Fake warmup: could not resolve mesh {mesh_dim_names}: "
                f"{type(e).__name__}: {e}"
            )
    if mesh is None or not placements:
        # Fall back to the raw local tensor; AOTAutograd will see a plain
        # tensor at this input slot and compile a non-subclass version.
        # The guard system will force a recompile at real training time
        # when a real DTensor is passed.
        return local

    return DTensor.from_local(
        local,
        device_mesh=mesh,
        placements=placements,
        run_check=False,
    )


def _block_mask_to_metadata(mask: BlockMask) -> dict[str, Any]:
    """Serialize a BlockMask to a JSON-compatible dict (shape + device only)."""
    seq_len_q, seq_len_kv = mask.shape[-2], mask.shape[-1]
    # Infer device from the first internal tensor.
    device = str(mask.kv_num_blocks.device)
    return {
        "type": "BlockMask",
        "seq_len_q": seq_len_q,
        "seq_len_kv": seq_len_kv,
        "device": device,
    }


def _metadata_to_block_mask(metadata: dict[str, Any]) -> BlockMask:
    """Reconstruct a synthetic causal BlockMask from recorded metadata.

    Uses torchtitan's real ``get_causal_mask_mod`` factory so the resulting
    ``mask_mod`` closure has the same ``__code__`` object id as real
    training's mask_mod. If Dynamo later traces flex_attention with this
    BlockMask, the guards it emits on ``mask_mod.__code__`` match what
    real training produces, so the warmup-compiled cache entry can be
    reused. A locally-defined ``causal_mask`` would produce a different
    code object id and force recompilation on every real training call.
    """
    from torchtitan.models.attention import get_causal_mask_mod

    seq_len_q = metadata["seq_len_q"]
    seq_len_kv = metadata["seq_len_kv"]
    device = _remap_device(metadata["device"])

    return create_block_mask(
        get_causal_mask_mod(),
        B=None,
        H=None,
        Q_LEN=seq_len_q,
        KV_LEN=seq_len_kv,
        device=device,
    )


def _serialize_arg(arg: Any) -> Any:
    """Serialize a single arg to JSON-compatible format.

    Recurses into dict / list / tuple containers so that composite inputs like
    ``attention_masks: dict[str, BlockMask]`` round-trip correctly. Primitives
    (None, bool, int, float, str) are wrapped so they can be distinguished from
    "unsupported type" on the deserialize side.
    """
    if isinstance(arg, torch.Tensor):
        return _tensor_to_metadata(arg)
    if isinstance(arg, BlockMask):
        return _block_mask_to_metadata(arg)
    if isinstance(arg, dict):
        return {
            "type": "dict",
            "items": {k: _serialize_arg(v) for k, v in arg.items()},
        }
    if isinstance(arg, (list, tuple)):
        return {
            "type": "tuple" if isinstance(arg, tuple) else "list",
            "items": [_serialize_arg(v) for v in arg],
        }
    if arg is None or isinstance(arg, (bool, int, float, str)):
        return {"type": "primitive", "value": arg}
    return None


def _serialize_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Serialize args and kwargs to JSON-compatible format."""
    return {
        "args": [_serialize_arg(arg) for arg in args],
        "kwargs": {k: _serialize_arg(v) for k, v in kwargs.items()},
    }


def _serialize_output(output: Any) -> dict[str, Any]:
    """
    Serialize a stage's forward output to JSON-compatible format.

    Handles a single tensor or a tuple/list of tensors.
    Non-tensor values are stored as null.
    """
    if isinstance(output, torch.Tensor):
        outputs = (output,)
    elif isinstance(output, (tuple, list)):
        outputs = tuple(output)
    else:
        outputs = (output,)

    return {
        "args": [
            _tensor_to_metadata(o) if isinstance(o, torch.Tensor) else None
            for o in outputs
        ],
        "kwargs": {},
    }


def _deserialize_as_meta(metadata: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    """
    Deserialize a stage's recorded args metadata into a tuple of meta tensors.
    Non-tensor entries (dicts, primitives, BlockMasks, None) are skipped.
    Both plain ``Tensor`` and ``DTensor`` entries are converted to meta
    tensors (using the global shape for DTensors).
    """
    result: list[torch.Tensor] = []
    for entry in metadata["args"]:
        if isinstance(entry, dict) and entry.get("type") in ("Tensor", "DTensor"):
            result.append(_metadata_to_meta_tensor(entry))
    return tuple(result)


def _deserialize_arg(meta: Any) -> Any:
    """Deserialize a single recorded arg back to a synthetic value."""
    if meta is None:
        return None
    if not isinstance(meta, dict):
        return None
    kind = meta.get("type")
    if kind == "Tensor" or kind == "DTensor":
        return _metadata_to_tensor(meta)
    if kind == "BlockMask":
        return _metadata_to_block_mask(meta)
    if kind == "dict":
        return {k: _deserialize_arg(v) for k, v in meta["items"].items()}
    if kind == "list":
        return [_deserialize_arg(v) for v in meta["items"]]
    if kind == "tuple":
        return tuple(_deserialize_arg(v) for v in meta["items"])
    if kind == "primitive":
        return meta["value"]
    return None


def _deserialize_args(metadata: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Deserialize metadata to synthetic tensors/masks.

    None entries are preserved so that the forward signature is respected —
    e.g. ``attention_masks=None`` must be passed through, not dropped, or
    required kwargs disappear and positional arg indices shift.
    """
    args = tuple(_deserialize_arg(arg_meta) for arg_meta in metadata["args"])
    kwargs = {k: _deserialize_arg(v_meta) for k, v_meta in metadata["kwargs"].items()}
    return args, kwargs

class StageInputRecorder:
    """
    Records inputs AND outputs of model_parts during the first training iteration.

    Hooks into each model_part's forward method to capture:
    - Input shapes/dtypes/devices  -> saved as "stages"
    - Output shapes/dtypes/devices -> saved as "stage_outputs"
    """

    def __init__(self, model_parts: list[torch.nn.Module]):
        self.model_parts = model_parts
        self.recorded_inputs: list[dict[str, Any] | None] = [None] * len(model_parts)
        self.recorded_outputs: list[dict[str, Any] | None] = [None] * len(model_parts)
        self.original_forwards: list[Any] = []
        self.hooks_installed = False

    def install_hooks(self) -> None:
        """Install recording hooks on all model_parts."""
        if self.hooks_installed:
            logger.warning("Recording hooks already installed, skipping")
            return

        for idx, model_part in enumerate(self.model_parts):
            original_forward = model_part.forward
            self.original_forwards.append(original_forward)

            def make_recording_wrapper(stage_idx, orig_fwd, recorder):
                def recording_forward(self_model, *args, **kwargs):
                    output = orig_fwd(*args, **kwargs)
                    # Record inputs and outputs only once per stage
                    if recorder.recorded_inputs[stage_idx] is None:
                        recorder.recorded_inputs[stage_idx] = _serialize_args(
                            args, kwargs
                        )
                        recorder.recorded_outputs[stage_idx] = _serialize_output(output)
                        logger.debug(
                            f"Recorded stage {stage_idx}: "
                            f"{len(args)} input args, output recorded"
                        )
                    return output

                return recording_forward

            model_part.forward = types.MethodType(
                make_recording_wrapper(idx, original_forward, self), model_part
            )

        self.hooks_installed = True
        logger.info(f"Installed recording hooks on {len(self.model_parts)} model_parts")

    def remove_hooks(self) -> None:
        """Remove recording hooks and restore original forward methods."""
        if not self.hooks_installed:
            return

        for idx, model_part in enumerate(self.model_parts):
            if idx < len(self.original_forwards):
                model_part.forward = self.original_forwards[idx]

        self.hooks_installed = False
        logger.info("Removed recording hooks from model_parts")

    def save(self, save_path: str) -> None:
        """Save recorded inputs and outputs to JSON file."""
        unrecorded = [i for i, rec in enumerate(self.recorded_inputs) if rec is None]
        if unrecorded:
            logger.warning(
                f"Not all stages have recorded inputs. Missing stages: {unrecorded}"
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(
                {
                    "stages": self.recorded_inputs,
                    "stage_outputs": self.recorded_outputs,
                },
                f,
                indent=2,
            )

        logger.info(f"Saved stage inputs/outputs to {save_path}")


def maybe_load_stage_shapes(
    job_config,
    num_local_stages: int,
) -> list[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]] | None:
    """
    Load pre-recorded stage shapes as meta tensors if skip-shape-inference is enabled.

    Returns a list of (input_meta_tensors, output_meta_tensors) — one entry per local
    stage — or None if the feature is disabled, the file is missing, or the recorded
    stage count does not match (in which case runtime _shape_inference runs as normal).

    The returned tuples can be passed directly to
    ``PipelineStage(input_args=..., output_args=...)`` to bypass PyTorch's runtime
    _shape_inference (a full dummy forward pass across all PP ranks on the first step).

    Args:
        job_config: Job configuration (reads leto.enable_skip_shape_inference,
            job.dump_folder, and leto.stage_inputs_folder).
        num_local_stages: Number of pipeline stages local to this rank.
    """
    if not job_config.leto.enable_skip_shape_inference:
        return None

    if job_config.parallelism.pipeline_parallel_degree <= 1:
        return None

    rank = dist.get_rank() if dist.is_initialized() else 0
    record_path = os.path.join(
        job_config.job.dump_folder,
        job_config.leto.stage_inputs_folder,
        f"rank_{rank}.json",
    )

    if not os.path.exists(record_path):
        logger.warning(
            f"Stage shape record not found at {record_path}. "
            "Cannot skip shape inference — falling back to runtime inference."
        )
        return None

    with open(record_path, "r") as f:
        data = json.load(f)

    recorded_inputs = data.get("stages", [])
    recorded_outputs = data.get("stage_outputs", [])

    if len(recorded_inputs) != num_local_stages:
        logger.warning(
            f"Recorded stage count ({len(recorded_inputs)}) != "
            f"expected local stages ({num_local_stages}). "
            "Falling back to runtime shape inference."
        )
        return None

    if len(recorded_outputs) != num_local_stages:
        logger.warning(
            f"Recorded output count ({len(recorded_outputs)}) != "
            f"expected local stages ({num_local_stages}). "
            "Falling back to runtime shape inference."
        )
        return None

    result = []
    for stage_idx in range(num_local_stages):
        in_meta = recorded_inputs[stage_idx]
        out_meta = recorded_outputs[stage_idx]

        if in_meta is None or out_meta is None:
            logger.warning(
                f"Stage {stage_idx} has missing input or output metadata. "
                "Falling back to runtime shape inference."
            )
            return None

        input_tensors = _deserialize_as_meta(in_meta)
        output_tensors = _deserialize_as_meta(out_meta)
        result.append((input_tensors, output_tensors))

    logger.info(
        f"Loaded stage shapes for {num_local_stages} local stage(s) from {record_path}"
    )
    return result


def maybe_record_stage_inputs(
    model_parts: list[torch.nn.Module],
    job_config,
    step: int,
    recorder: "StageInputRecorder | None" = None,
) -> "StageInputRecorder | None":
    """
    Handle stage input/output recording based on config and training step.

    - At step 1: installs hooks.
    - At step 2: saves the JSON file and removes hooks.

    Args:
        model_parts: List of model parts to record.
        job_config: Job configuration.
        step: Current training step.
        recorder: Existing recorder instance (if any).

    Returns:
        Recorder instance if recording is active, None otherwise.
    """
    if not job_config.leto.enable_stage_input_record:
        return None

    # Install hooks before first iteration
    if step == 1 and recorder is None:
        recorder = StageInputRecorder(model_parts)
        recorder.install_hooks()
        logger.info("Stage input/output recording enabled for first iteration")
        return recorder

    # Save and cleanup after first iteration.
    if step == 2 and recorder is not None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        save_folder = os.path.join(
            job_config.job.dump_folder, job_config.leto.stage_inputs_folder
        )
        save_path = os.path.join(save_folder, f"rank_{rank}.json")

        recorder.save(save_path)
        recorder.remove_hooks()

        logger.info(f"Stage input/output recording completed, saved to {save_path}")
        return None

    return recorder


def maybe_warmup_stages(
    model_parts: list[torch.nn.Module],
    job_config,
    loss_fn: Optional[Callable] = None,
    pp_has_last_stage: bool = True,
    parallel_dims: Any = None,
) -> None:
    """
    Warmup stages with recorded inputs if enabled in config.

    Args:
        model_parts: List of model parts to warmup
        job_config: Job configuration
        loss_fn: Optional compiled loss function. Only applied to the last
            pipeline stage's output (where ``pred`` has logits shape). On
            non-last stages the output is activations whose trailing dim
            differs from the logits, and running ``loss_fn`` on them would
            compile a variant of the loss graph with the wrong shape.
        pp_has_last_stage: Whether this rank holds the last PP stage.
    """
    if not job_config.leto.enable_stage_warmup:
        return

    _install_subclass_unwrap_debug()

    rank = dist.get_rank() if dist.is_initialized() else 0
    record_folder = os.path.join(
        job_config.job.dump_folder, job_config.leto.stage_inputs_folder
    )
    record_path = os.path.join(record_folder, f"rank_{rank}.json")

    mode = getattr(job_config.leto, "stage_warmup_mode", "fake")
    if mode == "real":
        warmup_stages_real(
            model_parts,
            record_path,
            loss_fn=loss_fn,
            pp_has_last_stage=pp_has_last_stage,
            local_batch_size=job_config.training.local_batch_size,
            seq_len=job_config.training.seq_len,
            parallel_dims=parallel_dims,
        )
    else:
        warmup_stages(
            model_parts,
            record_path,
            warmup_backward=True,
            loss_fn=loss_fn,
            pp_has_last_stage=pp_has_last_stage,
            local_batch_size=job_config.training.local_batch_size,
            seq_len=job_config.training.seq_len,
            parallel_dims=parallel_dims,
        )

    _reload_compiled_fx_graphs_after_warmup()


_subclass_unwrap_debug_installed = False


def _install_subclass_unwrap_debug() -> None:
    """Wrap ``runtime_unwrap_tensor_subclasses`` so that, if the
    ``SubclassCreationMeta`` assertion fails, we log the idx, current
    value type, and recorded meta type before re-raising. This catches
    the mismatch between warmup-compile-time and real-training-runtime
    subclass structure.
    """
    global _subclass_unwrap_debug_installed
    if _subclass_unwrap_debug_installed:
        return
    _subclass_unwrap_debug_installed = True

    import torch._functorch._aot_autograd.subclass_utils as _sub_mod
    from torch._functorch._aot_autograd.schemas import SubclassCreationMeta

    orig = _sub_mod.runtime_unwrap_tensor_subclasses

    from torch.distributed._functional_collectives import AsyncCollectiveTensor

    def wrapped(wrapped_args, *, append_symints, subclass_metas=None):
        # Pre-unwrap any AsyncCollectiveTensor args whose corresponding
        # subclass_meta entry is PlainTensorMeta. Warmup traces under
        # FakeTensorMode produce plain FakeTensors for functional
        # collective outputs (the fake kernels don't wrap in ACS), so
        # AOT records ``PlainTensorMeta`` for those slots. At real
        # training the same slots hold ``AsyncCollectiveTensor(tensor)``
        # from live async collectives; forcing a ``.wait()`` here unwraps
        # them to the inner tensor so the subclass structure matches
        # what was recorded.
        if subclass_metas is not None and isinstance(wrapped_args, list):
            for idx in range(len(wrapped_args)):
                x = wrapped_args[idx]
                if not isinstance(x, AsyncCollectiveTensor):
                    continue
                if idx >= len(subclass_metas):
                    continue
                meta = subclass_metas[idx]
                if isinstance(meta, SubclassCreationMeta):
                    continue
                wrapped_args[idx] = x.trigger_wait()
        try:
            return orig(
                wrapped_args,
                append_symints=append_symints,
                subclass_metas=subclass_metas,
            )
        except AssertionError:
            from torch.utils._python_dispatch import (
                is_traceable_wrapper_subclass,
            )
            logger.error("subclass_inp_meta mismatch diagnostic:")
            for idx, x in enumerate(wrapped_args):
                is_sub = (
                    is_traceable_wrapper_subclass(x)
                    if isinstance(x, torch.Tensor)
                    else False
                )
                meta = (
                    subclass_metas[idx]
                    if subclass_metas is not None and idx < len(subclass_metas)
                    else None
                )
                logger.error(
                    f"  idx={idx} type={type(x).__name__} "
                    f"is_subclass={is_sub} "
                    f"meta_type={type(meta).__name__ if meta is not None else 'None'}"
                )
            raise

    _sub_mod.runtime_unwrap_tensor_subclasses = wrapped
    # Rebind the local copy imported at module-load time in runtime_wrappers.
    import torch._functorch._aot_autograd.runtime_wrappers as _rw_mod
    _rw_mod.runtime_unwrap_tensor_subclasses = wrapped

    # Symmetric fix for backward: when warmup-time AOT trace saw an
    # AsyncCollectiveTensor input (because we forced ACT under fake mode
    # in `_patch_for_fake_warmup`), AOT records the corresponding
    # backward tangent slot as expecting ACT. At real training the
    # tangent arrives as a plain Tensor (no live functional collective
    # is producing the gradient), so `process_runtime_tangent`'s
    # `maybe_coerce` finds no `__coerce_same_metadata_as_tangent__` on
    # plain Tensor and raises. Pre-wrap such plain Tensors in ACT so
    # the maybe_coerce path lands on ACT's coerce method, which calls
    # `trigger_wait()` and returns a plain Tensor — matching what the
    # graph expects after unflatten.
    from torch._functorch._aot_autograd.schemas import (
        SubclassCreationMeta as _SCM,
    )

    _orig_prt = _rw_mod.AOTDispatchAutograd.process_runtime_tangent

    def _prt_simple(x, meta):
        if (
            isinstance(meta, _SCM)
            and meta.original_subclass_type is AsyncCollectiveTensor
            and isinstance(x, torch.Tensor)
            and not isinstance(x, AsyncCollectiveTensor)
        ):
            wrapped_x = AsyncCollectiveTensor(x)
            wrapped_x.completed = True
            x = wrapped_x
        return _orig_prt(x, meta)

    _rw_mod.AOTDispatchAutograd.process_runtime_tangent = staticmethod(
        _prt_simple
    )

import contextlib


def _iter_all_submodules(model_parts):
    for mp in model_parts:
        for submod in mp.modules():
            yield submod


def _find_ep_hooks(model_parts, method_name):
    """
    Walk every submodule of every model_part and find pre/forward hooks whose
    closure contains a bound method whose underlying function's ``__qualname__``
    is ``ExpertParallel.<method_name>``.

    Returns a list of tuples ``(hook_dict, key, orig_hook, ep_instance,
    device_mesh, fqn)``. ``hook_dict`` is the live ``OrderedDict`` from the
    submodule (``_forward_pre_hooks`` or ``_forward_hooks``), so the caller
    can replace entries in place and restore them later. ``fqn`` is the
    submodule's fully-qualified name within its model_part — stable across
    record-time and warmup-time, so caller can key recorded EP state on it.
    """
    from torch.distributed.device_mesh import DeviceMesh

    target_qualname = f"ExpertParallel.{method_name}"
    hook_dict_names = ("_forward_pre_hooks", "_forward_hooks")

    results = []
    for mp_idx, mp in enumerate(model_parts):
        for fqn, submod in mp.named_modules():
            for hook_dict_name in hook_dict_names:
                hook_dict = getattr(submod, hook_dict_name, None)
                if not hook_dict:
                    continue
                for key, hook in list(hook_dict.items()):
                    closure = getattr(hook, "__closure__", None)
                    if not closure:
                        continue
                    ep_inst = None
                    dm = None
                    for cell in closure:
                        try:
                            val = cell.cell_contents
                        except ValueError:
                            continue
                        # Bound method of ExpertParallel.<method_name>?
                        fn = getattr(val, "__func__", None)
                        if fn is not None and getattr(
                            fn, "__qualname__", ""
                        ) == target_qualname:
                            ep_inst = val.__self__
                        elif isinstance(val, DeviceMesh):
                            dm = val
                    if ep_inst is not None and dm is not None:
                        full_fqn = f"mp{mp_idx}.{fqn}" if fqn else f"mp{mp_idx}"
                        results.append((hook_dict, key, hook, ep_inst, dm, full_fqn))
    return results


@contextlib.contextmanager
def _patch_for_fake_warmup(model_parts):
    """
    Monkey-patch layers that can't run safely under FakeTensorMode, replacing
    them with shape-correct stubs so the whole forward is runnable.

    Why each patch exists:

    - ``ExpertParallel._token_dispatch`` / ``_token_combine``: the real impl
      does ``.tolist()`` (data-dependent) and ``all_to_all_single_autograd``
      (collective). Both poison CUDA under fake mode.
    - ``moe.utils._permute`` / ``_unpermute``: call a custom CUDA kernel
      (``generate_permute_indices``) with no fake impl.
    - ``FlexAttentionWrapper._compiled_flex_attn``: a module-level
      ``torch.compile(flex_attention)`` whose Inductor-compiled Triton kernel
      is launched below the ``__torch_dispatch__`` layer and reads
      ``data_ptr()`` on FakeTensors → illegal memory access.

    MoE patches assume *perfectly balanced* token routing; the resulting
    shapes match what the real run sees on average and stays bit-identical
    for non-MoE workloads and gpt-oss. The remaining MoE-specific drift
    is addressed by `stage_warmup_mode = "real"` which traces the actual
    forward instead.
    """
    from torchtitan.distributed import expert_parallel as ep_mod
    from torchtitan.models.moe import utils as moe_utils
    from torchtitan.models.moe import moe as shared_moe
    from torchtitan.models import attention as attention_mod
    from torchtitan.models.gpt_oss.model import moe as gpt_oss_moe
    from torchtitan.tools.utils import _round_up

    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from torch._dynamo.guards import GuardBuilder, CheckFunctionManager
    import torch._C._dynamo.guards as _c_guards_mod
    from torch._subclasses.fake_tensor import FakeTensor
    import torch.distributed._functional_collectives as _fc_mod

    orig_token_dispatch = ep_mod.ExpertParallel._token_dispatch
    orig_token_combine = ep_mod.ExpertParallel._token_combine
    orig_permute = moe_utils._permute
    orig_unpermute = moe_utils._unpermute
    orig_flex_attn = attention_mod.FlexAttentionWrapper._compiled_flex_attn
    orig_experts_for_loop = gpt_oss_moe._run_experts_for_loop
    orig_experts_grouped_mm = gpt_oss_moe._run_experts_grouped_mm
    orig_shared_experts_for_loop = shared_moe._run_experts_for_loop
    orig_shared_experts_grouped_mm = shared_moe._run_experts_grouped_mm
    orig_autotuner_run = CachingAutotuner.run
    orig_tensor_match = GuardBuilder.TENSOR_MATCH
    orig_cfm_init = CheckFunctionManager.__init__
    orig_empty_strided_cuda = _c_guards_mod._empty_strided_cuda
    orig_reinterpret_tensor = _c_guards_mod._reinterpret_tensor
    orig_fake_new = FakeTensor.__new__
    orig_maybe_wrap_tensor = _fc_mod._maybe_wrap_tensor

    # Disable on-disk inductor / AOT-autograd / autotune cache writes during
    # warmup. fake-mode-compiled artifacts can land in the shared cache under
    # keys that real-mode compiles later look up, causing real training to
    # reuse fake-mode artifacts and drift bit-for-bit. Reads stay enabled so
    # warmup still benefits from any real-mode entries already populated by
    # earlier phases (warmup_run / init / normal in run_e2e.sh).
    from torch._inductor.codecache import FxGraphCache as _FxGraphCache
    from torch._functorch._aot_autograd.autograd_cache import (
        AOTAutogradCache as _AOTAutogradCache,
    )
    from torch._inductor.runtime.autotune_cache import (
        AutotuneCacheBundler as _AutotuneCacheBundler,
    )
    orig_fx_save = _FxGraphCache._save_graph
    orig_fx_write_local = _FxGraphCache._write_to_local_cache
    orig_aot_save = _AOTAutogradCache.save
    orig_aot_write_local = _AOTAutogradCache._write_to_local_cache
    orig_autotune_put = _AutotuneCacheBundler.put

    def fake_permute(x, num_tokens_per_expert, ep_degree, num_local_experts):
        align = moe_utils.TOKEN_GROUP_ALIGN_SIZE_M
        x_padded_per_expert = x.shape[0] + num_local_experts * align
        padded_max_len = _round_up(x_padded_per_expert, align)

        # Match the real _permute: it vstacks a padding row then gathers.
        x_vstacked = torch.vstack((x, x.new_zeros((x.shape[-1],))))
        input_shape = x_vstacked.shape

        permuted_indices = torch.zeros(
            padded_max_len, dtype=torch.long, device=x.device
        )
        x_out = x.new_zeros((padded_max_len, x.shape[-1]))

        # Balanced counts per local expert.
        per_expert = padded_max_len // max(num_local_experts, 1)
        balanced_counts = num_tokens_per_expert.new_full(
            (num_local_experts,), per_expert
        )
        return input_shape, x_out, permuted_indices, balanced_counts

    def fake_unpermute(out, input_shape, permuted_indices):
        # Real _unpermute returns out_unpermuted[:-1] (drops the padding row).
        out_unpermuted = out.new_zeros(input_shape)
        return out_unpermuted[:-1]

    def fake_token_dispatch(self, mod, inputs, device_mesh):
        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        input_shape, routed_out, permuted_indices, counts = fake_permute(
            routed_input, num_tokens_per_expert, ep_degree, num_local_experts
        )
        self.input_shape = input_shape
        self.permuted_indices = permuted_indices

        per_rank = routed_input.shape[0] // max(ep_degree, 1)
        self.input_splits = [per_rank] * ep_degree
        self.output_splits = [per_rank] * ep_degree
        return routed_out, counts

    def fake_token_combine(self, mod, routed_output, device_mesh):
        return fake_unpermute(routed_output, self.input_shape, self.permuted_indices)

    def fake_run_experts(
        mlp1_weight,
        mlp1_bias,
        mlp2_weight,
        mlp2_bias,
        swiglu_limit,
        x,
        num_tokens_per_expert,
        tp_degree=1,
    ):
        # Real _run_experts_for_loop and _run_experts_grouped_mm both produce
        # an output with the same shape as x (the loop variant pads with
        # zeros back to x.shape[0]; the grouped-mm variant preserves shape).
        return x.new_zeros(x.shape)

    def fake_shared_run_experts(w1, w2, w3, x, num_tokens_per_expert):
        # Shared moe._run_experts_for_loop calls .tolist() on
        # num_tokens_per_expert which raises under FakeTensorMode. Both
        # variants (for_loop and grouped_mm) preserve the input shape of x,
        # so return a zero tensor matching x.
        return x.new_zeros(x.shape)

    def fake_flex_attn(q, k, v, *, block_mask=None, scale=None, return_lse=False, **kw):
        # Output shape of flex_attention matches q: (B, H, S, D).
        out = q.new_zeros(q.shape, dtype=q.dtype)
        if return_lse:
            # LSE shape is (B, H, S), always float32 in flex_attention.
            lse = q.new_zeros(q.shape[:-1], dtype=torch.float32)
            return out, lse
        return out

    def patched_tensor_match(self, guard, value=None):
        # Skip TENSOR_MATCH guard emission during warmup. Correctness is
        # preserved by:
        #   - reconstructing DTensor inputs with proper mesh/placements
        #     in ``_metadata_to_dtensor`` (AOTAutograd sees the right
        #     subclass at the input slot),
        #   - overriding ``FakeTensor.pytype = torch.Tensor`` globally
        #     during warmup (below), so any guard that bottoms out at
        #     a leaf tensor (e.g. ``param._local_tensor``) records the
        #     Tensor pytype rather than FakeTensor. Real training's
        #     real tensors then match the cached guards.
        return None

    def patched_fake_new(cls, fake_mode, elem, device, **kwargs):
        # Dynamo's TENSOR_MATCH guard records ``pytype=value.pytype`` for
        # FakeTensors and falls back to ``type(value)`` when pytype is
        # unset (guards.py:2946). The default ``pytype=None`` means the
        # guard stores ``FakeTensor`` as the expected type, which fails
        # at real-training lookup time. Default pytype to ``torch.Tensor``
        # for FakeTensors created during warmup so guards key on Tensor.
        if kwargs.get("pytype") is None:
            kwargs["pytype"] = torch.Tensor
        return orig_fake_new(cls, fake_mode, elem, device, **kwargs)

    def patched_cfm_init(self, f_code, output_graph, *args, **kwargs):
        # Skip Dynamo's post-build guard self-check during warmup. Our
        # patched_tensor_match makes guards expect ``torch.Tensor``, but the
        # self-check (guards.py:3908) runs the new guards against the actual
        # tracing-time FakeTensor values, which fail the Tensor type check
        # and raise "Guard failed on the same frame it was created". At real
        # training time, inputs are real torch.Tensor so the guard passes.
        if output_graph is not None:
            try:
                output_graph.skip_guards_check = True
            except Exception:
                pass
        return orig_cfm_init(self, f_code, output_graph, *args, **kwargs)

    def fake_empty_strided_cuda(size, stride, dtype):
        # ``torch._C._dynamo.guards._empty_strided_cuda`` is a fast-path
        # allocator used inside Inductor-generated code that bypasses
        # __torch_dispatch__, so it produces a real CUDA tensor even when
        # FakeTensorMode is active. That breaks warmup because subsequent
        # ops mix real-CUDA temporaries with FakeTensor parameters. Route
        # through ``torch.empty_strided`` which IS dispatched and therefore
        # intercepted by the ambient FakeTensorMode.
        return torch.empty_strided(size, stride, dtype=dtype, device="cuda")

    def fake_reinterpret_tensor(tensor, size, stride, offset=0):
        # ``torch._C._dynamo.guards._reinterpret_tensor`` bypasses Python
        # dispatch and constructs a view by reaching into the underlying
        # ``elem`` of a FakeTensor (which is a meta tensor), losing the
        # FakeTensor wrapper. ``torch.as_strided`` goes through dispatch and
        # preserves the FakeTensor subclass.
        return torch.as_strided(tensor, size, stride, offset)

    def fake_autotuner_run(self, *args, stream, **kwargs):
        # Inductor Triton kernel launcher. We let precompile() build all
        # candidates (it doesn't launch kernels, only compiles), but skip
        # both autotune (which would benchmark with FakeTensor pointers and
        # fault) and the actual launcher call. We DO NOT truncate
        # self.launchers, so the first real-mode call will trigger the
        # standard autotune path and pick the genuine winner — same kernel
        # a no-warmup baseline would pick. Truncating here would lock us
        # into the first candidate (an arbitrary choice) and produce a
        # different bf16 reduction order than the baseline.
        if len(self.launchers) == 0:
            self.precompile()
        return None

    ep_mod.ExpertParallel._token_dispatch = fake_token_dispatch
    ep_mod.ExpertParallel._token_combine = fake_token_combine
    moe_utils._permute = fake_permute
    moe_utils._unpermute = fake_unpermute
    # Leave FlexAttentionWrapper._compiled_flex_attn untouched: we want Dynamo
    # to trace it so the frame cache entry gets built during warmup. The
    # CachingAutotuner.run no-op below ensures the Triton kernel does not
    # actually launch on fake inputs.
    gpt_oss_moe._run_experts_for_loop = fake_run_experts
    gpt_oss_moe._run_experts_grouped_mm = fake_run_experts
    shared_moe._run_experts_for_loop = fake_shared_run_experts
    shared_moe._run_experts_grouped_mm = fake_shared_run_experts
    CachingAutotuner.run = fake_autotuner_run
    GuardBuilder.TENSOR_MATCH = patched_tensor_match
    CheckFunctionManager.__init__ = patched_cfm_init
    _c_guards_mod._empty_strided_cuda = fake_empty_strided_cuda
    _c_guards_mod._reinterpret_tensor = fake_reinterpret_tensor
    FakeTensor.__new__ = patched_fake_new

    # Class-level replacement above doesn't affect already-registered DTensor
    # pre/forward hooks on MoE submodules, because those hooks closed over
    # *bound methods* of the original ``_token_dispatch``/``_token_combine``
    # at model-parallelization time — before our patch ran. We have to swap
    # the hook entries directly.
    dispatch_hook_records = _find_ep_hooks(model_parts, "_token_dispatch")
    combine_hook_records = _find_ep_hooks(model_parts, "_token_combine")

    def _make_fake_pre_hook(ep_inst, device_mesh):
        def _hook(mod, inputs):
            return fake_token_dispatch(ep_inst, mod, inputs, device_mesh)
        return _hook

    def _make_fake_post_hook(ep_inst, device_mesh):
        def _hook(mod, inputs, output):
            return fake_token_combine(ep_inst, mod, output, device_mesh)
        return _hook

    for hook_dict, key, _orig, ep_inst, dm, fqn in dispatch_hook_records:
        hook_dict[key] = _make_fake_pre_hook(ep_inst, dm)
    for hook_dict, key, _orig, ep_inst, dm, fqn in combine_hook_records:
        hook_dict[key] = _make_fake_post_hook(ep_inst, dm)

    logger.info(
        f"Fake warmup: patched {len(dispatch_hook_records)} _token_dispatch "
        f"and {len(combine_hook_records)} _token_combine hooks"
    )

    # Patch _maybe_wrap_tensor so eager-mode collective wrappers (called
    # from within model.forward but outside any active Dynamo/AOT proxy
    # capture) wrap their result in AsyncCollectiveTensor instead of
    # eagerly emitting wait_tensor.
    #
    # Why: real training, when Dynamo enters compile region with an ACT
    # input, AOT unflattens ACT and emits a wait_tensor op as the first
    # node of the graph. Under our outer FakeTensorMode, _are_we_tracing
    # returns True (FakeTensorMode is active) so _maybe_wrap_tensor
    # short-circuits to wait_tensor(self), producing a plain tensor and
    # the next compile region's primals are plain — no wait_tensor op
    # in the graph. AOT's min-cut partitioner sees a different graph
    # topology (one fewer non-fusible boundary), makes different
    # save-vs-recompute decisions, and Inductor produces a different
    # set of fused kernels. By forcing ACT under fake mode (when no
    # proxy mode is active = outside compile region), we make the
    # warmup-time AOT trace observe the same input subclass structure
    # as real training.
    from torch.fx.experimental.proxy_tensor import get_proxy_mode as _gpm
    def _patched_maybe_wrap_tensor(self):
        # Mirror real-mode `_are_we_tracing` *minus* the FakeTensorMode
        # check. Real mode never has an outer FakeTensorMode active;
        # warmup does (it's our top-level context). The original
        # `_are_we_tracing` returns True under our outer FakeTensorMode,
        # which short-circuits to `wait_tensor(self)` and produces a
        # plain tensor — but in real mode that same call site (inside
        # AOT metadata collection or Dynamo/AOT proxy) wraps in ACT
        # because real mode doesn't have an outer FakeTensorMode.
        # The result is that warmup-time AOT records different subclass
        # metadata than real-mode would, and the resulting AOT joint
        # graph diverges (extra/missing wait_tensor nodes).
        #
        # Fix: only emit wait_tensor if Dynamo / proxy / PythonDispatcher
        # is active (which mirrors what real mode triggers). Wrap in
        # ACT otherwise — including when our outer FakeTensorMode is
        # the only dispatch mode active.
        if _fc_mod.is_torchdynamo_compiling():
            return _fc_mod.wait_tensor(self)
        if _gpm() is not None:
            return _fc_mod.wait_tensor(self)
        if torch._C._dispatch_tls_is_dispatch_key_included(
            torch._C.DispatchKey.PythonDispatcher
        ):
            return _fc_mod.wait_tensor(self)
        from torch.distributed._functional_collectives import AsyncCollectiveTensor
        return AsyncCollectiveTensor(self)
    _fc_mod._maybe_wrap_tensor = _patched_maybe_wrap_tensor
    # Functional-collective wrappers also import the symbol by-value at
    # module import; rebind their callers' references too.
    for _mod_name in (
        "torch.distributed.tensor._dispatch",
        "torch.distributed.tensor._redistribute",
        "torch.distributed.tensor._collective_utils",
    ):
        try:
            import importlib as _il
            _m = _il.import_module(_mod_name)
            if hasattr(_m, "_maybe_wrap_tensor"):
                setattr(_m, "_maybe_wrap_tensor", _patched_maybe_wrap_tensor)
        except Exception:
            pass

    # Replace cache writes with no-ops. Reads still work since they go through
    # different code paths (FxGraphCache.load_with_key, AOTAutogradCache.load).
    _FxGraphCache._save_graph = staticmethod(lambda *a, **kw: None)
    _FxGraphCache._write_to_local_cache = staticmethod(lambda *a, **kw: None)
    _AOTAutogradCache.save = staticmethod(lambda *a, **kw: None)
    _AOTAutogradCache._write_to_local_cache = staticmethod(lambda *a, **kw: None)
    _AutotuneCacheBundler.put = classmethod(lambda cls, *a, **kw: None)

    try:
        yield
    finally:
        _FxGraphCache._save_graph = orig_fx_save
        _FxGraphCache._write_to_local_cache = orig_fx_write_local
        _AOTAutogradCache.save = orig_aot_save
        _AOTAutogradCache._write_to_local_cache = orig_aot_write_local
        _AutotuneCacheBundler.put = orig_autotune_put
        ep_mod.ExpertParallel._token_dispatch = orig_token_dispatch
        ep_mod.ExpertParallel._token_combine = orig_token_combine
        moe_utils._permute = orig_permute
        moe_utils._unpermute = orig_unpermute
        attention_mod.FlexAttentionWrapper._compiled_flex_attn = orig_flex_attn
        gpt_oss_moe._run_experts_for_loop = orig_experts_for_loop
        gpt_oss_moe._run_experts_grouped_mm = orig_experts_grouped_mm
        shared_moe._run_experts_for_loop = orig_shared_experts_for_loop
        shared_moe._run_experts_grouped_mm = orig_shared_experts_grouped_mm
        CachingAutotuner.run = orig_autotuner_run
        GuardBuilder.TENSOR_MATCH = orig_tensor_match
        CheckFunctionManager.__init__ = orig_cfm_init
        _c_guards_mod._empty_strided_cuda = orig_empty_strided_cuda
        _c_guards_mod._reinterpret_tensor = orig_reinterpret_tensor
        FakeTensor.__new__ = orig_fake_new
        _fc_mod._maybe_wrap_tensor = orig_maybe_wrap_tensor
        for _mod_name in (
            "torch.distributed.tensor._dispatch",
            "torch.distributed.tensor._redistribute",
            "torch.distributed.tensor._collective_utils",
        ):
            try:
                import importlib as _il
                _m = _il.import_module(_mod_name)
                if hasattr(_m, "_maybe_wrap_tensor"):
                    setattr(_m, "_maybe_wrap_tensor", orig_maybe_wrap_tensor)
            except Exception:
                pass
        for hook_dict, key, orig, _ep, _dm, _fqn in dispatch_hook_records:
            hook_dict[key] = orig
        for hook_dict, key, orig, _ep, _dm, _fqn in combine_hook_records:
            hook_dict[key] = orig


def _reduce_output_to_scalar(output: Any) -> "torch.Tensor | None":
    """Sum-reduce a forward output (Tensor or tuple/list of Tensors) to a scalar
    loss for driving backward. Returns None if no grad-requiring tensor is found."""
    if isinstance(output, torch.Tensor):
        return output.sum() if output.requires_grad else None
    if isinstance(output, (tuple, list)):
        grad_tensors = [
            o for o in output if isinstance(o, torch.Tensor) and o.requires_grad
        ]
        if not grad_tensors:
            return None
        return sum(t.sum() for t in grad_tensors)
    return None


def warmup_stages(
    model_parts: list[torch.nn.Module],
    record_path: str,
    warmup_backward: bool = False,
    loss_fn: Optional[Callable] = None,
    pp_has_last_stage: bool = True,
    local_batch_size: int = 1,
    seq_len: int = 2048,
    parallel_dims: Any = None,
) -> None:
    """
    Warmup stages by running forward (and optionally backward) with synthetic
    inputs under ``FakeTensorMode``.

    Why fake tensors: this runs in a standby process that shares the device with
    (or stands in for) the real training process. We want Dynamo tracing,
    AOTAutograd partitioning, and Inductor codegen to happen — populating the
    in-process Dynamo guarded cache and the on-disk FX/Inductor caches — without
    launching real CUDA kernels or allocating activation memory. When this same
    process later runs real training, the first real forward/backward hits the
    warm Dynamo cache and skips retracing/recompile.

    ``warmup_backward`` defaults to False because backward through
    expert-parallel MoE drives collective ops that have no fake impl and will
    poison the CUDA context via real NCCL calls with garbage shapes. Forward
    can already fail on the same path; this just reduces the surface area.

    Args:
        model_parts: List of model parts (stage submodules) to warmup
        record_path: Path to the JSON file with recorded input metadata
        warmup_backward: If True, also run .backward() on each stage's output
            to compile the backward partition. Leave False for MoE+EP models.
    """
    global _current_parallel_dims
    _current_parallel_dims = parallel_dims

    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch._inductor import config as inductor_config

    if not os.path.exists(record_path):
        logger.warning(
            f"Stage input record file not found at {record_path}, skipping warmup"
        )
        _current_parallel_dims = None
        return

    with open(record_path, "r") as f:
        data = json.load(f)

    recorded_stages = data["stages"]

    if len(recorded_stages) != len(model_parts):
        logger.warning(
            f"Mismatch between recorded stages ({len(recorded_stages)}) "
            f"and model_parts ({len(model_parts)}). This may happen if pipeline "
            f"configuration changed. Skipping warmup."
        )
        _current_parallel_dims = None
        return

    logger.info(
        f"Starting stage warmup for {len(model_parts)} model_parts "
        "under FakeTensorMode..."
    )

    # KEEP Inductor's on-disk FX graph cache ENABLED during warmup. With
    # the `_maybe_wrap_tensor` patch in `_patch_for_fake_warmup`, the
    # warmup-time AOT joint graph matches the real-mode joint graph
    # node-for-node, so the cache key is stable across modes. Real
    # training's compile then hits the warmup-populated cache and reuses
    # the exact same Inductor-compiled artifacts (same tiling_scores,
    # same autotune candidate list, same chosen launcher) — bit-identity.
    # Disabling the cache here would force real training to recompile,
    # and the recompile would observe slightly different size hints
    # than warmup (because Inductor's first-seen concrete shape value
    # is what it benchmarks against, and that depends on which call
    # site triggered the compile). Keeping the cache enabled defers the
    # benchmark to warmup time so both runs land on the same artifact.
    prev_fx_graph_cache = inductor_config.fx_graph_cache
    inductor_config.fx_graph_cache = True

    # Snapshot real grads before warmup. RMP restores param.grad from the
    # active's last step on standby ranks (init.py:restore_param_gradients),
    # and resilient_opt's recovery path reads param.grad. The per-stage
    # zero_grad(set_to_none=True) below would wipe those, so we save here and
    # restore after warmup. Detaching to None during warmup also prevents the
    # fake backward from accumulating fake tensors into real grad slots.
    saved_grads: list[dict[int, torch.Tensor]] = []
    for mp in model_parts:
        grads_for_mp: dict[int, torch.Tensor] = {}
        for p in mp.parameters():
            if p.grad is not None:
                grads_for_mp[id(p)] = p.grad
                p.grad = None
        saved_grads.append(grads_for_mp)

    # allow_non_fake_inputs=True lets the real cuda tensors produced by
    # _deserialize_args (and the real parameters inside model_part) be
    # auto-lifted to FakeTensors on dispatch, so we don't have to manually
    # convert every input or parameter.
    with FakeTensorMode(allow_non_fake_inputs=True), _patch_for_fake_warmup(
        model_parts
    ):
        # Pre-compute attention_masks ONCE using the first model_part's
        # ``get_attention_masks`` with a full-batch stub, then reuse for
        # ALL virtual stages. This is critical for PP Interleaved1F1B
        # where each rank has multiple virtual stages: only the first
        # receives 2-D tokens; later ones receive 3-D activations and
        # would fall back to ``_metadata_to_block_mask`` (which uses
        # bare ``get_causal_mask_mod()`` rather than
        # ``and_masks(causal, document)``). The resulting ``mask_mod``
        # code-object mismatch triggers Dynamo recompiles on every
        # layer. Building once ensures all stages trace against the
        # same mask_mod closure, matching real training's single
        # ``get_attention_masks`` call in ``forward_backward_step``.
        shared_attention_masks = None
        _all_split_masks = None  # list of per-microbatch masks for PP
        model_args = getattr(model_parts[0], "model_args", None)
        attn_type = getattr(model_args, "attn_type", "sdpa")
        if attn_type in ("flex", "varlen"):
            class _StubTokenizer:
                eos_id = 0
            device = f"cuda:{torch.cuda.current_device()}"
            stub_tokens = torch.zeros(
                local_batch_size,
                seq_len,
                dtype=torch.long,
                device=device,
            )
            try:
                # pyrefly: ignore [not-callable]
                shared_attention_masks = model_parts[0].get_attention_masks(
                    input_batch=stub_tokens,
                    tokenizer=_StubTokenizer(),
                    extra_inputs={},
                )
                # For PP, the pipeline schedule splits the BlockMask per
                # microbatch via ``_split_block_mask``, wrapping each
                # chunk's ``mask_mod`` in a ``batch_offset_mask_mod``
                # closure (microbatch.py). Warmup must match, otherwise
                # the Dynamo guard on ``mask_mod.__code__`` sees
                # ``and_mask`` (from warmup) vs ``batch_offset_mask_mod``
                # (from real training) → recompile on every layer. Split
                # here and use a single chunk for all stages — the code
                # object of ``batch_offset_mask_mod`` is the same across
                # all chunks.
                if (
                    len(model_parts) > 1
                    and parallel_dims is not None
                    and getattr(parallel_dims, "pp_enabled", False)
                    and isinstance(shared_attention_masks, BlockMask)
                ):
                    from torch.distributed.pipelining.microbatch import (
                        _split_block_mask,
                    )
                    num_chunks = max(
                        local_batch_size,
                        shared_attention_masks.kv_num_blocks.size(0),
                    )
                    if num_chunks > 1:
                        split = _split_block_mask(
                            shared_attention_masks, num_chunks
                        )
                        _all_split_masks = split
                        shared_attention_masks = split[0]
                elif isinstance(shared_attention_masks, dict):
                    # gpt_oss returns dict[str, BlockMask].
                    # Split each value if PP.
                    if (
                        len(model_parts) > 1
                        and parallel_dims is not None
                        and getattr(parallel_dims, "pp_enabled", False)
                    ):
                        from torch.distributed.pipelining.microbatch import (
                            _split_block_mask,
                        )
                        # Collect per-key splits for building
                        # _all_split_masks as list of dicts.
                        key_splits: dict[str, list] = {}
                        max_chunks = 1
                        new_masks = {}
                        for k, v in shared_attention_masks.items():
                            if isinstance(v, BlockMask):
                                nc = max(
                                    local_batch_size,
                                    v.kv_num_blocks.size(0),
                                )
                                if nc > 1:
                                    s = _split_block_mask(v, nc)
                                    new_masks[k] = s[0]
                                    key_splits[k] = s
                                    max_chunks = max(max_chunks, len(s))
                                else:
                                    new_masks[k] = v
                            else:
                                new_masks[k] = v
                        shared_attention_masks = new_masks
                        if max_chunks > 1:
                            _all_split_masks = []
                            for ci in range(max_chunks):
                                chunk_dict = {}
                                for k, v in new_masks.items():
                                    if k in key_splits:
                                        chunk_dict[k] = key_splits[k][
                                            min(ci, len(key_splits[k]) - 1)
                                        ]
                                    else:
                                        chunk_dict[k] = v
                                _all_split_masks.append(chunk_dict)
            except Exception as e:
                logger.warning(
                    f"Fake warmup: get_attention_masks failed "
                    f"({type(e).__name__}: {e}), using per-stage recorded masks."
                )

        # Build the list of microbatch masks to iterate. For PP with
        # block_causal, ``_split_block_mask`` creates per-microbatch
        # closures that capture a ``batch_offset`` constant. Dynamo
        # specializes on this constant. We must compile with EACH
        # distinct batch_offset so real training hits a cached entry
        # for every microbatch. For non-PP or causal (B=1), the split
        # short-circuits and there's just one mask.
        microbatch_masks = (
            _all_split_masks
            if _all_split_masks is not None and len(_all_split_masks) > 1
            else [shared_attention_masks]
        )

        for mb_idx, _current_microbatch_mask in enumerate(microbatch_masks):
            if mb_idx > 0:
                logger.info(
                    f"Fake warmup: extra microbatch pass {mb_idx} "
                    f"(compiling for batch_offset={mb_idx})"
                )
            for stage_idx, (model_part, stage_metadata) in enumerate(
                zip(model_parts, recorded_stages)
            ):
                if stage_metadata is None:
                    logger.warning(
                        f"No recorded inputs for stage {stage_idx}, skipping"
                    )
                    continue

                synthetic_args, synthetic_kwargs = _deserialize_args(stage_metadata)
                try:
                    # Override attention_masks for ALL virtual stages with
                    # the current microbatch's mask.
                    if (
                        _current_microbatch_mask is not None
                        and "attention_masks" in synthetic_kwargs
                    ):
                        synthetic_kwargs["attention_masks"] = (
                            _current_microbatch_mask
                        )
                    output = model_part(*synthetic_args, **synthetic_kwargs)
                    if warmup_backward:
                        loss = None
                        is_last_stage_module = (
                            pp_has_last_stage
                            and stage_idx == len(model_parts) - 1
                        )
                        if (
                            is_last_stage_module
                            and loss_fn is not None
                            and isinstance(output, torch.Tensor)
                        ):
                            labels = torch.zeros(
                                output.shape[:-1],
                                dtype=torch.long,
                                device=output.device,
                            )
                            from torch.distributed.tensor import DTensor

                            if isinstance(output, DTensor):
                                from torch.distributed.tensor.parallel import (
                                    loss_parallel,
                                )

                                with loss_parallel():
                                    loss = loss_fn(output, labels)
                                    if loss is not None:
                                        loss.backward()
                            else:
                                loss = loss_fn(output, labels)
                                if loss is not None:
                                    loss.backward()
                        else:
                            loss = _reduce_output_to_scalar(output)
                            if loss is not None:
                                loss.backward()

                    del synthetic_args, synthetic_kwargs, output
                except Exception as e:
                    logger.warning(
                        f"Fake warmup failed for stage {stage_idx} "
                        f"(mb={mb_idx}): "
                        f"{type(e).__name__}: {e}. "
                        "This stage will pay full compile cost on the "
                        "first real step."
                    )

                    is_cuda_err = isinstance(
                        e, torch.AcceleratorError
                    ) or "CUDA error" in str(e)
                    if is_cuda_err:
                        raise RuntimeError(
                            f"Fake warmup poisoned CUDA context at stage "
                            f"{stage_idx}. This usually means an op inside "
                            "this stage (e.g. expert-parallel all-to-all) "
                            "fell through FakeTensorMode and launched a real "
                            "kernel with invalid arguments. Disable "
                            "enable_stage_warmup for this model or skip "
                            "this stage explicitly."
                        ) from e
                    raise e

                model_part.zero_grad(set_to_none=True)

    # Restore the real grads we snapshotted before warmup. Params whose grad
    # was None before warmup stay None.
    for mp, grads_for_mp in zip(model_parts, saved_grads):
        for p in mp.parameters():
            p.grad = grads_for_mp.get(id(p))

    inductor_config.fx_graph_cache = prev_fx_graph_cache

    # FSDP2 caches the unsharded full param across forwards in
    # ``FSDPParam._unsharded_param`` and ``all_gather_outputs``. Under warmup,
    # the all-gather ran under FakeTensorMode so the cached unsharded param is
    # a FakeTensor. On subsequent real forwards, FSDP's init_unsharded_param
    # takes the copy_ path (since ``hasattr(self, "_unsharded_param")`` is
    # True) and keeps that fake nn.Parameter bound to ``module.weight``.
    # Purge the cache so the next real all-gather recreates it fresh.
    purged = 0
    for mp in model_parts:
        for mod in mp.modules():
            get_state = getattr(mod, "_get_fsdp_state", None)
            if get_state is None:
                continue
            try:
                state = get_state()
            except Exception:
                continue
            pg = getattr(state, "_fsdp_param_group", None)
            if pg is None:
                continue
            for fp in pg.fsdp_params:
                if hasattr(fp, "_unsharded_param"):
                    try:
                        del fp._unsharded_param
                    except Exception:
                        pass
                fp.all_gather_outputs = []
                if hasattr(fp, "_unsharded_inner_tensors"):
                    fp._unsharded_inner_tensors = []
                purged += 1
    if purged:
        logger.info(f"Post-warmup FSDP purge: cleared unsharded cache on {purged} FSDPParams")

    # DIAGNOSTIC: walk every parameter and buffer on every model_part and
    # report any that have been rebound to a FakeTensor during warmup.
    from torch._subclasses.fake_tensor import FakeTensor

    def _describe_tensor(t) -> str:
        parts = [
            f"type={type(t).__name__}",
            f"shape={tuple(t.shape)}",
            f"dtype={t.dtype}",
        ]
        try:
            parts.append(f"device={t.device}")
        except Exception:
            pass
        local = getattr(t, "_local_tensor", None)
        if local is not None and local is not t:
            parts.append(
                f"_local_tensor.type={type(local).__name__}"
                f" _local_tensor.shape={tuple(local.shape)}"
            )
        data = getattr(t, "data", None)
        if data is not None and data is not t:
            parts.append(f"data.type={type(data).__name__}")
        parts.append(f"mro={[c.__name__ for c in type(t).__mro__[:4]]}")
        return " ".join(parts)

    def _is_fake(t) -> bool:
        """Return True if ``t`` is a FakeTensor, or wraps one at any depth
        (DTensor._local_tensor, subclass __tensor_flatten__ inner tensors, etc.)."""
        if t is None:
            return False
        if isinstance(t, FakeTensor):
            return True
        # DTensor stores the local shard under ``_local_tensor``.
        local = getattr(t, "_local_tensor", None)
        if local is not None and local is not t:
            if _is_fake(local):
                return True
        # Generic tensor subclass: walk inner tensors via __tensor_flatten__.
        flatten = getattr(type(t), "__tensor_flatten__", None)
        if flatten is not None:
            try:
                inner_names, _ctx = flatten(t)
                for n in inner_names:
                    inner = getattr(t, n, None)
                    if inner is not None and inner is not t and _is_fake(inner):
                        return True
            except Exception:
                pass
        # ``.data`` unwrap for plain Parameters.
        data = getattr(t, "data", None)
        if data is not None and data is not t and isinstance(data, FakeTensor):
            return True
        return False

    leaked: list[tuple[str, str]] = []
    for mp_idx, mp in enumerate(model_parts):
        for name, param in mp.named_parameters():
            if _is_fake(param):
                leaked.append(
                    (f"model_part[{mp_idx}].param.{name}", _describe_tensor(param))
                )
        for name, buf in mp.named_buffers():
            if _is_fake(buf):
                leaked.append(
                    (f"model_part[{mp_idx}].buffer.{name}", _describe_tensor(buf))
                )
        # Catch tensors stored as regular instance attributes (outside the
        # Parameter/Buffer registries) on any submodule. E.g. rope_cache,
        # custom caches, etc.
        for mod_name, mod in mp.named_modules():
            for attr_name, val in list(vars(mod).items()):
                if attr_name.startswith("_"):
                    continue
                if isinstance(val, torch.Tensor) and _is_fake(val):
                    leaked.append(
                        (
                            f"model_part[{mp_idx}].{mod_name}.{attr_name} (attr)",
                            _describe_tensor(val),
                        )
                    )

    if leaked:
        logger.error("Fake warmup leaked FakeTensor into persistent module state:")
        for name, desc in leaked:
            logger.error(f"  {name}: {desc}")
        raise RuntimeError(
            f"Fake warmup leaked {len(leaked)} tensor(s) — see log above."
        )
    logger.info(
        f"Post-warmup param/buffer check: all "
        f"{sum(1 for mp in model_parts for _ in mp.parameters())} params and "
        f"{sum(1 for mp in model_parts for _ in mp.buffers())} buffers are real tensors"
    )

    _current_parallel_dims = None

    # Reset all FSDP state-context and per-state lifecycle flags so the
    # first real-training forward re-runs `_lazy_init` and sees a fresh
    # state machine. Without this, warmup's forward+backward leaves
    # `iter_forward_root`, `post_backward_final_callback_queued`,
    # `is_last_backward`, `_training_state`, and the comm-context CUDA
    # streams in a state that subtly skips work the baseline run does
    # (e.g. queueing the post-backward final callback or replaying the
    # all-gather/reduce-scatter stream creation order). Tiny ordering
    # differences in NCCL stream submission cause ~3e-6 loss drift.
    from torch.distributed.fsdp._fully_shard._fsdp_state import (
        FSDPState as _FSDPState, TrainingState as _TS,
    )
    for mp in model_parts:
        for mod in mp.modules():
            get_state = getattr(mod, "_get_fsdp_state", None)
            if get_state is None:
                continue
            try:
                state = get_state()
            except Exception:
                continue
            if not isinstance(state, _FSDPState):
                continue
            state._is_root = None
            state._training_state = _TS.IDLE
            ctx = getattr(state, "_state_ctx", None)
            if ctx is not None:
                ctx.all_states = []
                ctx.iter_forward_root = None
                ctx.post_backward_final_callback_queued = False
                ctx.is_last_backward = True
                ctx.post_optim_event = None
            comm_ctx = getattr(state, "_comm_ctx", None)
            if comm_ctx is not None:
                for attr in (
                    "all_gather_copy_in_stream", "all_gather_stream",
                    "reduce_scatter_stream", "all_reduce_stream",
                    "all_gather_state", "reduce_scatter_state",
                    "post_forward_order",
                ):
                    if hasattr(comm_ctx, attr):
                        try:
                            delattr(comm_ctx, attr)
                        except Exception:
                            pass
            pg = getattr(state, "_fsdp_param_group", None)
            if pg is not None:
                pg._training_state = _TS.IDLE

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _build_warmup_attention_masks(
    model_parts: list[torch.nn.Module],
    local_batch_size: int,
    seq_len: int,
    parallel_dims: Any,
):
    """Pre-build the per-microbatch list of `attention_masks` to feed each
    model_part. Mirrors the logic embedded in ``warmup_stages``: builds a
    single set of masks via the first model_part's ``get_attention_masks``,
    then splits them per microbatch when PP is enabled so the
    `mask_mod.__code__` Dynamo guards see the same `batch_offset_mask_mod`
    closures real training builds in `pipelining.microbatch._split_block_mask`.

    Returns the list of microbatch masks (singleton if PP not enabled or
    attn_type != flex/varlen). For flex/varlen models the elements are
    BlockMask or dict[str, BlockMask]; for sdpa it is ``[None]``.
    """
    shared_attention_masks = None
    _all_split_masks = None
    model_args = getattr(model_parts[0], "model_args", None)
    attn_type = getattr(model_args, "attn_type", "sdpa")
    if attn_type not in ("flex", "varlen"):
        return [None]

    class _StubTokenizer:
        eos_id = 0

    device = f"cuda:{torch.cuda.current_device()}"
    stub_tokens = torch.zeros(
        local_batch_size, seq_len, dtype=torch.long, device=device,
    )
    try:
        # pyrefly: ignore [not-callable]
        shared_attention_masks = model_parts[0].get_attention_masks(
            input_batch=stub_tokens,
            tokenizer=_StubTokenizer(),
            extra_inputs={},
        )
        if (
            len(model_parts) > 1
            and parallel_dims is not None
            and getattr(parallel_dims, "pp_enabled", False)
            and isinstance(shared_attention_masks, BlockMask)
        ):
            from torch.distributed.pipelining.microbatch import _split_block_mask
            num_chunks = max(
                local_batch_size,
                shared_attention_masks.kv_num_blocks.size(0),
            )
            if num_chunks > 1:
                split = _split_block_mask(shared_attention_masks, num_chunks)
                _all_split_masks = split
                shared_attention_masks = split[0]
        elif isinstance(shared_attention_masks, dict):
            if (
                len(model_parts) > 1
                and parallel_dims is not None
                and getattr(parallel_dims, "pp_enabled", False)
            ):
                from torch.distributed.pipelining.microbatch import _split_block_mask
                key_splits: dict[str, list] = {}
                max_chunks = 1
                new_masks = {}
                for k, v in shared_attention_masks.items():
                    if isinstance(v, BlockMask):
                        nc = max(local_batch_size, v.kv_num_blocks.size(0))
                        if nc > 1:
                            s = _split_block_mask(v, nc)
                            new_masks[k] = s[0]
                            key_splits[k] = s
                            max_chunks = max(max_chunks, len(s))
                        else:
                            new_masks[k] = v
                    else:
                        new_masks[k] = v
                shared_attention_masks = new_masks
                if max_chunks > 1:
                    _all_split_masks = []
                    for ci in range(max_chunks):
                        chunk_dict = {}
                        for k, v in new_masks.items():
                            if k in key_splits:
                                chunk_dict[k] = key_splits[k][min(ci, len(key_splits[k]) - 1)]
                            else:
                                chunk_dict[k] = v
                        _all_split_masks.append(chunk_dict)
    except Exception as e:
        logger.warning(
            f"Warmup: get_attention_masks failed "
            f"({type(e).__name__}: {e}), using per-stage recorded masks."
        )

    return (
        _all_split_masks
        if _all_split_masks is not None and len(_all_split_masks) > 1
        else [shared_attention_masks]
    )


def warmup_stages_real(
    model_parts: list[torch.nn.Module],
    record_path: str,
    *,
    loss_fn: Optional[Callable] = None,
    pp_has_last_stage: bool = True,
    local_batch_size: int = 1,
    seq_len: int = 2048,
    parallel_dims: Any = None,
) -> None:
    """Real-tensor variant of :func:`warmup_stages`.

    Runs an actual CUDA forward + backward through each ``model_part`` with
    zero-init recorded inputs (no FakeTensorMode, no MoE/permute stubs). The
    AOT autograd / Inductor compile path sees the exact same FX graph it
    would see at real training time, so the warmup-compiled Dynamo cache is
    bit-compatible with the no-warmup baseline. Drawback vs the fake variant:
    one full real iteration's worth of activation memory and kernel-launch
    time, plus actual collective traffic on the PG.

    ``loss_fn`` is invoked on the last-stage output exactly as in the fake
    path (loss_parallel + dummy zero labels). Gradients are zeroed and
    pre-warmup grads are restored, mirroring the cleanup the fake path
    does. FSDP unsharded-cache purge from the fake path is skipped here
    because FSDP's all-gather under real CUDA does not leave fake state
    behind (the cached unsharded param is a real ``nn.Parameter``).
    """
    global _current_parallel_dims
    _current_parallel_dims = parallel_dims

    if not os.path.exists(record_path):
        logger.warning(
            f"Stage input record file not found at {record_path}, skipping warmup"
        )
        _current_parallel_dims = None
        return

    with open(record_path, "r") as f:
        data = json.load(f)
    recorded_stages = data["stages"]

    if len(recorded_stages) != len(model_parts):
        logger.warning(
            f"Mismatch between recorded stages ({len(recorded_stages)}) "
            f"and model_parts ({len(model_parts)}). Skipping warmup."
        )
        _current_parallel_dims = None
        return

    logger.info(
        f"Starting REAL stage warmup for {len(model_parts)} model_parts "
        "(real CUDA forward+backward)..."
    )

    # Snapshot real grads so the warmup backward's grad accumulation does
    # not pollute resilient_opt's state. Restored after warmup.
    saved_grads: list[dict[int, torch.Tensor]] = []
    for mp in model_parts:
        grads_for_mp: dict[int, torch.Tensor] = {}
        for p in mp.parameters():
            if p.grad is not None:
                grads_for_mp[id(p)] = p.grad
                p.grad = None
        saved_grads.append(grads_for_mp)

    # Snapshot every buffer so any in-place mutation during the warmup
    # forward is undone. The MoE module's
    # ``self.tokens_per_expert.add_(num_tokens_per_expert)`` is the
    # canonical case: ``build_optimizers_with_moe_load_balancing``'s
    # pre-hook reads this counter at every optimizer step, so even one
    # warmup forward shifts the load-balance bias trajectory and breaks
    # bit-identity at step 2+ for gpt-oss / deepseek-moe. (The
    # FakeTensorMode warmup is immune to this — in-place ops on real
    # storage under fake mode return a fake result without writing real
    # memory.) Cloning every buffer is overkill but cheap and
    # future-proof against any other in-place buffer mutation.
    saved_buffers: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    for mp in model_parts:
        snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _, buf in mp.named_buffers():
            try:
                snapshots.append((buf, buf.detach().clone()))
            except Exception:
                pass
        saved_buffers.append(snapshots)

    microbatch_masks = _build_warmup_attention_masks(
        model_parts, local_batch_size, seq_len, parallel_dims,
    )

    for mb_idx, _current_microbatch_mask in enumerate(microbatch_masks):
        if mb_idx > 0:
            logger.info(
                f"Real warmup: extra microbatch pass {mb_idx} "
                f"(compiling for batch_offset={mb_idx})"
            )
        for stage_idx, (model_part, stage_metadata) in enumerate(
            zip(model_parts, recorded_stages)
        ):
            if stage_metadata is None:
                logger.warning(
                    f"No recorded inputs for stage {stage_idx}, skipping"
                )
                continue

            synthetic_args, synthetic_kwargs = _deserialize_args(stage_metadata)
            try:
                if (
                    _current_microbatch_mask is not None
                    and "attention_masks" in synthetic_kwargs
                ):
                    synthetic_kwargs["attention_masks"] = _current_microbatch_mask

                output = model_part(*synthetic_args, **synthetic_kwargs)

                loss = None
                is_last_stage_module = (
                    pp_has_last_stage and stage_idx == len(model_parts) - 1
                )
                if (
                    is_last_stage_module
                    and loss_fn is not None
                    and isinstance(output, torch.Tensor)
                ):
                    labels = torch.zeros(
                        output.shape[:-1],
                        dtype=torch.long,
                        device=output.device,
                    )
                    from torch.distributed.tensor import DTensor

                    if isinstance(output, DTensor):
                        from torch.distributed.tensor.parallel import loss_parallel
                        with loss_parallel():
                            loss = loss_fn(output, labels)
                            if loss is not None:
                                loss.backward()
                    else:
                        loss = loss_fn(output, labels)
                        if loss is not None:
                            loss.backward()
                else:
                    loss = _reduce_output_to_scalar(output)
                    if loss is not None:
                        loss.backward()

                del synthetic_args, synthetic_kwargs, output
            except Exception as e:
                logger.warning(
                    f"Real warmup failed for stage {stage_idx} "
                    f"(mb={mb_idx}): {type(e).__name__}: {e}. "
                    "This stage will pay full compile cost on the first "
                    "real step."
                )
                raise

            model_part.zero_grad(set_to_none=True)

    # Restore the real grads we snapshotted before warmup.
    for mp, grads_for_mp in zip(model_parts, saved_grads):
        for p in mp.parameters():
            p.grad = grads_for_mp.get(id(p))

    # Restore every buffer's pre-warmup contents in-place.
    for snapshots in saved_buffers:
        for buf, snap in snapshots:
            try:
                with torch.no_grad():
                    buf.copy_(snap)
            except Exception as e:
                logger.warning(
                    f"Real warmup: failed to restore buffer "
                    f"({type(e).__name__}: {e})"
                )

    _current_parallel_dims = None
    torch.cuda.synchronize()
    gc.collect()


def _reload_compiled_fx_graphs_after_warmup() -> None:
    """Reload the PyCodeCache module behind every live `CompiledFxGraph`.

    Confirmed root cause of an otherwise-residual ~3e-6 step-1 drift
    when warmup keeps the Dynamo cache and reuses warmup-built compiled
    artifacts at real time: `CompiledFxGraph.current_callable` is a
    closure (`align_inputs_from_check_idxs.<locals>.run`) that wraps a
    `module.call` from a PyCodeCache module instance loaded at warmup
    time. That module's globals capture the `CachingAutotuner`
    instances created during warmup `precompile()`, which carry
    warmup-time launcher state. When real training reuses the warmup
    module, the warmup-time launchers feed `autotune_to_one_config`
    with stale benchmark context and pick a different chosen kernel
    config than a fresh-from-disk module would.

    Fix (no Dynamo retrace, no real GPU kernels): mirror what
    `cache_hit_post_compile` does at retrace time, but in-place:
      - `prepare_for_serialization()` strips `current_callable` /
        `recursively_apply_fns` / `compiled_fn_runner` so the graph
        looks like a freshly-pickled one;
      - `after_deserialization(constants)` invokes
        `PyCodeCache.load_by_key_path(...)`. Because every Inductor-
        generated module for an FX graph with constants is loaded with
        `attrs is not None`, the cache always misses and triggers a
        fresh `_reload_python_module(...)`. The new module has fresh
        `CachingAutotuner` globals.

    Result: bit-identical losses across all four MoE / non-MoE
    workloads, no Dynamo retrace, ~45–64% step-1 wall-time saving from
    Triton-disk-cache reuse.
    """
    try:
        import gc
        from torch._inductor.output_code import (
            CompiledFxGraph,
            CompiledFxGraphConstants,
        )
    except Exception as e:
        logger.warning(
            f"Stage warmup post-warmup reload setup failed: "
            f"{type(e).__name__}: {e}"
        )
        return

    constants = CompiledFxGraphConstants()
    n_reloaded = 0
    n_failed = 0
    for obj in gc.get_objects():
        if not isinstance(obj, CompiledFxGraph):
            continue
        try:
            obj.prepare_for_serialization()
            obj.after_deserialization(constants)
            n_reloaded += 1
        except Exception as e:
            logger.warning(
                f"Stage warmup: CompiledFxGraph.after_deserialization "
                f"failed: {type(e).__name__}: {e}"
            )
            n_failed += 1
    logger.info(
        f"Stage warmup: reloaded PyCodeCache module on {n_reloaded} "
        f"CompiledFxGraphs ({n_failed} failed)"
    )
