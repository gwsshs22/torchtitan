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

import json
import os
import types
from typing import Any

import torch
import torch.distributed as dist

from torchtitan.tools.logging import logger


def _tensor_to_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    """Convert a tensor to metadata dict with shape, dtype, and device."""
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _metadata_to_meta_tensor(metadata: dict[str, Any]) -> torch.Tensor:
    """Create a meta-device tensor from recorded metadata (shape + dtype only)."""
    dtype_str = metadata["dtype"]
    dtype = getattr(torch, dtype_str.split(".")[-1])
    return torch.empty(metadata["shape"], dtype=dtype, device="meta")


def _serialize_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Serialize args and kwargs to JSON-compatible format."""
    return {
        "args": [
            _tensor_to_metadata(arg) if isinstance(arg, torch.Tensor) else None
            for arg in args
        ],
        "kwargs": {
            k: _tensor_to_metadata(v) if isinstance(v, torch.Tensor) else None
            for k, v in kwargs.items()
        },
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
    Non-tensor (null) entries are skipped.
    """
    return tuple(
        _metadata_to_meta_tensor(entry)
        for entry in metadata["args"]
        if entry is not None
    )


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

    if job_config.parallelism.pipeline_parallel_degree <= 1:
        return None

    # Install hooks before first iteration
    if step == 1 and recorder is None:
        recorder = StageInputRecorder(model_parts)
        recorder.install_hooks()
        logger.info("Stage input/output recording enabled for first iteration")
        return recorder

    # Save and cleanup after first iteration
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
