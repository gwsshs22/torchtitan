# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Stage input recording and warmup utilities for pipeline parallel training.

This module provides functionality to:
1. Record stage inputs (shapes, dtypes, devices) during the first training iteration
2. Replay recorded inputs to warmup stages (including torch.compile) before training

The warmup helps torch.compile create the backward graph during forward pass (AOT),
warming up both forward and backward compilation without needing actual data.
"""

import gc
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


def _metadata_to_tensor(metadata: dict[str, Any]) -> torch.Tensor:
    """Create a random tensor from metadata."""
    dtype_str = metadata["dtype"]
    # Parse dtype string like "torch.float32" to actual dtype
    dtype = getattr(torch, dtype_str.split(".")[-1])

    device_str = metadata["device"]

    tensor = torch.randn(metadata["shape"], dtype=dtype, device=device_str)
    # Enable gradient tracking for compile warmup
    if dtype.is_floating_point:
        tensor.requires_grad = True

    return tensor


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


def _deserialize_args(metadata: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Deserialize metadata to synthetic tensors."""
    args = tuple(
        _metadata_to_tensor(arg_meta) if arg_meta is not None else None
        for arg_meta in metadata["args"]
    )
    kwargs = {
        k: _metadata_to_tensor(v_meta) if v_meta is not None else None
        for k, v_meta in metadata["kwargs"].items()
    }
    # Filter out None values
    args = tuple(arg for arg in args if arg is not None)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    return args, kwargs


class StageInputRecorder:
    """
    Records inputs to model_parts during the first training iteration.

    This hooks into each model_part's forward method to capture the shapes,
    dtypes, and devices of input tensors.
    """

    def __init__(self, model_parts: list[torch.nn.Module]):
        self.model_parts = model_parts
        self.recorded_inputs: list[dict[str, Any] | None] = [None] * len(model_parts)
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
                    # Record inputs (only once per stage)
                    if recorder.recorded_inputs[stage_idx] is None:
                        recorder.recorded_inputs[stage_idx] = _serialize_args(args, kwargs)
                        logger.debug(
                            f"Recorded inputs for model_part {stage_idx}: "
                            f"{len(args)} args, {len(kwargs)} kwargs"
                        )
                    # Call original forward (it's already bound, so no self needed)
                    return orig_fwd(*args, **kwargs)

                return recording_forward

            # Bind the wrapper to the model_part instance
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
                model_part.forward = types.MethodType(
                    self.original_forwards[idx], model_part
                )

        self.hooks_installed = False
        logger.info("Removed recording hooks from model_parts")

    def save(self, save_path: str) -> None:
        """Save recorded inputs to JSON file."""
        # Ensure all stages have been recorded
        if any(rec is None for rec in self.recorded_inputs):
            unrecorded = [i for i, rec in enumerate(self.recorded_inputs) if rec is None]
            logger.warning(
                f"Not all stages have recorded inputs. Missing stages: {unrecorded}"
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w") as f:
            json.dump({"stages": self.recorded_inputs}, f, indent=2)

        logger.info(f"Saved stage inputs to {save_path}")


def warmup_stages(
    model_parts: list[torch.nn.Module],
    record_path: str,
) -> None:
    """
    Warmup stages by running forward pass with recorded synthetic inputs.

    This helps torch.compile create the backward graph during forward pass,
    warming up both forward and backward compilation.

    Args:
        model_parts: List of model parts (stage submodules) to warmup
        record_path: Path to the JSON file with recorded input metadata
    """
    if not os.path.exists(record_path):
        logger.warning(
            f"Stage input record file not found at {record_path}, skipping warmup"
        )
        return

    # Load recorded inputs
    with open(record_path, "r") as f:
        data = json.load(f)

    recorded_stages = data["stages"]

    if len(recorded_stages) != len(model_parts):
        logger.warning(
            f"Mismatch between recorded stages ({len(recorded_stages)}) "
            f"and model_parts ({len(model_parts)}). This may happen if pipeline "
            f"configuration changed. Skipping warmup."
        )
        return

    logger.info(f"Starting stage warmup for {len(model_parts)} model_parts...")

    for stage_idx, (model_part, stage_metadata) in enumerate(
        zip(model_parts, recorded_stages)
    ):
        if stage_metadata is None:
            logger.warning(f"No recorded inputs for stage {stage_idx}, skipping")
            continue

        # Create synthetic inputs from metadata
        synthetic_args, synthetic_kwargs = _deserialize_args(stage_metadata)

        logger.debug(
            f"Warming up stage {stage_idx} with {len(synthetic_args)} args, "
            f"{len(synthetic_kwargs)} kwargs"
        )

        # Run forward pass (with gradient tracking for compile warmup)
        try:
            output = model_part(*synthetic_args, **synthetic_kwargs)

            # Cleanup: delete tensors to release autograd graph
            del synthetic_args, synthetic_kwargs, output
            torch.cuda.synchronize()  # Ensure CUDA operations complete
            gc.collect()

        except Exception as e:
            logger.error(f"Failed to warmup stage {stage_idx}: {e}")
            raise

    # Final cleanup
    torch.cuda.empty_cache()

    logger.info("Stage warmup completed successfully")


def maybe_record_stage_inputs(
    model_parts: list[torch.nn.Module],
    job_config,
    step: int,
    recorder: StageInputRecorder | None = None,
) -> StageInputRecorder | None:
    """
    Handle stage input recording based on config and training step.

    Args:
        model_parts: List of model parts to record
        job_config: Job configuration
        step: Current training step
        recorder: Existing recorder instance (if any)

    Returns:
        Recorder instance if recording is active, None otherwise
    """
    if not job_config.leto.enable_stage_input_record:
        return None

    # Install hooks before first iteration
    if step == 0 and recorder is None:
        recorder = StageInputRecorder(model_parts)
        recorder.install_hooks()
        logger.info("Stage input recording enabled for first iteration")
        return recorder

    # Save and cleanup after first iteration
    if step == 1 and recorder is not None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        save_folder = os.path.join(
            job_config.job.dump_folder, job_config.leto.stage_inputs_folder
        )
        save_path = os.path.join(save_folder, f"rank_{rank}.json")

        recorder.save(save_path)
        recorder.remove_hooks()

        logger.info(f"Stage input recording completed and saved to {save_path}")
        return None

    return recorder


def maybe_warmup_stages(
    model_parts: list[torch.nn.Module],
    job_config,
) -> None:
    """
    Warmup stages with recorded inputs if enabled in config.

    Args:
        model_parts: List of model parts to warmup
        job_config: Job configuration
    """
    if not job_config.leto.enable_stage_warmup:
        return

    rank = dist.get_rank() if dist.is_initialized() else 0
    record_folder = os.path.join(
        job_config.job.dump_folder, job_config.leto.stage_inputs_folder
    )
    record_path = os.path.join(record_folder, f"rank_{rank}.json")

    warmup_stages(model_parts, record_path)
