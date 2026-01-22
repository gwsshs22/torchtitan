# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Leto measurement utilities for tracking training metrics."""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from torchtitan.tools.logging import logger


@dataclass
class IterationRecord:
    """Record for a single training iteration's checkpoint measurements."""
    checkpointed: bool
    checkpoint_start_time: float  # Internal, for computing durations
    staging_time_sec: float = 0.0
    checkpoint_total_time_sec: float = 0.0
    iteration_time_sec: float = 0.0

    def set_staging_done(self) -> None:
        """Record that staging is complete."""
        self.staging_time_sec = time.time() - self.checkpoint_start_time

    def set_checkpoint_done(self) -> None:
        """Record that checkpoint save is complete."""
        self.checkpoint_total_time_sec = time.time() - self.checkpoint_start_time


class LetoMeasurements:
    """Tracks and records training measurements for Leto launcher."""

    def __init__(self) -> None:
        self._process_start_time = time.time()
        self._weight_allocation_init_time_sec = 0.0
        self._checkpoint_load_time_sec = 0.0
        self._time_to_train_start_sec = 0.0
        self._model_size_bytes = 0
        self._optimizer_state_size_bytes = 0
        self._total_checkpoint_size_bytes = 0
        self._iterations: list[IterationRecord] = []

    def record_weight_allocation_time(self, duration: float) -> None:
        """Record the time taken for weight allocation initialization."""
        self._weight_allocation_init_time_sec = duration

    def record_checkpoint_load_time(self, duration: float) -> None:
        """Record the time taken to load checkpoint."""
        self._checkpoint_load_time_sec = duration

    def record_time_to_train_start(self) -> None:
        """Record the time from process start to training start."""
        self._time_to_train_start_sec = time.time() - self._process_start_time
        logger.info(f"[Leto] Time to train start sec = {self._time_to_train_start_sec}")

    def record_iteration_time(self, duration: float) -> None:
        """Record an iteration time. Must be called after report_checkpoint_start."""
        if self._iterations:
            self._iterations[-1].iteration_time_sec = duration

    def report_checkpoint_start(self, should_save: bool) -> IterationRecord:
        """Create a new iteration record. Returns the record for direct updates."""
        record = IterationRecord(
            checkpointed=should_save,
            checkpoint_start_time=time.time(),
        )
        self._iterations.append(record)
        return record

    def calculate_sizes(
        self,
        model_parts: list[torch.nn.Module],
        optimizers: list[torch.optim.Optimizer],
    ) -> None:
        """Calculate model and optimizer state sizes for measurements."""
        # Calculate model parameter size (in bytes, assuming current dtype)
        model_size = 0
        for m in model_parts:
            for p in m.parameters():
                model_size += p.numel() * p.element_size()
        self._model_size_bytes = model_size

        # Calculate optimizer state size
        # For Adam/AdamW, optimizer states include: exp_avg (momentum) and exp_avg_sq (variance)
        # Each is the same size as parameters, so roughly 2x model size
        # But we need to account for the actual state after first step
        optimizer_state_size = 0
        for opt in optimizers:
            for param_group in opt.param_groups:
                for p in param_group["params"]:
                    if p in opt.state:
                        state = opt.state[p]
                        for key, val in state.items():
                            if isinstance(val, torch.Tensor):
                                optimizer_state_size += val.numel() * val.element_size()
                    else:
                        # If state not yet initialized, estimate based on Adam
                        # (2 tensors: exp_avg and exp_avg_sq, each same size as param, in fp32)
                        optimizer_state_size += p.numel() * 4 * 2  # 4 bytes for fp32, 2 buffers

        self._optimizer_state_size_bytes = optimizer_state_size

        # Total checkpoint size = model (fp32) + optimizer states
        # Note: checkpoints are typically saved in fp32
        model_fp32_size = sum(
            p.numel() * 4 for m in model_parts for p in m.parameters()
        )
        self._total_checkpoint_size_bytes = model_fp32_size + optimizer_state_size

        logger.info(
            f"[Leto] Model size: {model_size / 1e9:.2f} GB, "
            f"Optimizer state size: {optimizer_state_size / 1e9:.2f} GB, "
            f"Total checkpoint size: {self._total_checkpoint_size_bytes / 1e9:.2f} GB"
        )

    def _to_dict(self) -> dict[str, Any]:
        """Convert measurements to dictionary format for serialization."""
        return {
            "process_start_time": self._process_start_time,
            "weight_allocation_init_time_sec": self._weight_allocation_init_time_sec,
            "checkpoint_load_time_sec": self._checkpoint_load_time_sec,
            "time_to_train_start_sec": self._time_to_train_start_sec,
            "iteration_times_sec": [r.iteration_time_sec for r in self._iterations],
            "checkpointed": [r.checkpointed for r in self._iterations],
            "checkpoint_total_times_sec": [r.checkpoint_total_time_sec for r in self._iterations],
            "staging_times_sec": [r.staging_time_sec for r in self._iterations],
            "model_size_bytes": self._model_size_bytes,
            "optimizer_state_size_bytes": self._optimizer_state_size_bytes,
            "total_checkpoint_size_bytes": self._total_checkpoint_size_bytes,
        }

    def write(self) -> None:
        """Write measurements to JSON file."""
        data = self._to_dict()
        logger.info(f"[Leto] Measurements={json.dumps(data, indent=2)}")

        # Get logs directory from environment variable set by leto launcher
        logs_dir = os.environ.get("LETO_LOGS_DIR", "")
        if not logs_dir:
            logger.debug("[Leto] LETO_LOGS_DIR not set, skipping measurements output")
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        output_file = os.path.join(logs_dir, f"measure_rank_{rank:02d}.json")
        try:
            os.makedirs(logs_dir, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"[Leto] Measurements written to {output_file}")
        except Exception as e:
            logger.warning(f"[Leto] Failed to write measurements to {output_file}: {e}")


# Global instance for easy access
_measurements = LetoMeasurements()


def get_measurements() -> LetoMeasurements:
    """Get the global LetoMeasurements instance."""
    return _measurements
