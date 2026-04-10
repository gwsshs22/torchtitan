"""Resilient optimizer (CPU snapshot): snapshot all state to pinned CPU memory.

Before optimizer.step(), copies all params + exp_avg + exp_avg_sq to
pinned CPU memory.  If a fault corrupts the in-place fused AdamW update,
restores from the CPU snapshot and replays.

This is the simplest baseline — trades GPU→CPU bandwidth for simplicity.
Extra memory = total state size in pinned CPU RAM (not GPU).

Usage:
    resilient_opt = ResilientOptimizerCpuSnapshot(optimizers, device)
    # In training loop:
    resilient_opt.step()
"""

import torch
from torch.distributed._tensor import DTensor

from torchtitan.tools.logging import logger


def _get_local(tensor):
    if isinstance(tensor, DTensor):
        return tensor._local_tensor
    return tensor


class ResilientOptimizerCpuSnapshot:
    """Snapshot all optimizer inputs to pinned CPU memory before each step."""

    def __init__(self, optimizers, device: torch.device):
        self._optimizers = optimizers
        self._device = device

        # Collect all (param, exp_avg, exp_avg_sq) tensors
        self._params: list[torch.Tensor] = []
        self._exp_avgs: list[torch.Tensor] = []
        self._exp_avg_sqs: list[torch.Tensor] = []

        for optimizer in optimizers:
            for param, state in optimizer.state.items():
                self._params.append(param)
                self._exp_avgs.append(state["exp_avg"])
                self._exp_avg_sqs.append(state["exp_avg_sq"])

        # Pre-allocate pinned CPU tensors matching each GPU tensor
        self._cpu_params = [
            torch.empty_like(_get_local(p), device="cpu").pin_memory()
            for p in self._params
        ]
        self._cpu_exp_avgs = [
            torch.empty_like(_get_local(m), device="cpu").pin_memory()
            for m in self._exp_avgs
        ]
        self._cpu_exp_avg_sqs = [
            torch.empty_like(_get_local(v), device="cpu").pin_memory()
            for v in self._exp_avg_sqs
        ]

        total_bytes = sum(
            t.numel() * t.element_size()
            for t in self._cpu_params + self._cpu_exp_avgs + self._cpu_exp_avg_sqs
        )
        logger.info(
            f"[ResilientOptCPU] {len(self._params)} params, "
            f"CPU snapshot: {total_bytes / (1024**2):.1f} MB pinned"
        )

    def step(self):
        """Snapshot to CPU → optimizer step → done."""
        self._snapshot()
        torch.cuda.current_stream().synchronize()
        self._optimizers.step()
        torch.cuda.current_stream().synchronize()

    def restore_and_replay(self):
        """Restore from CPU snapshot and replay the step."""
        self._restore()
        torch.cuda.current_stream().synchronize()
        self._optimizers.step()
        torch.cuda.current_stream().synchronize()

    @torch.no_grad()
    def _snapshot(self):
        """GPU → pinned CPU (async copy via non_blocking)."""
        for cpu, gpu in zip(self._cpu_params, self._params):
            cpu.copy_(_get_local(gpu), non_blocking=True)
        for cpu, gpu in zip(self._cpu_exp_avgs, self._exp_avgs):
            cpu.copy_(_get_local(gpu), non_blocking=True)
        for cpu, gpu in zip(self._cpu_exp_avg_sqs, self._exp_avg_sqs):
            cpu.copy_(_get_local(gpu), non_blocking=True)

    @torch.no_grad()
    def _restore(self):
        """Pinned CPU → GPU."""
        for cpu, gpu in zip(self._cpu_params, self._params):
            _get_local(gpu).copy_(cpu, non_blocking=True)
        for cpu, gpu in zip(self._cpu_exp_avgs, self._exp_avgs):
            _get_local(gpu).copy_(cpu, non_blocking=True)
        for cpu, gpu in zip(self._cpu_exp_avg_sqs, self._exp_avg_sqs):
            _get_local(gpu).copy_(cpu, non_blocking=True)


class AsyncCpuSnapshotOptimizer:
    """Async CPU snapshot baseline: kick off GPU→CPU on a side stream at
    start of step, wait for completion before optimizer.step().

    No recovery — purely for measuring snapshot latency overhead vs
    ResilientOptimizer.
    """

    def __init__(self, optimizers, device: torch.device):
        self._optimizers = optimizers
        self._device = device
        self._copy_stream = torch.cuda.Stream(device=device)
        self._snapshot_event: torch.cuda.Event | None = None

        self._params: list[torch.Tensor] = []
        self._exp_avgs: list[torch.Tensor] = []
        self._exp_avg_sqs: list[torch.Tensor] = []
        for optimizer in optimizers:
            for param, state in optimizer.state.items():
                self._params.append(param)
                self._exp_avgs.append(state["exp_avg"])
                self._exp_avg_sqs.append(state["exp_avg_sq"])

        self._cpu_params = [
            torch.empty_like(_get_local(p), device="cpu").pin_memory()
            for p in self._params
        ]
        self._cpu_exp_avgs = [
            torch.empty_like(_get_local(m), device="cpu").pin_memory()
            for m in self._exp_avgs
        ]
        self._cpu_exp_avg_sqs = [
            torch.empty_like(_get_local(v), device="cpu").pin_memory()
            for v in self._exp_avg_sqs
        ]

        total_bytes = sum(
            t.numel() * t.element_size()
            for t in self._cpu_params + self._cpu_exp_avgs + self._cpu_exp_avg_sqs
        )
        logger.info(
            f"[AsyncCpuSnapshot] {len(self._params)} params, "
            f"CPU snapshot: {total_bytes / (1024**2):.1f} MB pinned"
        )

    @torch.no_grad()
    def begin_snapshot(self):
        """Kick off async GPU→CPU on the side stream."""
        # Side stream waits for prior optimizer.step() on default stream.
        self._copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._copy_stream):
            for cpu, gpu in zip(self._cpu_params, self._params):
                cpu.copy_(_get_local(gpu), non_blocking=True)
            for cpu, gpu in zip(self._cpu_exp_avgs, self._exp_avgs):
                cpu.copy_(_get_local(gpu), non_blocking=True)
            for cpu, gpu in zip(self._cpu_exp_avg_sqs, self._exp_avg_sqs):
                cpu.copy_(_get_local(gpu), non_blocking=True)
        self._snapshot_event = self._copy_stream.record_event()

    def step(self):
        """Wait for snapshot completion, then run optimizer step."""
        if self._snapshot_event is None:
            raise RuntimeError(
                "AsyncCpuSnapshotOptimizer.step() called before begin_snapshot()"
            )

        # Default stream waits for the side-stream copy to finish before
        # optimizer.step() writes to the same tensors.
        torch.cuda.current_stream().wait_event(self._snapshot_event)
        self._snapshot_event = None
        self._optimizers.step()
