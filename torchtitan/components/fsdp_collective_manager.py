"""Unified manager for custom FSDP collective operations.

Composes custom allocate() and __call__() from different components
(e.g., Gemini for snapshot callbacks, RMP for gradient persistence)
into unified AllGather/ReduceScatter objects registered via the
official FSDP set_custom_* APIs.
"""

from typing import Callable, Sequence, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather, ReduceScatter
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    DefaultAllGather,
    DefaultReduceScatter,
)

from torchtitan.tools.logging import logger

_ReduceOp = Union[dist.ReduceOp, dist.ReduceOp.RedOpType]

# Callback types
AllGatherCallFn = Callable[
    [torch.Tensor, torch.Tensor, dist.ProcessGroup, bool],
    dist.Work | None,
]
ReduceScatterCallFn = Callable[
    [torch.Tensor, torch.Tensor, dist.ProcessGroup, _ReduceOp, bool],
    dist.Work | None,
]
AllocateFn = Callable[
    [Sequence[int | torch.SymInt], torch.dtype, torch.device],
    torch.Tensor,
]


class _UnifiedAllGather(AllGather):
    def __init__(self, call_fn: AllGatherCallFn | None = None):
        self._call_fn = call_fn
        self._default = DefaultAllGather()

    def allocate(self, size, *, dtype, device):
        return self._default.allocate(size, dtype=dtype, device=device)

    def __call__(self, output_tensor, input_tensor, group, async_op=False):
        if self._call_fn is not None:
            return self._call_fn(output_tensor, input_tensor, group, async_op)
        return self._default(output_tensor, input_tensor, group=group, async_op=async_op)


class _UnifiedReduceScatter(ReduceScatter):
    """One instance per FSDP module, composes custom allocate and __call__."""

    def __init__(
        self,
        call_fn: ReduceScatterCallFn | None = None,
        allocate_fn: AllocateFn | None = None,
    ):
        self._call_fn = call_fn
        self._allocate_fn = allocate_fn
        self._default = DefaultReduceScatter()

    def allocate(self, size, *, dtype, device):
        if self._allocate_fn is not None:
            return self._allocate_fn(size, dtype, device)
        return self._default.allocate(size, dtype=dtype, device=device)

    def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
        if self._call_fn is not None:
            return self._call_fn(output_tensor, input_tensor, group, op, async_op)
        return self._default(
            output_tensor=output_tensor, input_tensor=input_tensor,
            group=group, op=op, async_op=async_op,
        )


class FsdpCollectiveManager:
    """Manages custom FSDP collective registrations from multiple components.

    Usage:
        manager = FsdpCollectiveManager()
        manager.set_all_gather_call(gemini_ag_fn)
        manager.set_reduce_scatter_call(gemini_rs_fn)
        manager.set_reduce_scatter_allocate(module, allocate_fn)
        manager.attach(model_parts)
    """

    def __init__(self) -> None:
        self._ag_call_fn: AllGatherCallFn | None = None
        self._rs_call_fn: ReduceScatterCallFn | None = None
        self._rs_allocate_fns: dict[int, AllocateFn] = {}  # keyed by id(module)

    def set_all_gather_call(self, fn: AllGatherCallFn) -> None:
        self._ag_call_fn = fn

    def set_reduce_scatter_call(self, fn: ReduceScatterCallFn) -> None:
        self._rs_call_fn = fn

    def set_reduce_scatter_allocate(self, module: nn.Module, fn: AllocateFn) -> None:
        """Set custom allocate for a specific FSDP module's reduce-scatter."""
        self._rs_allocate_fns[id(module)] = fn

    def attach(self, model_parts: list[torch.nn.Module]) -> None:
        """Create unified comm objects and register via official FSDP APIs."""
        if self._ag_call_fn is None and self._rs_call_fn is None and not self._rs_allocate_fns:
            return

        ag = _UnifiedAllGather(call_fn=self._ag_call_fn)

        count = 0
        for model in model_parts:
            for module in model.modules():
                if not isinstance(module, FSDPModule):
                    continue

                module.set_custom_all_gather(ag)

                rs = _UnifiedReduceScatter(
                    call_fn=self._rs_call_fn,
                    allocate_fn=self._rs_allocate_fns.get(id(module)),
                )
                module.set_custom_reduce_scatter(rs)
                count += 1

        logger.info(
            f"[FsdpCollectiveManager] Attached to {count} FSDP modules "
            f"({len(self._rs_allocate_fns)} with custom allocate)"
        )
