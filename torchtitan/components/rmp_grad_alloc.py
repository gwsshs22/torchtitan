"""RMP-backed gradient allocator for FSDP2 reduce-scatter output buffers.

Registers per-group allocate functions on an FsdpCollectiveManager.
RS output buffers are persisted in RMP across process restarts,
enabling resilient optimizer replay. Tensors are keyed by a stable
group index so they survive ordering changes between runs.
"""

from typing import Sequence

import torch
from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

from leto.rmp.client import RmpClient
from torchtitan.components.fsdp_collective_manager import FsdpCollectiveManager
from torchtitan.tools.logging import logger


class _PerGroupAllocState:
    """Per-param-group allocator state."""

    __slots__ = ("is_input", "cached_output", "key")

    def __init__(self, cached_output: torch.Tensor | None = None, key: str = ""):
        self.is_input: bool = True  # toggles between input/output
        self.cached_output: torch.Tensor | None = cached_output
        self.key = key


class RmpGradientAllocator:
    """Allocates FSDP2 reduce-scatter output buffers from RMP.

    Each FSDP param group's foreach_reduce calls allocate() exactly twice
    per backward pass:
      - 1st call: RS input buffer (P-sized, transient)
      - 2nd call: RS output buffer (P/N-sized, persistent in RMP)

    Tensors are keyed by stable group index on the RMP server, so restart
    correctly matches tensors to groups regardless of allocation order.
    """

    def __init__(self, rmp_client: RmpClient, device: torch.device, allocated: bool):
        self._rmp_client = rmp_client
        self._device = device

        # Restart: fetch all gradient buffers upfront as key → tensor dict
        self._prefetched: dict[str, torch.Tensor] | None = None
        if not allocated:
            self._prefetched = self._rmp_client.get_gradient_tensor_list()
            total_bytes = sum(
                t.numel() * t.element_size() for t in self._prefetched.values()
            )
            logger.info(
                f"[RmpGradAlloc] Retrieved {len(self._prefetched)} "
                f"RS output buffers ({total_bytes / (1024*1024):.2f} MB total)"
            )

    def register(
        self,
        collective_manager: FsdpCollectiveManager,
        model_parts: list[torch.nn.Module],
    ) -> None:
        """Register per-group allocate functions on the collective manager."""
        group_idx = 0
        registered = 0
        for model in model_parts:
            for module in model.modules():
                state = _get_module_fsdp_state(module)
                if state is None or state._fsdp_param_group is None:
                    continue
                param_group = state._fsdp_param_group

                key = f"grad_rs/{group_idx}"
                group_idx += 1

                if not any(
                    p.sharded_param.requires_grad for p in param_group.fsdp_params
                ):
                    continue

                cached = None
                if self._prefetched is not None:
                    cached = self._prefetched.get(key)

                pg_state = _PerGroupAllocState(cached_output=cached, key=key)
                allocate_fn = self._make_allocate_fn(pg_state)
                collective_manager.set_reduce_scatter_allocate(module, allocate_fn)
                registered += 1

        self._prefetched = None
        logger.info(
            f"[RmpGradAlloc] Registered {registered} per-group allocators "
            f"on collective manager"
        )

    def _make_allocate_fn(self, pg_state: _PerGroupAllocState):
        """Create a per-group allocate function."""
        rmp_client = self._rmp_client

        def allocate(
            size: Sequence[int | torch.SymInt],
            dtype: torch.dtype,
            device: torch.device,
        ) -> torch.Tensor:
            if pg_state.is_input:
                # RS input buffer (transient)
                pg_state.is_input = False
                return torch.empty(*size, dtype=dtype, device=device)
            else:
                # RS output buffer (persistent, RMP-backed)
                pg_state.is_input = True
                if pg_state.cached_output is not None:
                    return pg_state.cached_output

                shape = tuple(int(s) for s in size)
                device_idx = device.index if hasattr(device, 'index') else 0
                tensor = rmp_client.allocate_gradient_tensor(
                    name=pg_state.key,
                    shape=shape,
                    dtype=dtype,
                    device=device_idx,
                )
                pg_state.cached_output = tensor
                logger.info(
                    f"[RmpGradAlloc] Allocated RS output buffer '{pg_state.key}': "
                    f"numel={tensor.numel()}, dtype={dtype}, device={device}"
                )
                return tensor

        return allocate
