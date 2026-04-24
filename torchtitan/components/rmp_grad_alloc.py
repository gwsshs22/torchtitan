"""RMP-backed gradient allocator for FSDP2 reduce-scatter output buffers.

Registers per-group allocate functions on an FsdpCollectiveManager.
RS output buffers are persisted in RMP across process restarts,
enabling resilient optimizer replay. Tensors are keyed by a stable
group index so they survive ordering changes between runs.
"""

from typing import Sequence

import torch
from torch._guards import detect_fake_mode
from torch.distributed._tensor import DTensor
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

    def restore_param_gradients(
        self,
        model_parts: list[torch.nn.Module],
    ) -> int:
        """Assign prefetched RMP gradient tensors back to param.grad.

        Must be called before register() (which clears _prefetched).
        Walks FSDP param groups in the same order as register() to
        match ``grad_rs/{group_idx}`` keys to parameters.

        Handles padding: the RS output buffer may be larger than the
        param's numel (due to world-size padding), so we take a
        ``[:numel]`` slice.

        Returns the number of gradients restored.
        """
        if self._prefetched is None:
            return 0

        restored = 0
        group_idx = 0
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

                grad_tensor = self._prefetched.get(key)
                if grad_tensor is None:
                    continue

                grad_flat = grad_tensor.view(-1)
                offset = 0
                for fsdp_param in param_group.fsdp_params:
                    if not fsdp_param.sharded_param.requires_grad:
                        continue
                    param = fsdp_param.sharded_param
                    # Use local tensor numel (sharded), not DTensor global numel
                    local = (
                        param._local_tensor
                        if isinstance(param, DTensor)
                        else param
                    )
                    local_numel = local.numel()
                    local_grad = grad_flat[offset : offset + local_numel].view(
                        local.shape
                    )
                    if isinstance(param, DTensor):
                        param.grad = DTensor.from_local(
                            local_grad,
                            device_mesh=param.device_mesh,
                            placements=param.placements,
                            run_check=False,
                        )
                    else:
                        param.grad = local_grad
                    offset += local_numel
                    restored += 1

        logger.info(
            f"[RmpGradAlloc] Restored {restored} param gradients from RMP"
        )
        return restored

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
                # Stage warmup runs backward under FakeTensorMode. Reconstructing
                # a CUDA IPC tensor calls set_(cuda_storage) on a meta tensor,
                # which fake dispatch rejects as a device mismatch. Also we
                # must not register anything on the RMP server during warmup.
                if detect_fake_mode() is not None:
                    return torch.empty(*size, dtype=dtype, device=device)
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
                logger.debug(
                    f"[RmpGradAlloc] Allocated RS output buffer '{pg_state.key}': "
                    f"numel={tensor.numel()}, dtype={dtype}, device={device}"
                )
                return tensor

        return allocate
