"""Resilient optimizer (shadow): full GPU shadow buffer for all state.

Before optimizer.step(), copies all params + exp_avg + exp_avg_sq to
shadow tensors via cudaMemcpyAsync (one per tensor, DtoD).

This is the fastest approach but requires 2x GPU memory for the
optimizer state (params + exp_avg + exp_avg_sq all duplicated).

Usage:
    resilient_opt = ResilientOptimizerGpuSnapshot(optimizers, rmp_client, device)
    # In training loop:
    resilient_opt.step()
"""

import torch
from torch.distributed._tensor import DTensor

from leto.rmp.client import RmpClient, TensorSpec
from torchtitan.tools.logging import logger


def _get_local(tensor):
    if isinstance(tensor, DTensor):
        return tensor._local_tensor
    return tensor


class ResilientOptimizerGpuSnapshot:
    """Full GPU shadow buffer — per-tensor memcpy snapshot before each step."""

    def __init__(
        self,
        optimizers,
        rmp_client: RmpClient,
        device: torch.device,
    ):
        self._optimizers = optimizers

        # Collect (original, shadow) pairs
        originals: list[torch.Tensor] = []
        shadow_specs: list[TensorSpec] = []
        device_idx = device.index if hasattr(device, "index") else 0
        idx = 0

        for optimizer in optimizers:
            for param, state in optimizer.state.items():
                for prefix, tensor in [
                    ("param", param),
                    ("exp_avg", state["exp_avg"]),
                    ("exp_avg_sq", state["exp_avg_sq"]),
                ]:
                    local = _get_local(tensor)
                    originals.append(local)
                    shadow_specs.append(
                        TensorSpec(
                            name=f"shadow/{prefix}/{idx}",
                            shape=tuple(local.shape),
                            dtype=local.dtype,
                            device=device_idx,
                        )
                    )
                idx += 1

        shadow_tensors, _ = rmp_client.get_or_allocate_tensors(shadow_specs)

        # Build paired list: (original_flat_uint8, shadow_flat_uint8)
        # Contiguous flat views ensure .copy_() dispatches as cudaMemcpyAsync.
        self._pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, spec in enumerate(shadow_specs):
            orig = originals[i].contiguous().view(torch.uint8).reshape(-1)
            shad = shadow_tensors[spec.name].contiguous().view(torch.uint8).reshape(-1)
            self._pairs.append((orig, shad))

        total_bytes = sum(s.numel() for _, s in self._pairs)
        logger.info(
            f"[ResilientOptShadow] {idx} params, "
            f"shadow: {total_bytes / (1024**2):.1f} MB GPU"
        )

    def step(self):
        """Snapshot all state → optimizer step."""
        self._snapshot()
        self._optimizers.step()

    def restore_and_replay(self):
        """Restore all state from shadow and replay the step."""
        self._restore()
        self._optimizers.step()

    @torch.no_grad()
    def _snapshot(self):
        """Per-tensor cudaMemcpyAsync DtoD."""
        for orig, shad in self._pairs:
            shad.copy_(orig, non_blocking=True)

    @torch.no_grad()
    def _restore(self):
        """Per-tensor cudaMemcpyAsync DtoD."""
        for orig, shad in self._pairs:
            orig.copy_(shad, non_blocking=True)
