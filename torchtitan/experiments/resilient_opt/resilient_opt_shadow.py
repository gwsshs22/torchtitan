"""Resilient optimizer (shadow): full GPU shadow buffer for all state.

Before optimizer.step(), copies all params + exp_avg + exp_avg_sq to
a complete set of GPU shadow tensors.  If a fault corrupts the in-place
fused AdamW update, restores from the shadow and replays.

This is the fastest approach — single bulk copy, no chunking — but
requires 2x GPU memory for the optimizer state (params + exp_avg +
exp_avg_sq all duplicated).

Usage:
    resilient_opt = ResilientOptimizerShadow(optimizers, rmp_client, device)
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


class ResilientOptimizerShadow:
    """Full GPU shadow buffer — snapshot all state before each step."""

    def __init__(
        self,
        optimizers,
        rmp_client: RmpClient,
        device: torch.device,
    ):
        self._optimizers = optimizers

        # Collect all (param, exp_avg, exp_avg_sq)
        self._params: list[torch.Tensor] = []
        self._exp_avgs: list[torch.Tensor] = []
        self._exp_avg_sqs: list[torch.Tensor] = []

        shadow_specs: list[TensorSpec] = []
        device_idx = device.index if hasattr(device, "index") else 0
        idx = 0

        for optimizer in optimizers:
            for param, state in optimizer.state.items():
                self._params.append(param)
                self._exp_avgs.append(state["exp_avg"])
                self._exp_avg_sqs.append(state["exp_avg_sq"])

                for prefix, tensor in [
                    ("param", param),
                    ("exp_avg", state["exp_avg"]),
                    ("exp_avg_sq", state["exp_avg_sq"]),
                ]:
                    local = _get_local(tensor)
                    shadow_specs.append(
                        TensorSpec(
                            name=f"shadow/{prefix}/{idx}",
                            shape=tuple(local.shape),
                            dtype=local.dtype,
                            device=device_idx,
                        )
                    )
                idx += 1

        # Allocate shadow tensors (one-to-one with originals)
        shadow_tensors, _ = rmp_client.get_or_allocate_tensors(shadow_specs)

        self._shadow_params = [shadow_tensors[f"shadow/param/{i}"] for i in range(idx)]
        self._shadow_exp_avgs = [shadow_tensors[f"shadow/exp_avg/{i}"] for i in range(idx)]
        self._shadow_exp_avg_sqs = [shadow_tensors[f"shadow/exp_avg_sq/{i}"] for i in range(idx)]

        total_bytes = sum(t.numel() * t.element_size() for t in shadow_tensors.values())
        logger.info(
            f"[ResilientOptShadow] {idx} params, "
            f"shadow: {total_bytes / (1024**2):.1f} MB GPU"
        )

    def step(self):
        """Snapshot all state → optimizer step."""
        self._snapshot()
        torch.cuda.current_stream().synchronize()
        self._optimizers.step()
        torch.cuda.current_stream().synchronize()

    def restore_and_replay(self):
        """Restore all state from shadow and replay the step."""
        self._restore()
        torch.cuda.current_stream().synchronize()
        self._optimizers.step()
        torch.cuda.current_stream().synchronize()

    @torch.no_grad()
    def _snapshot(self):
        """Bulk GPU→GPU copy using foreach."""
        torch._foreach_copy_(
            self._shadow_params,
            [_get_local(p) for p in self._params],
        )
        torch._foreach_copy_(
            self._shadow_exp_avgs,
            [_get_local(t) for t in self._exp_avgs],
        )
        torch._foreach_copy_(
            self._shadow_exp_avg_sqs,
            [_get_local(t) for t in self._exp_avg_sqs],
        )

    @torch.no_grad()
    def _restore(self):
        """Bulk GPU→GPU copy from shadow back to originals."""
        torch._foreach_copy_(
            [_get_local(p) for p in self._params],
            self._shadow_params,
        )
        torch._foreach_copy_(
            [_get_local(t) for t in self._exp_avgs],
            self._shadow_exp_avgs,
        )
        torch._foreach_copy_(
            [_get_local(t) for t in self._exp_avg_sqs],
            self._shadow_exp_avg_sqs,
        )
