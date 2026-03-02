# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.models.moe import MoEArgs
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.model import BaseModelArgs
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


@dataclass
class HunyuanMoEModelArgs(BaseModelArgs):
    """Model arguments for Hunyuan-MoE (tencent/Hunyuan-A13B-Pretrain).

    Architecture: Llama-style GQA attention with QK norm + MoE FFN on every layer.
    Uses rotate_half RoPE with dynamic NTK alpha scaling.
    """

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 128167
    head_dim: int = 128
    moe_inter_dim: int = 3072  # intermediate size per expert

    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_alpha: float = 1000.0  # dynamic NTK alpha scaling factor
    max_seq_len: int = 4096
    depth_init: bool = True

    qk_norm: bool = True
    enable_weight_tying: bool = True

    attn_type: str = "sdpa"
    attn_mask_type: str = "causal"

    # MoE params (all layers are MoE)
    moe_args: MoEArgs = field(default_factory=MoEArgs)

    @property
    def effective_rope_theta(self) -> float:
        """Compute effective RoPE theta with dynamic NTK alpha scaling.

        effective_theta = rope_theta * alpha^(head_dim / (head_dim - 2))
        """
        if self.rope_alpha > 1.0:
            return self.rope_theta * self.rope_alpha ** (
                self.head_dim / (self.head_dim - 2)
            )
        return self.rope_theta

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if self.moe_args.use_grouped_mm and not has_cuda_capability(9, 0):
            logger.warning(
                "Failed to use grouped mm, which is only supported on SM90 or later",
            )
            self.moe_args.use_grouped_mm = False

        self.moe_args._debug_force_load_balance = (
            job_config.debug.moe_force_load_balance
        )

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, int]:
        return get_moe_model_nparams_and_flops(
            self, model, 2 * self.head_dim, seq_len
        )
