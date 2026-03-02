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
class SolarOpenModelArgs(BaseModelArgs):
    """Model arguments for Solar-Open (MoE language model).

    Solar-Open uses Llama-style GQA attention with MoE FFN on every layer.
    Note: n_heads * head_dim (e.g. 64*128=8192) can differ from dim (4096).
    """

    dim: int = 4096
    n_layers: int = 48
    n_heads: int = 64
    n_kv_heads: int = 8
    vocab_size: int = 196608
    head_dim: int = 128
    moe_inter_dim: int = 1280  # intermediate size per expert

    norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    max_seq_len: int = 4096
    depth_init: bool = True

    attn_type: str = "sdpa"
    attn_mask_type: str = "causal"

    # MoE params (all layers are MoE)
    moe_args: MoEArgs = field(default_factory=MoEArgs)

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
