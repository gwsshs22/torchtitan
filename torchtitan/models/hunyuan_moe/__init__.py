# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_hunyuan_moe
from .model.args import HunyuanMoEModelArgs
from .model.model import HunyuanMoEModel
from .model.state_dict_adapter import HunyuanMoEStateDictAdapter

__all__ = [
    "parallelize_hunyuan_moe",
    "HunyuanMoEModelArgs",
    "HunyuanMoEModel",
    "hunyuan_moe_args",
]


hunyuan_moe_args = {
    # Small debug model for testing
    "debugmodel": HunyuanMoEModelArgs(
        vocab_size=2048,
        max_seq_len=4096,
        head_dim=64,
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        moe_inter_dim=256,
        rope_theta=10000.0,
        rope_alpha=1.0,  # no alpha scaling for debug
        qk_norm=True,
        enable_weight_tying=True,
        moe_args=MoEArgs(
            num_experts=16,
            num_shared_experts=1,
            top_k=4,
            score_func="softmax",
            route_norm=True,
            score_before_experts=False,
        ),
    ),
    # Hunyuan-A13B (80B total, 13B active)
    "A13B": HunyuanMoEModelArgs(
        vocab_size=128167,
        max_seq_len=4096,
        head_dim=128,
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        moe_inter_dim=3072,
        rope_theta=10000.0,
        rope_alpha=1000.0,
        qk_norm=True,
        enable_weight_tying=True,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            score_before_experts=False,
        ),
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=HunyuanMoEModel,
        model_args=hunyuan_moe_args,
        parallelize_fn=parallelize_hunyuan_moe,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=HunyuanMoEStateDictAdapter,
    )
