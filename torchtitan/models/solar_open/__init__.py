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

from .infra.parallelize import parallelize_solar_open
from .model.args import SolarOpenModelArgs
from .model.model import SolarOpenModel
from .model.state_dict_adapter import SolarOpenStateDictAdapter

__all__ = [
    "parallelize_solar_open",
    "SolarOpenModelArgs",
    "SolarOpenModel",
    "solar_open_args",
]


solar_open_args = {
    # Small debug model for testing
    "debugmodel": SolarOpenModelArgs(
        vocab_size=2048,
        max_seq_len=4096,
        head_dim=128,
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        moe_inter_dim=512,
        rope_theta=1000000.0,
        moe_args=MoEArgs(
            num_experts=16,
            num_shared_experts=1,
            top_k=4,
            score_func="sigmoid",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=True,
        ),
    ),
    # Solar-Open 100B configuration
    "100B": SolarOpenModelArgs(
        vocab_size=196608,
        max_seq_len=4096,
        head_dim=128,
        dim=4096,
        n_layers=48,
        n_heads=64,
        n_kv_heads=8,
        moe_inter_dim=1280,
        rope_theta=1000000.0,
        moe_args=MoEArgs(
            num_experts=128,
            num_shared_experts=1,
            top_k=8,
            score_func="sigmoid",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=True,
        ),
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=SolarOpenModel,
        model_args=solar_open_args,
        parallelize_fn=parallelize_solar_open,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=SolarOpenStateDictAdapter,
    )
