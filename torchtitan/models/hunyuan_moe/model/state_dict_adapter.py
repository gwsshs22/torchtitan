# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
State dict adapter for Hunyuan-MoE model.

Conversion needed:

1. Expert weights: HF stores fused 3D tensors, torchtitan stores separate:
   - HF experts.gate_up_proj: [E, 2*I, H] -> TT experts.w1 [E, I, H] + w3 [E, I, H]
   - HF experts.down_proj: [E, H, I] -> TT experts.w2 [E, H, I] (direct)

2. No RoPE weight permutation needed (both HF and TT use rotate_half).

3. Tied embeddings: HF may not store lm_head.weight separately.
"""

import re
from collections import defaultdict
from typing import Any

import torch

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .args import HunyuanMoEModelArgs


class HunyuanMoEStateDictAdapter(StateDictAdapter):
    def __init__(self, model_args: HunyuanMoEModelArgs, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)
        self.model_args = model_args
        self.hf_assets_path = hf_assets_path

        # Direct mappings: HF key -> torchtitan key
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            # QK norm
            "model.layers.{}.self_attn.query_layernorm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.key_layernorm.weight": "layers.{}.attention.k_norm.weight",
            # Norms
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE router (HF uses mlp.gate.wg, TT uses moe.router.gate)
            "model.layers.{}.mlp.gate.wg.weight": "layers.{}.moe.router.gate.weight",
            # MoE experts - down_proj maps directly (same shape [E, H, I])
            "model.layers.{}.mlp.experts.down_proj": "layers.{}.moe.experts.w2",
            # Shared expert (HF uses shared_mlp, TT uses moe.shared_experts)
            "model.layers.{}.mlp.shared_mlp.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
            "model.layers.{}.mlp.shared_mlp.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
            "model.layers.{}.mlp.shared_mlp.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
            # Final norm and output
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
            # Skip rotary_emb (computed, not stored)
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict = {}

        # Collect w1/w3 pairs for combining into gate_up_proj
        to_combine: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

        for key, value in state_dict.items():
            if "layers" in key:
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
            else:
                layer_num = None
                abstract_key = key

            if abstract_key in to_hf_map:
                new_key = to_hf_map[abstract_key]
                if new_key is None:
                    continue
                if layer_num:
                    new_key = new_key.format(layer_num)
                hf_state_dict[new_key] = value
            elif abstract_key in [
                "layers.{}.moe.experts.w1",
                "layers.{}.moe.experts.w3",
            ]:
                # Collect w1 and w3 to merge into gate_up_proj
                hf_fqn = "model.layers.{}.mlp.experts.gate_up_proj".format(layer_num)
                to_combine[hf_fqn][abstract_key.format(layer_num)] = value

        # Merge w1 + w3 -> gate_up_proj [E, 2*I, H]
        for hf_fqn, tt_fqn_map in to_combine.items():
            # pyrefly: ignore [missing-attribute]
            layer_num = re.search(r"\d+", hf_fqn).group(0)
            w1 = tt_fqn_map["layers.{}.moe.experts.w1".format(layer_num)]
            w3 = tt_fqn_map["layers.{}.moe.experts.w3".format(layer_num)]
            # w1: [E, I, H], w3: [E, I, H] -> gate_up: [E, 2*I, H]
            hf_state_dict[hf_fqn] = torch.cat([w1, w3], dim=1)

        # Handle tied embeddings: if weight tying, lm_head shares embed_tokens
        if self.model_args.enable_weight_tying:
            if "lm_head.weight" not in hf_state_dict:
                hf_state_dict["lm_head.weight"] = hf_state_dict[
                    "model.embed_tokens.weight"
                ]

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}

        # Handle tied embeddings: if lm_head.weight missing, use embed_tokens
        if (
            "lm_head.weight" not in hf_state_dict
            and "model.embed_tokens.weight" in hf_state_dict
        ):
            hf_state_dict["lm_head.weight"] = hf_state_dict[
                "model.embed_tokens.weight"
            ]

        for key, value in hf_state_dict.items():
            if "layers" in key:
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
            else:
                layer_num = None
                abstract_key = key

            if abstract_key == "model.layers.{}.mlp.experts.gate_up_proj":
                # Split gate_up_proj [E, 2*I, H] -> w1 [E, I, H] + w3 [E, I, H]
                w1, w3 = value.chunk(2, dim=1)
                state_dict["layers.{}.moe.experts.w1".format(layer_num)] = w1
                state_dict["layers.{}.moe.experts.w3".format(layer_num)] = w3
            elif abstract_key in self.from_hf_map:
                new_key = self.from_hf_map[abstract_key]
                if new_key is None:
                    continue
                if layer_num:
                    new_key = new_key.format(layer_num)
                state_dict[new_key] = value

        return state_dict
