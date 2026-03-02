# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
State dict adapter for Solar-Open model.

Two conversions are needed:

1. Expert weights: HF stores fused 3D tensors, torchtitan stores separate:
   - HF experts.gate_up_proj: [E, 2*I, H] -> TT experts.w1 [E, I, H] + w3 [E, I, H]
   - HF experts.down_proj: [E, H, I] -> TT experts.w2 [E, H, I] (direct)

2. Q/K weight RoPE permutation: HF uses rotate_half pairing (i, i+d/2),
   torchtitan uses complex-exponential pairing (2k, 2k+1). We permute the
   output dimensions of wq/wk weights to bridge this difference.
"""

import re
from collections import defaultdict
from typing import Any

import torch

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .args import SolarOpenModelArgs


def _build_rope_permutation(head_dim: int) -> torch.Tensor:
    """Build permutation index to convert rotate_half -> complex RoPE pairing.

    HF rotate_half pairs dimensions (i, i + d/2).
    TT complex pairs consecutive dimensions (2k, 2k+1).

    To make TT's complex RoPE give the same result as HF's rotate_half,
    we permute Q/K weight rows so that originally paired dimensions become
    consecutive.

    Returns:
        Permutation tensor of shape (head_dim,).
    """
    d2 = head_dim // 2
    perm = torch.zeros(head_dim, dtype=torch.long)
    for k in range(d2):
        perm[2 * k] = k
        perm[2 * k + 1] = k + d2
    return perm


def _permute_qk_weight_for_complex_rope(
    weight: torch.Tensor, head_dim: int, inverse: bool = False
) -> torch.Tensor:
    """Permute Q or K projection weight rows for RoPE convention change.

    Weight shape: [out_features, in_features] where out_features = n_heads * head_dim.
    We permute within each head's output dimensions.

    Args:
        weight: Q or K weight matrix of shape [n_heads * head_dim, dim].
        head_dim: Dimension per head.
        inverse: If True, convert TT -> HF (inverse permutation).
    """
    out_features, in_features = weight.shape
    n_heads = out_features // head_dim
    perm = _build_rope_permutation(head_dim)
    if inverse:
        inv_perm = torch.zeros_like(perm)
        inv_perm[perm] = torch.arange(head_dim)
        perm = inv_perm

    # Reshape to [n_heads, head_dim, in_features], permute, reshape back
    w = weight.view(n_heads, head_dim, in_features)
    w = w[:, perm, :]
    return w.view(out_features, in_features)


class SolarOpenStateDictAdapter(StateDictAdapter):
    def __init__(self, model_args: SolarOpenModelArgs, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)
        self.model_args = model_args
        self.hf_assets_path = hf_assets_path

        # Direct mappings: HF key -> torchtitan key
        # Keys with {} are layer-indexed patterns
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            # Norms
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE router
            "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            # MoE experts - down_proj maps directly (same shape [E, H, I])
            "model.layers.{}.mlp.experts.down_proj": "layers.{}.moe.experts.w2",
            # Shared expert
            "model.layers.{}.mlp.shared_experts.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
            "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
            "model.layers.{}.mlp.shared_experts.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
            # Final norm and output
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
            # Skip rotary_emb (computed, not stored)
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            # Skip e_score_correction_bias (buffer, trained separately)
            "model.layers.{}.mlp.gate.e_score_correction_bias": None,
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict = {}
        head_dim = self.model_args.head_dim

        # TT keys that need inverse RoPE permutation
        _qk_tt_keys = {
            "layers.{}.attention.wq.weight",
            "layers.{}.attention.wk.weight",
        }

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
                # Inverse-permute Q/K weights back to HF convention
                if abstract_key in _qk_tt_keys:
                    value = _permute_qk_weight_for_complex_rope(
                        value, head_dim, inverse=True
                    )
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

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}
        head_dim = self.model_args.head_dim

        # Keys that need RoPE permutation (Q and K projection weights)
        _qk_keys = {
            "model.layers.{}.self_attn.q_proj.weight",
            "model.layers.{}.self_attn.k_proj.weight",
        }

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
                # Permute Q/K weights for RoPE convention change
                if abstract_key in _qk_keys:
                    value = _permute_qk_weight_for_complex_rope(value, head_dim)
                state_dict[new_key] = value

        return state_dict
