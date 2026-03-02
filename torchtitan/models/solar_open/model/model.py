# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Solar-Open model implementation for torchtitan.

Architecture: Llama-style GQA attention + MoE FFN on every layer.
Reference: https://huggingface.co/upstage/Solar-Open-100B
"""

import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks, BlockMask

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.attention import (
    create_attention_mask,
    create_varlen_metadata_for_document,
    FlexAttentionWrapper,
    get_causal_mask_mod,
    get_document_mask_mod,
    ScaledDotProductAttentionWrapper,
    VarlenAttentionWrapper,
    VarlenMetadata,
)
from torchtitan.models.moe import MoE
from torchtitan.protocols.model import AttentionMasksType
from torchtitan.protocols.train_spec import ModelProtocol

from .args import SolarOpenModelArgs


# --------------------------------------------------------------------------
# RoPE (complex exponential style, compatible with DTensor/TP)
#
# Uses consecutive-pair convention: pairs (2k, 2k+1).
# HF uses rotate_half convention: pairs (i, i+d/2).
# The state_dict_adapter permutes Q/K weights to bridge the difference.
# --------------------------------------------------------------------------


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (cis).

    Args:
        dim: Dimension of the frequency tensor (head_dim).
        end: End index for precomputing frequencies (max_seq_len).
        theta: Scaling factor for frequency computation.

    Returns:
        Precomputed complex exponential values (complex64).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(
    freqs_cis: torch.Tensor, x: torch.Tensor, positions: torch.Tensor | None = None
) -> torch.Tensor:
    """Reshape frequency tensor for broadcasting with target tensor x."""
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    if positions is None:
        freqs_cis = freqs_cis[0:seqlen]
        assert freqs_cis.shape == (seqlen, x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)
    elif positions.size(0) == 1:
        assert positions.shape == (1, seqlen)
        freqs_cis = freqs_cis[positions.squeeze(0)]
        assert freqs_cis.shape == (seqlen, x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)
    else:
        assert positions.shape == (x.shape[0], seqlen)
        freqs_cis_expanded = freqs_cis[None, :, None, :].expand(x.shape[0], -1, -1, -1)
        freqs_cis = torch.gather(
            freqs_cis_expanded,
            dim=1,
            index=positions.view(x.shape[0], seqlen, 1, 1).expand(
                x.shape[0], seqlen, 1, freqs_cis_expanded.shape[-1]
            ),
        )
        return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_, positions)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# --------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head GQA attention module for Solar-Open.

    Note: n_heads * head_dim can differ from dim (e.g., 64*128=8192 != 4096).
    """

    def __init__(self, model_args: SolarOpenModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(
            model_args.dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            model_args.dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

        self.attn_type = model_args.attn_type
        match self.attn_type:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "varlen":
                # pyrefly: ignore [bad-assignment]
                self.inner_attention = VarlenAttentionWrapper()
            case "sdpa":
                # pyrefly: ignore [bad-assignment]
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 to infer local heads (TP may shard them)
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, positions=positions)

        # Repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        match self.attn_type:
            case "flex":
                assert isinstance(attention_masks, BlockMask), attention_masks
                output = self.inner_attention(xq, xk, xv, block_mask=attention_masks)
                output = output.transpose(1, 2).contiguous()
            case "varlen":
                assert isinstance(attention_masks, VarlenMetadata), attention_masks
                output = self.inner_attention(
                    xq, xk, xv, self.head_dim, attention_masks
                )
            case "sdpa":
                assert attention_masks is None
                output = self.inner_attention(xq, xk, xv)
                output = output.transpose(1, 2).contiguous()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

        output = output.view(bs, seqlen, -1)
        return self.wo(output)


# --------------------------------------------------------------------------
# TransformerBlock
# --------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Transformer block with GQA attention and MoE FFN."""

    def __init__(self, layer_id: int, model_args: SolarOpenModelArgs):
        super().__init__()
        self.attention = Attention(model_args)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        # All layers are MoE in Solar-Open
        self.moe_enabled = True
        self.moe = MoE(
            model_args.moe_args,
            dim=model_args.dim,
            hidden_dim=model_args.moe_inter_dim,
        )

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        x = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        x = x + self.moe(self.ffn_norm(x))
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.moe.init_weights(self.weight_init_std, buffer_device)


# --------------------------------------------------------------------------
# SolarOpenModel
# --------------------------------------------------------------------------


class SolarOpenModel(ModelProtocol):
    """Solar-Open Transformer model with GQA attention and MoE FFN."""

    def __init__(self, model_args: SolarOpenModelArgs):
        super().__init__(model_args)
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.register_buffer(
            "freqs_cis", self._precompute_freqs_cis(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            if self.freqs_cis is not None:
                self.freqs_cis.copy_(self._precompute_freqs_cis())
            else:
                self.freqs_cis = self._precompute_freqs_cis()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                # pyrefly: ignore [not-callable]
                layer.init_weights(buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _get_flex_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        mask_mods = [get_causal_mask_mod()]
        match self.model_args.attn_mask_type:
            case "causal":
                B = 1
            case "block_causal":
                B = input_batch.shape[0]
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.model_args.attn_mask_type}"
                )
        return create_attention_mask(
            and_masks(*mask_mods), B, None, input_batch.shape[1], input_batch.shape[1]
        )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        match self.model_args.attn_type:
            case "flex":
                return self._get_flex_attention_masks(
                    input_batch, tokenizer, extra_inputs
                )
            case "varlen":
                if self.model_args.attn_mask_type != "block_causal":
                    raise ValueError(
                        f"varlen attention is only supported with block_causal "
                        f"attention mask type, got {self.model_args.attn_mask_type}"
                    )
                return create_varlen_metadata_for_document(
                    input_batch, tokenizer.eos_id
                )
            case _:
                raise TypeError("Only varlen and flex attn masks are supported")

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        # pyrefly: ignore[not-callable, invalid-argument]
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis, attention_masks, positions)

        # pyrefly: ignore[not-callable, invalid-argument]
        h = self.norm(h) if self.norm else h
        # pyrefly: ignore[not-callable, invalid-argument]
        output = self.output(h) if self.output else h
        return output
