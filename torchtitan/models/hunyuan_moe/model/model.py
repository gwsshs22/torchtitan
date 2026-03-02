# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hunyuan-MoE model implementation for torchtitan.

Architecture: Llama-style GQA attention with QK norm + MoE FFN on every layer.
Uses rotate_half RoPE with dynamic NTK alpha scaling.
Reference: https://huggingface.co/tencent/Hunyuan-A13B-Pretrain
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

from .args import HunyuanMoEModelArgs


# --------------------------------------------------------------------------
# RoPE (rotate_half style, same as qwen3)
# --------------------------------------------------------------------------


def precompute_rope_cache(
    dim: int, max_seq_len: int, base: float = 10000.0
) -> torch.Tensor:
    """Precompute RoPE cos/sin cache.

    Returns tensor of shape [max_seq_len, dim * 2] where the first dim values
    are cosines and the last dim values are sines.
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)
    idx_theta = torch.outer(t, freqs).float()
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_for_broadcast(
    rope_cache: torch.Tensor, x: torch.Tensor, positions: torch.Tensor | None = None
) -> torch.Tensor:
    """Reshape RoPE cache for broadcasting with target tensor x.

    Args:
        rope_cache: Shape [max_seq_len, head_dim * 2].
        x: Target tensor of shape [bz, seq_len, n_heads, head_dim].
        positions: Position indices, shape (1, seq_len) or (bz, seq_len).
    """
    ndim = x.ndim
    assert ndim > 1
    bz, seqlen, _, head_dim = x.shape
    if positions is None:
        rope_cache = rope_cache[0:seqlen]
        assert rope_cache.shape == (seqlen, head_dim * 2)
        shape = [-1, seqlen, 1, head_dim * 2]
        return rope_cache.view(*shape)
    elif positions.size(0) == 1:
        assert positions.shape == (1, seqlen)
        rope_cache = rope_cache[positions.squeeze(0)]
        assert rope_cache.shape == (seqlen, head_dim * 2)
        shape = [-1, seqlen, 1, head_dim * 2]
        return rope_cache.view(*shape)
    else:
        assert positions.shape == (bz, seqlen)
        rope_cache_expanded = rope_cache[None, :, None, :].expand(bz, -1, -1, -1)
        rope_cache = torch.gather(
            rope_cache_expanded,
            dim=1,
            index=positions.view(bz, seqlen, 1, 1).expand(bz, seqlen, 1, head_dim * 2),
        )
        assert rope_cache.shape == (bz, seqlen, 1, head_dim * 2)
        return rope_cache


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings using rotate_half convention."""
    head_dim = xq.shape[-1]
    rope_cache = reshape_for_broadcast(rope_cache, xq, positions)
    cos = rope_cache[..., :head_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope_cache[..., head_dim:].to(dtype=xq.dtype, device=xq.device)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
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
# Attention (GQA with post-RoPE QK norm)
# --------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head GQA attention with QK norm applied after RoPE."""

    q_norm: nn.RMSNorm | None
    k_norm: nn.RMSNorm | None

    def __init__(self, model_args: HunyuanMoEModelArgs):
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

        # QK norm (applied after RoPE, unlike qwen3 which applies before)
        if model_args.qk_norm:
            self.q_norm = nn.RMSNorm(
                self.head_dim, eps=model_args.norm_eps, elementwise_affine=True
            )
            self.k_norm = nn.RMSNorm(
                self.head_dim, eps=model_args.norm_eps, elementwise_affine=True
            )
        else:
            self.q_norm = None
            self.k_norm = None

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
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 to infer local heads (TP may shard them)
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Apply RoPE first
        xq, xk = apply_rotary_emb(xq, xk, rope_cache=rope_cache, positions=positions)

        # Apply QK norm after RoPE
        if self.q_norm:  # pyrefly: ignore[invalid-argument]
            xq = self.q_norm(xq)
        if self.k_norm:  # pyrefly: ignore[invalid-argument]
            xk = self.k_norm(xk)

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
    """Transformer block with GQA attention (post-RoPE QK norm) and MoE FFN."""

    def __init__(self, layer_id: int, model_args: HunyuanMoEModelArgs):
        super().__init__()
        self.attention = Attention(model_args)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        # All layers are MoE in Hunyuan-MoE
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
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        x = x + self.attention(
            self.attention_norm(x), rope_cache, attention_masks, positions
        )
        x = x + self.moe(self.ffn_norm(x))
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.moe.init_weights(self.weight_init_std, buffer_device)


# --------------------------------------------------------------------------
# HunyuanMoEModel
# --------------------------------------------------------------------------


class HunyuanMoEModel(ModelProtocol):
    """Hunyuan-MoE Transformer model with GQA attention (post-RoPE QK norm) and MoE FFN."""

    def __init__(self, model_args: HunyuanMoEModelArgs):
        super().__init__(model_args)
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.register_buffer(
            "rope_cache", self._precompute_rope_cache(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        # NOTE: Weight tying (lm_head shares weight with tok_embeddings) is NOT
        # done in the constructor because it conflicts with TP parallelization:
        # tok_embeddings uses RowwiseParallel (Shard(1)) while output uses
        # ColwiseParallel (Shard(0)). Instead, weight tying is handled by the
        # state_dict_adapter during checkpoint save/load.

    def _precompute_rope_cache(self) -> torch.Tensor:
        return precompute_rope_cache(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.effective_rope_theta,
        )

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        buffer_device = buffer_device or self.rope_cache.device
        with torch.device(buffer_device):
            if self.rope_cache is not None:
                self.rope_cache.copy_(self._precompute_rope_cache())
            else:
                self.rope_cache = self._precompute_rope_cache()
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
            h = layer(h, self.rope_cache, attention_masks, positions)

        # pyrefly: ignore[not-callable, invalid-argument]
        h = self.norm(h) if self.norm else h
        # pyrefly: ignore[not-callable, invalid-argument]
        output = self.output(h) if self.output else h
        return output
