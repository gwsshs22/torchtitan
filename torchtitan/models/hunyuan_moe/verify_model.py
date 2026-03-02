#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Verification script to compare Hunyuan-MoE model outputs between
HuggingFace and torchtitan implementations.

Usage:
    python -m torchtitan.models.hunyuan_moe.verify_model
"""

import torch

from torchtitan.models.moe import MoEArgs

from .model.args import HunyuanMoEModelArgs
from .model.model import HunyuanMoEModel
from .model.state_dict_adapter import HunyuanMoEStateDictAdapter


def build_small_hf_config():
    """Build a small HF config for verification."""
    from transformers import HunYuanMoEV1Config

    return HunYuanMoEV1Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=4,
        moe_topk=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        rope_parameters=None,  # default rope, theta=10000.0
    )


def build_matching_tt_args():
    """Build matching torchtitan args for the small config."""
    return HunyuanMoEModelArgs(
        vocab_size=256,
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        moe_inter_dim=64,
        max_seq_len=512,
        rope_theta=10000.0,
        rope_alpha=1.0,  # no alpha scaling for verification
        norm_eps=1e-5,
        qk_norm=True,
        enable_weight_tying=True,
        depth_init=False,
        moe_args=MoEArgs(
            num_experts=4,
            num_shared_experts=1,
            top_k=2,
            score_func="softmax",
            route_norm=True,
            score_before_experts=False,
            use_grouped_mm=False,  # Use for-loop for deterministic comparison
            load_balance_coeff=None,  # Disable load balancing for comparison
        ),
    )


def transfer_weights_hf_to_tt(hf_model, tt_model, tt_args):
    """Transfer weights from HF model to torchtitan model using state_dict_adapter."""
    adapter = HunyuanMoEStateDictAdapter(tt_args, None)

    # Get HF state dict
    hf_state_dict = hf_model.state_dict()

    # Convert HF -> torchtitan format
    tt_state_dict = adapter.from_hf(hf_state_dict)

    # Load into torchtitan model
    missing, unexpected = tt_model.load_state_dict(tt_state_dict, strict=False)

    # Filter out expected missing keys (buffers and MoE internals)
    expected_missing = {"rope_cache"}
    real_missing = [k for k in missing if not any(e in k for e in expected_missing)]
    real_missing = [
        k
        for k in real_missing
        if "expert_bias" not in k and "tokens_per_expert" not in k
    ]

    if real_missing:
        print(f"WARNING: Missing keys after weight transfer: {real_missing}")
    if unexpected:
        print(f"WARNING: Unexpected keys: {unexpected}")

    return len(real_missing) == 0


def compare_outputs(hf_model, tt_model, seq_len=32, batch_size=2, vocab_size=256):
    """Compare forward pass outputs between HF and TT models."""
    device = next(hf_model.parameters()).device

    # Create random input tokens
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # HF forward pass
    with torch.no_grad():
        hf_output = hf_model(input_ids=input_ids)
        hf_logits = hf_output.logits

    # TT forward pass
    with torch.no_grad():
        tt_logits = tt_model(input_ids)

    # Compare
    max_diff = (hf_logits - tt_logits).abs().max().item()
    mean_diff = (hf_logits - tt_logits).abs().mean().item()
    is_close = torch.allclose(hf_logits, tt_logits, atol=1e-4, rtol=1e-4)

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "is_close": is_close,
        "hf_logits_shape": tuple(hf_logits.shape),
        "tt_logits_shape": tuple(tt_logits.shape),
    }


def main():
    print("=" * 60)
    print("Hunyuan-MoE: HuggingFace vs Torchtitan Verification")
    print("=" * 60)

    # CUDA required: torch.histc in MoE router doesn't support Long on CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available. torch.histc may fail on CPU.")
    dtype = torch.float32

    # Build configs
    print("\n[1] Building configs...")
    hf_config = build_small_hf_config()
    tt_args = build_matching_tt_args()
    print(
        f"    HF config: {hf_config.num_hidden_layers} layers, "
        f"{hf_config.hidden_size} dim, {hf_config.num_experts} experts"
    )
    print(
        f"    TT args: {tt_args.n_layers} layers, "
        f"{tt_args.dim} dim, {tt_args.moe_args.num_experts} experts"
    )

    # Build models
    print("\n[2] Building models...")
    from transformers import HunYuanMoEV1ForCausalLM

    hf_model = HunYuanMoEV1ForCausalLM(hf_config).to(device=device, dtype=dtype)
    hf_model.eval()

    tt_model = HunyuanMoEModel(tt_args).to(device=device, dtype=dtype)
    tt_model.eval()

    hf_params = sum(p.numel() for p in hf_model.parameters())
    tt_params = sum(p.numel() for p in tt_model.parameters())
    print(f"    HF params: {hf_params:,}")
    print(f"    TT params: {tt_params:,}")

    # Transfer weights
    print("\n[3] Transferring weights HF -> TT...")
    success = transfer_weights_hf_to_tt(hf_model, tt_model, tt_args)
    print(f"    Weight transfer: {'SUCCESS' if success else 'FAILED'}")

    # Compare outputs
    print("\n[4] Comparing forward pass outputs...")
    results = compare_outputs(
        hf_model, tt_model, seq_len=32, batch_size=2, vocab_size=tt_args.vocab_size
    )
    print(f"    HF logits shape: {results['hf_logits_shape']}")
    print(f"    TT logits shape: {results['tt_logits_shape']}")
    print(f"    Max absolute diff: {results['max_diff']:.6e}")
    print(f"    Mean absolute diff: {results['mean_diff']:.6e}")
    print(f"    torch.allclose (atol=1e-4): {results['is_close']}")

    # Test reverse direction: TT -> HF state dict
    print("\n[5] Testing reverse conversion (TT -> HF)...")
    adapter = HunyuanMoEStateDictAdapter(tt_args, None)
    tt_sd = tt_model.state_dict()
    hf_sd_converted = adapter.to_hf(tt_sd)
    expected_hf_keys = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.query_layernorm.weight",
        "model.layers.0.self_attn.key_layernorm.weight",
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.mlp.gate.wg.weight",
        "model.layers.0.mlp.shared_mlp.gate_proj.weight",
        "lm_head.weight",
    ]
    all_present = all(k in hf_sd_converted for k in expected_hf_keys)
    print(f"    Expected HF keys present: {all_present}")
    if not all_present:
        missing_keys = [k for k in expected_hf_keys if k not in hf_sd_converted]
        print(f"    Missing: {missing_keys}")

    # Verify gate_up_proj shape
    gup = hf_sd_converted.get("model.layers.0.mlp.experts.gate_up_proj")
    if gup is not None:
        expected_shape = (
            tt_args.moe_args.num_experts,
            2 * tt_args.moe_inter_dim,
            tt_args.dim,
        )
        shape_ok = tuple(gup.shape) == expected_shape
        print(
            f"    gate_up_proj shape: {tuple(gup.shape)} "
            f"(expected {expected_shape}): {'OK' if shape_ok else 'MISMATCH'}"
        )

    # Summary
    print("\n" + "=" * 60)
    if results["is_close"] and success:
        print("VERIFICATION PASSED")
    else:
        print("VERIFICATION FAILED")
        if not results["is_close"]:
            print(f"  Output mismatch: max_diff={results['max_diff']:.6e}")
        if not success:
            print("  Weight transfer had issues")
    print("=" * 60)

    return results["is_close"] and success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
