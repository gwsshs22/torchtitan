"""Operator discovery for MoEvement sparse checkpointing.

An operator is the unit of sparse-checkpoint scheduling (see
docs/moevement_port_plan.md §3.2 in the leto repo): per MoE layer, one op per
EP-local expert (a dim-0 slice of the grouped expert weights' local shards)
and one gate op (router params + the fp32-consumed expert_bias buffer); per
model part, one aggregate non_expert op holding every remaining trainable
parameter (shared_experts, norms, attention, embeddings, output head). Names
carry the global virtual-stage id, so interleaved-PP parts never collide.

Sizes are always taken from the EP/FSDP-local shard (DTensor.to_local()):
capture and restore happen on the same topology, so the local shard is the
snapshot unit. Expert modules are not required to be the base GroupedExperts
(gpt-oss uses GptOssGroupedExperts with mlp{1,2}_{weight,bias}): every direct
parameter of moe.experts is treated as expert-indexed on dim 0.
"""

import enum
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from torchtitan.components.moevement.scheduler import SchedulableOp
from torchtitan.models.moe.moe import MoE

# Active ops snapshot the fp32 master shard (4 B) plus the Adam
# exp_avg/exp_avg_sq shards (8 B); frozen ops snapshot a bf16 cast of the
# fp32 shard (2 B). fp32-consumed buffers (expert_bias) are captured at full
# fp32 in BOTH slots: the forward reads them raw, so a bf16 round-trip would
# perturb routing top-k and break bit-identical replay.
_ACTIVE_BYTES_PER_PARAM_ELEM = 12
_FROZEN_BYTES_PER_PARAM_ELEM = 2
_BUFFER_BYTES_PER_ELEM = 4


class OperatorKind(enum.Enum):
    EXPERT = "expert"
    GATE = "gate"
    NON_EXPERT = "non_expert"


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _entry_numel(param: nn.Parameter, expert_idx: int | None) -> int:
    local = _local(param)
    if expert_idx is None:
        return local.numel()
    # One expert's dim-0 slice of the EP-local shard.
    return local.numel() // local.shape[0]


@dataclass
class Operator:
    name: str
    kind: OperatorKind
    stage: int  # global virtual-stage id
    # (fqn, param, expert_idx): expert_idx selects dim 0 of the EP-local
    # shard of grouped expert weights; None means the whole local shard.
    param_entries: list[tuple[str, nn.Parameter, int | None]]
    # fp32-consumed persistent buffers (expert_bias): full fp32 in both slots.
    buffer_entries: list[tuple[str, torch.Tensor]] = field(default_factory=list)
    active_bytes: int = field(init=False)
    frozen_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        param_numel = sum(
            _entry_numel(param, expert_idx)
            for _, param, expert_idx in self.param_entries
        )
        buffer_numel = sum(
            _local(tensor).numel() for _, tensor in self.buffer_entries
        )
        self.active_bytes = (
            _ACTIVE_BYTES_PER_PARAM_ELEM * param_numel
            + _BUFFER_BYTES_PER_ELEM * buffer_numel
        )
        self.frozen_bytes = (
            _FROZEN_BYTES_PER_PARAM_ELEM * param_numel
            + _BUFFER_BYTES_PER_ELEM * buffer_numel
        )

    def schedulable(self) -> SchedulableOp:
        return SchedulableOp(
            name=self.name,
            active_bytes=self.active_bytes,
            frozen_bytes=self.frozen_bytes,
            is_expert=self.kind is OperatorKind.EXPERT,
        )


def _layer_label(moe_fqn: str) -> str:
    # "layers.3.moe" -> "3" (the block's key in the layers ModuleDict); fall
    # back to the sanitized module path for MoE layers outside a numbered
    # container. Identical module structure => identical labels across ranks.
    for component in reversed(moe_fqn.split(".")):
        if component.isdigit():
            return component
    return moe_fqn.replace(".", "_") or "moe"


def _discover_part_operators(model_part: nn.Module, stage: int) -> list[Operator]:
    operators: list[Operator] = []
    excluded_prefixes: list[str] = []
    for moe_fqn, moe in model_part.named_modules():
        if not isinstance(moe, MoE):
            continue
        prefix = f"{moe_fqn}." if moe_fqn else ""
        layer = _layer_label(moe_fqn)

        # Every direct parameter of the experts module is expert-indexed on
        # dim 0 (w1/w2/w3 for GroupedExperts, mlp*_{weight,bias} for
        # gpt-oss). Parameters of nested submodules are NOT claimed here:
        # they have no defined per-expert slicing and must fail the coverage
        # check below instead of being captured wrong.
        weight_entries = [
            (f"{prefix}experts.{name}", param)
            for name, param in moe.experts.named_parameters(recurse=False)
            if param.requires_grad
        ]
        assert weight_entries, (
            f"MoE layer {moe_fqn or '<root>'} has no trainable expert weights"
        )
        local_counts = [_local(param).shape[0] for _, param in weight_entries]
        num_local_experts = local_counts[0]
        assert all(count == num_local_experts for count in local_counts), (
            f"expert weights of {moe_fqn or '<root>'} disagree on the local "
            f"expert count (dim 0): "
            f"{[(fqn, count) for (fqn, _), count in zip(weight_entries, local_counts)]}"
        )
        for expert_idx in range(num_local_experts):
            operators.append(
                Operator(
                    name=f"stage{stage}_layer{layer}_expert{expert_idx}",
                    kind=OperatorKind.EXPERT,
                    stage=stage,
                    param_entries=[
                        (fqn, param, expert_idx) for fqn, param in weight_entries
                    ],
                )
            )

        gate_entries: list[tuple[str, nn.Parameter, int | None]] = [
            (f"{prefix}router.{name}", param, None)
            for name, param in moe.router.named_parameters()
            if param.requires_grad
        ]
        buffer_entries: list[tuple[str, torch.Tensor]] = []
        if moe.expert_bias is not None:
            buffer_entries.append((f"{prefix}expert_bias", moe.expert_bias))
        operators.append(
            Operator(
                name=f"stage{stage}_layer{layer}_gate",
                kind=OperatorKind.GATE,
                stage=stage,
                param_entries=gate_entries,
                buffer_entries=buffer_entries,
            )
        )
        excluded_prefixes.append(f"{prefix}experts.")
        excluded_prefixes.append(f"{prefix}router.")

    # Membership is subtree-based rather than "whatever expert/gate ops did
    # not claim": an unclaimed parameter under experts./router. must trip the
    # coverage invariant, not be silently swept into non_expert.
    non_expert_entries: list[tuple[str, nn.Parameter, int | None]] = [
        (fqn, param, None)
        for fqn, param in model_part.named_parameters()
        if param.requires_grad
        and not any(fqn.startswith(prefix) for prefix in excluded_prefixes)
    ]
    operators.append(
        Operator(
            name=f"stage{stage}_non_expert",
            kind=OperatorKind.NON_EXPERT,
            stage=stage,
            param_entries=non_expert_entries,
        )
    )
    return operators


def _check_coverage(
    model_parts: list[nn.Module], operators: list[Operator]
) -> None:
    """Every trainable parameter of every model part must appear in exactly
    one operator's param_entries; for expert-sliced params, the per-expert
    slices together must cover dim 0 of the local shard exactly once."""
    expected_fqns: dict[int, str] = {}
    params_by_id: dict[int, nn.Parameter] = {}
    for part_idx, model_part in enumerate(model_parts):
        for fqn, param in model_part.named_parameters():
            if param.requires_grad:
                expected_fqns[id(param)] = f"model_parts[{part_idx}].{fqn}"
                params_by_id[id(param)] = param

    full_claims: dict[int, list[str]] = {}
    slice_claims: dict[int, list[int]] = {}
    for op in operators:
        for fqn, param, expert_idx in op.param_entries:
            key = id(param)
            assert key in expected_fqns, (
                f"operator {op.name} claims a parameter that is not a "
                f"trainable parameter of any model part: {fqn}"
            )
            if expert_idx is None:
                full_claims.setdefault(key, []).append(op.name)
            else:
                slice_claims.setdefault(key, []).append(expert_idx)

    for key, fqn in expected_fqns.items():
        claimants = full_claims.get(key, [])
        slices = slice_claims.get(key)
        if slices is not None:
            assert not claimants, (
                f"parameter {fqn} is claimed both whole (by {claimants}) and "
                f"as per-expert slices"
            )
            num_local_experts = _local(params_by_id[key]).shape[0]
            assert sorted(slices) == list(range(num_local_experts)), (
                f"per-expert slices of {fqn} do not cover dim 0 exactly "
                f"once: got indices {sorted(slices)}, expected "
                f"0..{num_local_experts - 1}"
            )
        else:
            assert len(claimants) == 1, (
                f"parameter {fqn} appears in {len(claimants)} operators "
                f"({claimants}); every trainable parameter must appear in "
                f"exactly one"
            )


def discover_operators(
    model_parts: list[nn.Module], stage_ids: list[int]
) -> list[Operator]:
    """Enumerate operators over model_parts, one (part, global virtual-stage
    id) pair per entry, in local-chunk order. Iteration follows
    named_modules/named_parameters registration order only, so the operator
    list and names are identical across ranks with identical module
    structure."""
    assert len(model_parts) == len(stage_ids), (
        f"got {len(model_parts)} model parts but {len(stage_ids)} stage ids"
    )
    operators: list[Operator] = []
    for model_part, stage in zip(model_parts, stage_ids):
        operators.extend(_discover_part_operators(model_part, stage))
    _check_coverage(model_parts, operators)
    names = [op.name for op in operators]
    duplicates = [name for idx, name in enumerate(names) if name in names[:idx]]
    assert not duplicates, f"duplicate operator names: {duplicates}"
    return operators
