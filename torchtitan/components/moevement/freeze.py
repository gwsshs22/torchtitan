"""Frozen-operator wgrad skipping for MoEvement replay (paper §3.3).

During sparse->dense replay, an operator that has not yet been activated
must **not** be updated. The port's original mechanic (plan §3.6, §6-C7)
ran the full backward and dropped the frozen parameters' ``.grad`` to
``None`` in the optimizer-step pre-hook slot: numerically exact, zero FLOP
saving. The paper instead runs frozen operators **forward + input-gradient
only** and skips the weight-gradient entirely, which is where its ~33%
cheaper replay comes from (fwd:dgrad:wgrad ≈ 1:1:1).

This module implements that, with two mechanisms because our operators do
not all own whole tensors:

* **Whole-tensor operators** — ``non_expert`` (the dense backbone of a model
  part) and ``gate`` (router weights) own every element of their parameters,
  so ``param.requires_grad_(False)`` is enough. FSDP2 propagates the flag to
  the all-gathered parameter at each unshard
  (``FSDPParam.to_unsharded`` -> ``set_requires_grad_if_needed``), the
  autograd graph then omits the weight-gradient, and
  ``FSDPParamGroup.post_backward`` finds no gradients and returns after
  resharding — so the reduce-scatter is skipped too (a bonus saving).

* **Grouped expert tensors** — ``w1/w2/w3`` (or gpt-oss's
  ``mlp{1,2}_{weight,bias}``) hold every EP-local expert of a layer in ONE
  tensor sharded on dim 0, and "frozen" is a per-SLICE property. There is no
  autograd path that computes the wgrad for an arbitrary subset of dim-0
  slices of a ``torch._grouped_mm`` without either repacking the token rows
  (which the offsets cannot express: rows are sorted by expert and a
  cumulative offset vector cannot skip a middle group) or splitting into one
  call per contiguous run of active experts (unbounded graph variants). So
  the skip is applied at LAYER granularity: a layer's expert weights are
  handed to the grouped-mm detached only when EVERY EP-local expert of that
  layer is frozen. A partially-frozen layer keeps its full backward and
  falls back to grad masking, exactly as before. This is what keeps the
  compiled-variant count bounded — see ``BOUNDED VARIANTS`` below.

BOUNDED VARIANTS
----------------
The frozen set is expressed as a per-module boolean, never as a shape or a
count, so each compiled region sees at most **two** AOTAutograd variants:
"weights require grad" (the normal one, already compiled by warmup) and
"weights do not". Concretely:

* ``moe_module._run_experts_grouped_mm`` is a single ``torch.compile``d
  global shared by every MoE layer (llama4/infra/parallelize.py), so the
  detached call site adds exactly ONE extra graph for the whole model.
* the compiled per-submodule regions (attention, feed_forward, router, norms)
  guard on their parameters' ``requires_grad``, so each adds one extra
  variant. Since ``non_expert`` is one operator covering the whole dense
  backbone of a model part, those flip together — there is no combinatorial
  blow-up over subsets.

SAFETY RULES
------------
1. **A model part is never left with zero trainable parameters.** With PP,
   stage 0's inputs are token ids that do not require grad; if every
   parameter of the part were frozen, ``backward`` would have nothing to
   compute and would raise. When the frozen set would empty a part, that
   part's ``non_expert`` operator stays trainable (and is grad-masked as
   before). Costs nothing in the common case.
2. **FSDP2 mixed-precision init.** ``FSDPParamGroup._init_mp_dtypes`` runs
   once, at the first forward, and records ``None`` for the group's original
   / reduce dtype when the group has no trainable parameter at that moment.
   A replay armed by ``load()`` starts before that first forward, so the
   controller defers its first freeze until one full step has run in this
   process (``arm()`` -> first ``apply()`` is a no-op). One replayed
   iteration therefore pays the old full-backward cost; the rest are skipped.
3. **Gradient-norm clipping.** ``training.max_norm`` is applied with a total
   norm computed over the parameters that HAVE gradients. Skipping the
   frozen wgrads shrinks that set, which would change the clip coefficient
   and hence the surviving (activated) operators' updates. The replay
   therefore pins the total norm to the value captured at the original
   iteration; see ``MoevementCheckpointManager.replay_clip_total_norm``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan.components.moevement.operators import (
    Operator,
    OperatorKind,
    _layer_label,
)
from torchtitan.models.moe.moe import MoE
from torchtitan.tools.logging import logger


@dataclass
class FreezeStats:
    """What the last ``apply()`` actually skipped (for logs and tests)."""

    frozen_ops: int = 0
    frozen_params: int = 0
    frozen_param_numel: int = 0
    expert_modules_skipped: int = 0
    expert_modules_partial: int = 0
    parts_kept_trainable: int = 0
    applied: bool = False

    def summary(self) -> str:
        return (
            f"ops={self.frozen_ops} params={self.frozen_params} "
            f"numel={self.frozen_param_numel} "
            f"expert_modules(skip={self.expert_modules_skipped}, "
            f"partial={self.expert_modules_partial}) "
            f"parts_kept_trainable={self.parts_kept_trainable}"
        )


@dataclass
class _PartInfo:
    index: int
    module: nn.Module
    op_names: list[str] = field(default_factory=list)
    non_expert_op: str | None = None


class FrozenSkipController:
    """Applies / clears the §3.3 frozen-skip state over a replay.

    Lifecycle: ``arm()`` when a replay is armed, ``apply(activated)`` at the
    start of every replayed iteration, ``clear()`` when the replay ends (and
    defensively at close). ``apply`` is idempotent and cheap — it only
    touches flags whose value changed.
    """

    def __init__(
        self,
        operators: list[Operator],
        model_parts: list[nn.Module],
        stage_ids: list[int],
        enabled: bool = True,
        agree_group: Any = None,
        agree_device: torch.device | None = None,
        scope: str | None = None,
    ) -> None:
        self.enabled = enabled
        # Freezing is COLLECTIVE-VISIBLE: a fully-frozen expert module has no
        # gradients, so FSDP2 skips that group's reduce-scatter entirely
        # (_fsdp_param_group.py: "if len(fsdp_params_with_grad) == 0: return").
        # Under EP each rank scores only its EP-local experts, so the frozen
        # set is rank-dependent and ranks would issue different numbers of
        # collectives -> permanent NCCL desync. Every decision is therefore
        # AND-ed across the group before it is applied. Must be the same
        # "stage column" group _reduce_popularity uses: under PP the operator
        # vectors differ in length between stages, so a WORLD reduce is
        # malformed. None => single-process (tests); apply locally.
        self._agree_group = agree_group
        self._agree_device = agree_device
        self._agree_logged = False
        # Scope switch, in the style of labexps/warmup's LETO_WARMUP_NOOP:
        # nulls part of the mechanism while keeping every surrounding code
        # path (clip-norm pinning included) intact.
        #   expert (default) — only the grouped-expert detach
        #   whole  — only requires_grad_(False) on non_expert / gate
        #   all    — both
        #   none   — freeze nothing (isolates the pinned clip norm)
        #
        # The default is `expert`, NOT `all`, and that is a correctness
        # requirement rather than a preference. Measured on qwen3_pp2
        # (labexps/moevement_verify/m12_frozen_skip_ab.sh, one binary, only
        # this variable changed):
        #
        #   off/none/expert -> 0 mismatches against the fault-free baseline
        #   whole/all       -> diverges, reproducibly, at the replay boundary
        #
        # `requires_grad_(False)` mutates the torch.compile'd JOINT
        # forward/backward graph, so the partition changes and the surviving
        # operators' arithmetic is no longer bit-identical. It shows up two
        # ways: at the first post-replay step (the recompiled dense variant),
        # and -- when a pipeline stage recomputes activations over live p2p
        # instead of replaying them from the upstream log -- in the replayed
        # steps themselves. The grouped-expert detach has no such effect: it
        # drops a weight-gradient without changing the graph's partitioning,
        # and it is the paper's actual §3.3 target (expert weights dominate a
        # MoE backward). `expert` therefore keeps the technique and bit-
        # identity; `all` buys the dense backbone's share at the cost of
        # exactness. Do not restore `all` as the default without re-running
        # that A/B.
        # An explicit `scope=` wins over the env var, so tests that exercise a
        # specific half of the mechanism say so in the call rather than
        # depending on ambient environment.
        self._scope = (
            scope
            if scope is not None
            else os.environ.get("LETO_FROZEN_SKIP_SCOPE", "expert")
        ).lower()
        assert self._scope in ("all", "whole", "expert", "none"), (
            f"LETO_FROZEN_SKIP_SCOPE must be all|whole|expert|none, "
            f"got {self._scope!r}"
        )
        self._operators = operators
        self._by_name = {op.name: op for op in operators}
        self._armed = False
        self._steps_since_arm = 0
        self._frozen_now: set[str] = set()
        self.stats = FreezeStats()

        # Whole-tensor operators: op name -> its parameters.
        self._whole_tensor_ops: dict[str, list[nn.Parameter]] = {}
        # Expert modules: experts module -> the op names of its EP-local
        # experts. Rebuilt from the module tree the same way
        # operators.discover_operators names them.
        self._expert_modules: list[tuple[nn.Module, list[str], str]] = []
        # Model part bookkeeping for safety rule 1.
        self._parts: list[_PartInfo] = []
        # Parameters we flipped, so clear() restores exactly those.
        self._flipped: list[nn.Parameter] = []

        for op in operators:
            if op.kind is OperatorKind.EXPERT:
                continue
            self._whole_tensor_ops[op.name] = [
                param for _fqn, param, _idx in op.param_entries
            ]

        for part_index, (model_part, stage) in enumerate(
            zip(model_parts, stage_ids)
        ):
            info = _PartInfo(index=part_index, module=model_part)
            for op in operators:
                if op.stage != stage:
                    continue
                info.op_names.append(op.name)
                if op.kind is OperatorKind.NON_EXPERT:
                    info.non_expert_op = op.name
            self._parts.append(info)

            for moe_fqn, moe in model_part.named_modules():
                if not isinstance(moe, MoE):
                    continue
                label = _layer_label(moe_fqn)
                experts = moe.experts
                names = [
                    name
                    for name in self._by_name
                    if name.startswith(f"stage{stage}_layer{label}_expert")
                ]
                if not names:
                    continue
                supported = hasattr(experts, "_moevement_freeze_wgrad")
                if not supported:
                    logger.warning(
                        "[moevement] expert module %s (%s) has no "
                        "_moevement_freeze_wgrad hook; its weight-gradients "
                        "will not be skipped during replay",
                        moe_fqn, type(experts).__name__,
                    )
                    continue
                self._expert_modules.append((experts, sorted(names), moe_fqn))

    # ------------------------------------------------------------------

    def arm(self, model_parts: list[nn.Module] | None = None) -> None:
        """A replay was armed. Unless FSDP2's per-group mixed-precision init
        has already run (or can be forced right now), the first ``apply()``
        after this is a no-op — safety rule 2."""
        self._armed = True
        self._steps_since_arm = 0
        if model_parts is not None and _fsdp_mp_init_done(model_parts):
            # Already initialised (mid-run arm, or the force below worked):
            # no need to spend a replayed iteration on it.
            self._steps_since_arm = 1

    def note_step_completed(self) -> None:
        """Optional extra tick of the safety-rule-2 counter. ``apply()``
        already ticks it on every call, so this only ever makes the first
        real freeze happen sooner; kept for callers that want to say so
        explicitly."""
        self._steps_since_arm += 1

    @property
    def warmed_up(self) -> bool:
        return self._steps_since_arm >= 1

    def _agree(
        self, ready: int, expert_bits: list[int], whole_bits: list[int]
    ) -> tuple[int, list[int], list[int]]:
        """AND every freeze decision across the stage column.

        MIN over 0/1 is logical AND: an operator is skipped only if EVERY rank
        agrees it is fully frozen. That direction is the safe one — an
        out-voted rank simply pays the full backward and falls back to the
        exact grad-masking path, so numerics are unchanged.

        ``ready`` rides the same vector so safety rule 2 defers world-uniformly:
        it derives from the rank-local FSDP mixed-precision probe, which is a
        divergence source of exactly the same class.
        """
        # NOTE: agree_group None means "the DEFAULT (world) group", not "no
        # agreement" -- _build_popularity_group returns None whenever PP is
        # disabled, because at PP=1 the whole world is one stage column. An
        # earlier version of this guard treated None as "single process, skip",
        # which silently disabled the agreement on every PP=1 job -- including
        # qwen3_tp2fsdp4ep4, the very combo the desync was found on. The only
        # reason to skip is that there is no process group at all.
        if not (dist.is_available() and dist.is_initialized()):
            return ready, expert_bits, whole_bits

        payload = [ready] + expert_bits + whole_bits
        t = torch.tensor(payload, dtype=torch.int32, device=self._agree_device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=self._agree_group)
        out = [int(v) for v in t.tolist()]

        # Say ONCE that the agreement is live. A unanimous verdict logs nothing
        # below, so without this an agreement that never ran is indistinguish-
        # able from one that ran and agreed -- which is precisely how a version
        # of this guard that silently skipped itself on every PP=1 job survived
        # a full verification matrix.
        if not self._agree_logged:
            self._agree_logged = True
            logger.info(
                "[moevement] frozen-skip world agreement ACTIVE: %d-bit verdict "
                "MIN-reduced over a %d-rank group (%s)",
                len(payload),
                dist.get_world_size(group=self._agree_group),
                "default/world group" if self._agree_group is None
                else "stage column",
            )

        n_expert = len(expert_bits)
        agreed_ready = out[0]
        agreed_expert = out[1 : 1 + n_expert]
        agreed_whole = out[1 + n_expert :]

        if agreed_expert != expert_bits or agreed_whole != whole_bits:
            logger.info(
                "[moevement] frozen-skip verdict overridden by the world: "
                "expert modules %d -> %d, whole-tensor ops %d -> %d "
                "(a peer rank scored those operators active)",
                sum(expert_bits), sum(agreed_expert),
                sum(whole_bits), sum(agreed_whole),
            )
        if agreed_ready != ready:
            logger.info(
                "[moevement] frozen-skip deferred by the world "
                "(safety rule 2 not satisfied on every rank)"
            )
        return agreed_ready, agreed_expert, agreed_whole

    def apply(self, activated: set[str]) -> FreezeStats:
        """Freeze every operator not in ``activated`` that can be frozen."""
        stats = FreezeStats()
        if not self.enabled:
            # Config-derived and world-uniform, so returning here cannot
            # desynchronise the agreement collective below.
            self.stats = stats
            return stats

        # Safety rule 2: let one full step run before flipping anything
        # (nothing is frozen yet at this point — arm() is only called from
        # load() and from the warmup path, both with a clean state). The
        # verdict is world-agreed rather than returned on immediately,
        # because the collective below must be reached by every rank.
        ready = 0 if (self._armed and not self.warmed_up) else 1

        frozen = {op.name for op in self._operators if op.name not in activated}

        # Safety rule 1: never leave a model part with no trainable parameter.
        keep_trainable: set[str] = set()
        for part in self._parts:
            if not part.op_names:
                continue
            if all(name in frozen for name in part.op_names):
                if part.non_expert_op is not None:
                    keep_trainable.add(part.non_expert_op)
                    stats.parts_kept_trainable += 1
                else:
                    # No non_expert operator on this part (should not happen
                    # — discovery always emits one) — keep every operator
                    # trainable rather than risk an empty backward.
                    keep_trainable.update(part.op_names)
                    stats.parts_kept_trainable += 1
        frozen -= keep_trainable

        freeze_whole = self._scope in ("all", "whole")
        freeze_expert = self._scope in ("all", "expert")

        # Build the local verdict, AND it across the stage column, then apply
        # only the agreed bits. Fixed-length payload so the guard collective
        # itself never varies in size.
        whole_order = list(self._whole_tensor_ops)
        expert_bits = [
            1 if freeze_expert and all(n in frozen for n in names) else 0
            for _experts, names, _fqn in self._expert_modules
        ]
        whole_bits = [
            1 if freeze_whole and name in frozen else 0 for name in whole_order
        ]
        ready, expert_bits, whole_bits = self._agree(ready, expert_bits, whole_bits)

        if not ready:
            self._steps_since_arm += 1
            self.stats = stats
            return stats

        self._frozen_now = frozen
        self._flipped.clear()

        for idx, op_name in enumerate(whole_order):
            params = self._whole_tensor_ops[op_name]
            want_frozen = bool(whole_bits[idx])
            if want_frozen:
                stats.frozen_ops += 1
            for param in params:
                want_requires_grad = not want_frozen
                if param.requires_grad != want_requires_grad:
                    param.requires_grad_(want_requires_grad)
                if want_frozen:
                    self._flipped.append(param)
                    stats.frozen_params += 1
                    stats.frozen_param_numel += _local_numel(param)

        for idx, (experts, names, _fqn) in enumerate(self._expert_modules):
            all_frozen = bool(expert_bits[idx])
            any_frozen = any(name in frozen for name in names)
            if experts._moevement_freeze_wgrad != all_frozen:
                experts._moevement_freeze_wgrad = all_frozen
            if all_frozen:
                stats.expert_modules_skipped += 1
                stats.frozen_ops += len(names)
                for param in experts.parameters(recurse=False):
                    stats.frozen_params += 1
                    stats.frozen_param_numel += _local_numel(param)
            elif any_frozen:
                stats.expert_modules_partial += 1

        stats.applied = True
        self.stats = stats
        return stats

    def clear(self) -> None:
        """Restore full trainability. Idempotent; safe to call when the
        controller is disabled or was never applied."""
        for param in self._flipped:
            if not param.requires_grad:
                param.requires_grad_(True)
        self._flipped.clear()
        for experts, _names, _fqn in self._expert_modules:
            if experts._moevement_freeze_wgrad:
                experts._moevement_freeze_wgrad = False
        self._frozen_now = set()
        self._armed = False
        self._steps_since_arm = 0

    # ------------------------------------------------------------------

    def frozen_params_without_grad(self) -> list[nn.Parameter]:
        """Parameters this controller froze that consequently have no
        ``.grad``. The resilient optimizer's chunked step resolves
        ``param.grad`` by reference for every parameter in its precomputed
        schedule, so those must be materialised as zeros before its step —
        see ``MoevementCheckpointManager._fill_frozen_grads_for_resilient``.
        """
        out: list[nn.Parameter] = []
        for param in self._flipped:
            if param.grad is None:
                out.append(param)
        for experts, _names, _fqn in self._expert_modules:
            if not experts._moevement_freeze_wgrad:
                continue
            for param in experts.parameters(recurse=False):
                if param.grad is None:
                    out.append(param)
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "whole_tensor_ops": len(self._whole_tensor_ops),
            "expert_modules": len(self._expert_modules),
            "parts": len(self._parts),
        }


def _local_numel(param: torch.Tensor) -> int:
    local = getattr(param, "to_local", None)
    return local().numel() if local is not None else param.numel()


def _fsdp_mp_init_done(model_parts: list[nn.Module]) -> bool:
    """True when every FSDP2 parameter group has already recorded its
    mixed-precision dtypes (safety rule 2).

    ``FSDPParamGroup._init_mp_dtypes`` runs once, from the first forward's
    lazy init, and records ``None`` when the group holds no trainable
    parameter at that instant — which would later break the reduce-scatter
    of a group that becomes trainable again. If no forward has run yet we
    try to force the lazy init here; if that is not possible (older/newer
    torch, no FSDP at all), the caller falls back to skipping the freeze for
    one iteration.

    Returns False conservatively on any unexpected shape of the private API.
    """
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_state import (
            _get_module_fsdp_state,
        )
    except Exception:  # pragma: no cover - torch without FSDP2 internals
        return False

    states = []
    for model_part in model_parts:
        for module in model_part.modules():
            state = _get_module_fsdp_state(module)
            if state is not None:
                states.append(state)
    if not states:
        # No FSDP2 in this model: nothing to initialise, freezing is safe.
        return True

    for state in states:
        if getattr(state, "_is_root", None) is None:
            try:
                state._lazy_init()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "[moevement] could not force FSDP2 lazy init before the "
                    "first frozen-skip iteration (%s); deferring the skip by "
                    "one replayed step", e,
                )
                return False
    for state in states:
        group = getattr(state, "_fsdp_param_group", None)
        if group is None:
            continue
        if getattr(group, "_orig_dtype", None) is None:
            logger.warning(
                "[moevement] an FSDP2 parameter group has no recorded "
                "original dtype; deferring the frozen skip by one replayed "
                "step"
            )
            return False
    return True
