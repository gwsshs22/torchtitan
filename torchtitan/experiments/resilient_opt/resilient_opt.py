"""Resilient optimizer: zero GPU overhead via CPU bootstrap + gradient reuse.

One GPU→CPU copy, then exponential growth on GPU using freed gradients.
Chunk size doubles every 3 chunks.

Terminology:
    chunk_size     = bytes of PARAMETERS updated per chunk
    input_buffer   = 3 × chunk_size (param + exp_avg + exp_avg_sq backup)

Schedule (init_chunk_size=1MB, max_chunk_size=256MB):

    CPU phase:  allocate 9MB CPU pinned (= init * 3 * 3)
                backup 9MB state → step 3MB params → frees 3MB grad

    GPU level 0: chunk_size=1MB, input_buf=3MB (fits in 3MB freed grad)
                 ×3 steps → frees 3×1MB → pool=6MB

    GPU level 1: chunk_size=2MB, input_buf=6MB (fits in 6MB pool)
                 ×3 steps → frees 3×2MB → pool=12MB
    ...
    doubles every 3 steps until max_chunk_size

Memory budget:
    GPU: 0 extra
    CPU: 9 × init_chunk_size pinned

Usage:
    resilient_opt = ResilientOptimizer(optimizers, rmp_client, device)
    resilient_opt.bind()  # also call again after optimizer.load_state_dict()
    resilient_opt.step()
"""

import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.cuda._pin_memory_utils import pin_memory
from torch.distributed._tensor import DTensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate

from leto.rmp.client import RmpClient, TensorSpec
from torchtitan.tools.logging import logger

# Unified marker values.  Negative = "phase" sentinel; non-negative encodes
# chunk progress as before (K*2 = backup pending for chunk K, K*2+1 = backup
# done / step pending).  Distinct from chunk values because chunk indices
# are ≥ 0.
#
# Lifecycle within step():
#   IDLE  ──fill──► PRE_RUNNING ──counter+=1──► (still PRE_RUNNING)
#     [if MoE balancing] ──fill─► EB_BACKUP_RUNNING ──copy tpe+eb to CPU──►
#                        ──fill─► EB_UPDATE_RUNNING ──compute+apply+zero tpe──►
#     ──fill─► PRE_DONE ──chunks──► POST_RUNNING ──fill─► IDLE
#
# The PRE_RUNNING claim is set *before* the counter bump.  That makes
# (counter=N+1, mark=IDLE) reachable only at step-N+1 completion (not
# during the next step's pre-bump window), which is what eliminates the
# old (counter, marker, hook_state, params_step)-witness disambiguation.
#
# EB_BACKUP_RUNNING / EB_UPDATE_RUNNING split the manual MoE expert-bias
# update so a fault mid-update can be rolled back to the pre-update CPU
# snapshot and re-applied deterministically — replacing the previous
# generic pre-hook invocation that had no rollback guarantee.
_MARKER_IDLE = -1
_MARKER_PRE_RUNNING = -2
_MARKER_PRE_DONE = -3
_MARKER_POST_RUNNING = -4
_MARKER_EB_BACKUP_RUNNING = -5
_MARKER_EB_UPDATE_RUNNING = -6


def _get_local(tensor):
    """Get underlying local tensor from DTensor, or return as-is."""
    if isinstance(tensor, DTensor):
        return tensor._local_tensor
    return tensor


def _make_group_step_lambda(
    group, params, exp_avgs, exp_avg_sqs, steps, grad_specs,
    beta1, beta2, weight_decay, eps,
    scaled_views, clip_coef_0d,
):
    """Factory that closes over the args (avoids the loop-var late-binding trap).

    ``scaled_views[i]`` is a fixed-address scratch tensor for grad_specs[i];
    each call writes ``grad * clip_coef_0d`` into it (the live gradient is
    never modified) and the fused step consumes the scaled views. This is
    bit-identical to the stock ``_foreach_mul_(grad, clip_coef) -> fused``
    path (same multiply, same fp32 coef, same out dtype) while keeping the
    RMP gradient pristine so a mid-step fault is replayable. ``clip_coef_0d``
    is a stable-address RMP scalar whose contents change per step.
    """
    def lam():
        for (sl, start, end), buf in zip(grad_specs, scaled_views):
            flat = _get_local(sl.param_ref.grad).view(-1)
            sl.flat_grad = flat  # keep original for downstream _harvest_grads
            torch.mul(flat[start:end], clip_coef_0d, out=buf)
        torch._fused_adamw_(
            params, scaled_views, exp_avgs, exp_avg_sqs,
            [], steps,
            amsgrad=False,
            lr=group["lr"],
            beta1=beta1, beta2=beta2,
            weight_decay=weight_decay, eps=eps,
            maximize=False,
            grad_scale=None,
            found_inf=None,
        )
    return lam


@dataclass
class _ParamInfo:
    """Per-parameter info with pre-computed flat views (no alloc in hot path)."""

    param: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    step: torch.Tensor
    group: dict
    numel: int
    param_elem_size: int  # bytes per element for param (and grad)
    bytes_per_elem: int  # param + exp_avg + exp_avg_sq per element
    param_bytes: int  # numel × param_elem_size
    backup_bytes: int  # numel × bytes_per_elem
    # Pre-computed flat views (verified contiguous at init)
    flat_param: torch.Tensor = None  # type: ignore[assignment]
    flat_exp_avg: torch.Tensor = None  # type: ignore[assignment]
    flat_exp_avg_sq: torch.Tensor = None  # type: ignore[assignment]


@dataclass
class _MoEEntry:
    """One MoE layer's worth of (expert_bias, tokens_per_expert) state.

    Captured at construction time so the atomic update path doesn't need to
    re-walk the model.
    """

    expert_bias: torch.Tensor  # persistent buffer, replicated across EP/TP
    tokens_per_expert: torch.Tensor  # non-persistent buffer, accumulated in fwd
    load_balance_coeff: float
    ac_enabled: bool  # if True, tpe was double-counted by activation checkpoint
    # Pre-computed flat byte views of the local tensors (for backup/restore).
    eb_flat_uint8: torch.Tensor = None  # type: ignore[assignment]
    tpe_flat_uint8: torch.Tensor = None  # type: ignore[assignment]
    eb_bytes: int = 0
    tpe_bytes: int = 0


@dataclass
class _SliceEntry:
    """One unit of work: a contiguous slice of one param."""

    param_ref: torch.Tensor  # original nn.Parameter (for .grad access)
    step: torch.Tensor
    group: dict
    start: int
    end: int
    is_first_slice: bool
    param_bytes: int  # (end-start) × param_elem_size
    backup_bytes: int  # (end-start) × bytes_per_elem
    # Pre-computed flat views (zero-copy, from _ParamInfo)
    flat_param: torch.Tensor = None  # type: ignore[assignment]
    flat_exp_avg: torch.Tensor = None  # type: ignore[assignment]
    flat_exp_avg_sq: torch.Tensor = None  # type: ignore[assignment]
    flat_grad: torch.Tensor = None  # set at step time (grad may change)


@dataclass
class _ChunkCache:
    """Per-chunk fast-path state, lazily populated on the first normal step()."""

    backup_graph: "torch.cuda.CUDAGraph | None" = None
    # Side stream the backup graph was captured on. Retained so the graph's
    # recorded stream associations stay alive for the lifetime of the cache.
    backup_capture_stream: "torch.cuda.Stream | None" = None
    # Nullary callables that issue the kernels for this chunk's _step_chunk.
    step_lambdas: "list | None" = None


class ResilientOptimizer:
    """Zero-GPU-overhead resilient optimizer.

    CPU bootstrap (one copy) → 3-step doubling on freed gradient memory.
    """

    def __init__(
        self,
        optimizers,
        rmp_client: RmpClient,
        device: torch.device,
        init_chunk_size_mb: int = 1,
        max_chunk_size_mb: int = 256,
        model_parts: list | None = None,
        parallel_dims: Any | None = None,
    ):
        self._optimizers = optimizers
        self._device = device
        self._init_chunk_size = init_chunk_size_mb * 1024 * 1024
        self._max_chunk_size = max_chunk_size_mb * 1024 * 1024
        self._init_chunk_size_mb = init_chunk_size_mb
        self._max_chunk_size_mb = max_chunk_size_mb
        self._model_parts = model_parts or []
        self._parallel_dims = parallel_dims
        self._loss_mesh = (
            parallel_dims.get_optional_mesh("loss")
            if parallel_dims is not None
            else None
        )

        # Populated by bind(). step()/maybe_recover() require bind() first.
        self._all_params: list[_ParamInfo] = []
        self._num_params = 0
        self._schedule: list[list[_SliceEntry]] = []

        # Per-chunk fast-path cache (one entry per chunk, lazily filled on the
        # first normal step()).  Each entry holds:
        #   * backup_graph: CUDA graph capturing _backup_chunk_scatter for
        #     this chunk (fixed src/dst pointers across iters).
        #   * backup_capture_stream: side stream the graph was captured on
        #     (kept alive so the graph's recorded ops stay valid).
        #   * step_lambdas: list of nullary callables that re-issue the
        #     _step_chunk kernels.  Lambdas re-read group["lr"] each call
        #     (lr_scheduler ticks per step) and re-resolve param.grad each
        #     call (FSDP may reallocate grads on zero_grad(set_to_none=True)).
        # Invalidated on bind() / resync_after_external_load() / after
        # maybe_recover() because tensor pointers may have shifted.
        # Disabled entirely during maybe_recover via _recovery_in_progress.
        # Assumption: param / param.grad / optim-state buffer addresses are
        # stable across iterations once bind() has run. FSDP's FlatParameter
        # reuses the grad shard across iters; param storage doesn't move
        # outside of explicit reshard/load events (which invalidate via
        # bind / resync_after_external_load / maybe_recover).  Synthetic
        # tests that reassign tensors on every step must populate in
        # place to honor this assumption.
        self._chunk_caches: list[_ChunkCache] = []
        self._recovery_in_progress = False

        # MoE expert-bias state — only populated when the standard path
        # would also do load balancing. The signal is whether the
        # OptimizersContainer has any step pre-hook registered (which
        # ``build_optimizers_with_moe_load_balancing`` does, but the
        # plain ``build_optimizers`` doesn't). Mirroring the standard
        # path's behavior keeps loss bit-identical between RMP-GPU and
        # no-RMP runs for both kinds of MoE models.
        std_has_eb_hook = bool(
            getattr(optimizers, "_optimizer_step_pre_hooks", None)
        )
        self._moe_entries: list[_MoEEntry] = (
            self._collect_moe_entries() if std_has_eb_hook else []
        )
        self._has_moe = bool(self._moe_entries)
        self._eb_backup_buffer: torch.Tensor | None = None
        if self._has_moe:
            self._setup_eb_backup_buffer(rmp_client)

        # -- CPU buffer: init_chunk_size × 3 × 3 --------------------------
        # Backs up (init_chunk_size × 3) worth of params in one copy.
        # Allocated via RMP so it survives faults and is accessible on recovery.
        cpu_buffer_bytes = self._init_chunk_size * 3 * 3
        cpu_storage, cpu_allocated = rmp_client.get_or_allocate_cpu_memory(
            "resilient/cpu_buffer", cpu_buffer_bytes,
        )
        pin_memory(cpu_storage.data_ptr(), cpu_storage.nbytes())
        self._cpu_buffer = torch.empty(0, dtype=torch.uint8).set_(
            source=cpu_storage, storage_offset=0, size=(cpu_buffer_bytes,),
        )
        self._cpu_buffer_bytes = cpu_buffer_bytes
        # The CPU phase updates this many param bytes:
        self._cpu_phase_param_bytes = self._init_chunk_size * 3

        # -- RMP marker ----------------------------------------------------
        device_idx = device.index if hasattr(device, "index") else 0
        marker_tensors, gpu_allocated = rmp_client.get_or_allocate_tensors(
            [TensorSpec(
                name="resilient/marker", shape=(1,),
                dtype=torch.int64, device=device_idx,
            )]
        )
        self._marker = marker_tensors["resilient/marker"]
        if gpu_allocated:
            self._marker.fill_(_MARKER_IDLE)
            torch.cuda.current_stream().synchronize()

        # -- Step counter (RMP-backed GPU) ------------------------------------
        step_cnt_tensors, step_cnt_allocated = rmp_client.get_or_allocate_tensors(
            [TensorSpec(
                name="resilient/step_counter", shape=(1,),
                dtype=torch.int64, device=device_idx,
            )]
        )
        self._step_counter = step_cnt_tensors["resilient/step_counter"]
        # 0-dim view used by _foreach_copy_ in _step_chunk to broadcast the
        # counter into per-param state["step"] tensors (shapes must match
        # pairwise). Avoids the per-chunk `_step_counter.item()` host sync.
        self._step_counter_0d = self._step_counter.view([])
        # Initialized from optim.state["step"] on the first bind() after a
        # fresh allocation; deferred because the source tensor lives in
        # optimizer state, which we don't traverse in __init__.
        self._step_counter_needs_init = step_cnt_allocated

        # -- Gradient-clip coefficient (RMP-backed GPU scalar) ----------------
        # Holds clamp(max_norm / (total_norm + 1e-6), max=1.0) for the current
        # step. dist_utils.clip_grad_norm_(..., compute_only=True) computes it
        # WITHOUT scaling grads in place; we apply it per chunk inside the
        # fault-recoverable step (grad * coef -> scratch -> fused AdamW), so a
        # fault mid-step is replayable and grads in RMP are never mutated.
        #
        # Why RMP-backed: the value must survive a transient restart so the
        # recovery replay uses the same coefficient as the no-fault step
        # (set_clip_coef writes it before step()'s PRE_RUNNING/counter bump,
        # mirroring the single-scalar step-counter atomicity discipline).
        # Why a fixed-address 0-dim view (_clip_coef_0d): so the per-chunk
        # `torch.mul(grad, coef, out=scratch)` reads a stable device address
        # whose CONTENTS are updated in place each step — the same
        # CUDA-graph-safe pattern as _step_counter_0d (no value baked at
        # capture; no realloc). The step path is eager today, but this keeps
        # it correct if it is ever graph-captured.
        clip_coef_tensors, clip_coef_allocated = rmp_client.get_or_allocate_tensors(
            [TensorSpec(
                name="resilient/clip_coef", shape=(1,),
                dtype=torch.float32, device=device_idx,
            )]
        )
        self._clip_coef = clip_coef_tensors["resilient/clip_coef"]
        self._clip_coef_0d = self._clip_coef.view([])
        if clip_coef_allocated:
            # Default 1.0 == "no clipping" (and a true no-op multiply), so a
            # caller that never calls set_clip_coef() — e.g. the unit tests —
            # stays bit-identical to plain optimizer.step().
            self._clip_coef.fill_(1.0)
            torch.cuda.current_stream().synchronize()

        # -- Pre-reduction local norm contributions (RMP-backed GPU) ----------
        # Per-rank `_NormPartial` local scalar(s) from `get_total_norm`, taken
        # BEFORE the reduction collective (full_tensor / EP / PP all-reduce).
        # Persisted *before* the reduction so collective completion witnesses
        # universal persistence: any rank that advances past the reduction
        # ⟹ every rank entered the reduction ⟹ every rank executed
        # persist_clip_locals just prior ⟹ every rank's RMP holds the fresh
        # locals for the resume step. Recovery re-runs the SAME reducer over
        # those locals — no peer reads, no grad reads (immune to advanced-rank
        # mutated grads). See ``persist_clip_locals`` /
        # ``recompute_clip_coef_from_locals`` for the two phases.
        #
        # Two slots: EP layout uses [0]=ep_local, [1]=non_ep_local; non-EP
        # layout uses [0]=total_local. Slot 1 is unused (kept 0.0) in non-EP
        # runs and never feeds the reducer (ep_enabled is static per run).
        clip_locals_tensors, clip_locals_allocated = (
            rmp_client.get_or_allocate_tensors(
                [TensorSpec(
                    name="resilient/clip_local_norms", shape=(2,),
                    dtype=torch.float32, device=device_idx,
                )]
            )
        )
        self._clip_local_norms = clip_locals_tensors["resilient/clip_local_norms"]
        if clip_locals_allocated:
            self._clip_local_norms.fill_(0.0)
            torch.cuda.current_stream().synchronize()

        # CPU mirror of `_clip_local_norms` — survives a FATAL fault where the
        # RMP GPU server is torn down (transient recovery still reads the GPU
        # tensor; this CPU copy is the durable fallback). Mirrored after every
        # `persist_clip_locals` via a non_blocking GPU→CPU memcpy. The CPU
        # storage MUST be pinned for `non_blocking=True` to actually be an
        # async DMA — without pinning CUDA falls back to a synchronous copy,
        # which would defeat the purpose of avoiding a host-device sync.
        # The copy + the subsequent reducer collective are enqueued on the
        # current CUDA stream in order, so the collective-completion witness
        # ("any rank that advances ⟹ every rank persisted") still holds for
        # the CPU mirror too.
        clip_locals_bytes = (
            self._clip_local_norms.numel() * self._clip_local_norms.element_size()
        )
        clip_locals_cpu_storage, _ = rmp_client.get_or_allocate_cpu_memory(
            "resilient/clip_local_norms_cpu", clip_locals_bytes,
        )
        pin_memory(clip_locals_cpu_storage.data_ptr(), clip_locals_cpu_storage.nbytes())
        _clip_locals_cpu_u8 = torch.empty(0, dtype=torch.uint8).set_(
            source=clip_locals_cpu_storage, storage_offset=0,
            size=(clip_locals_bytes,),
        )
        # Re-view the pinned uint8 buffer as the same shape/dtype as the GPU
        # tensor so `_clip_local_norms_cpu.copy_(_clip_local_norms, ...)` is a
        # straight elementwise async DMA.
        self._clip_local_norms_cpu = _clip_locals_cpu_u8.view(
            self._clip_local_norms.dtype
        )

        # Persistent GPU scratch holding scaled grads for one chunk at a time
        # (grads themselves are never modified). Lazily sized to the largest
        # chunk's total grad bytes on the first step (grads exist by then);
        # reset on bind() because the schedule may change. Reused across
        # chunks: single-stream ordering serializes mul -> fused per chunk.
        self._grad_scratch: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # MoE expert-bias load balancing (manual, atomic, fault-tolerant)
    # ------------------------------------------------------------------
    #
    # The auxiliary-loss-free balancing scheme (Wang et al. 2024) updates
    # `expert_bias` from `tokens_per_expert` once per optimizer step.  It
    # used to be wired up as a torch.optim pre-hook on the optimizer
    # container; ResilientOptimizer would invoke it via _run_pre_hooks at
    # step time.  But that path is non-rollbackable: a fault landing in
    # the middle of the all-reduce or the per-layer `expert_bias.add_`
    # leaves expert_bias / tokens_per_expert in a partially-applied state
    # that the recovery side has no way to roll back, and replaying the
    # pre-hook on top of zero'd or partial tpe produces a different delta.
    #
    # Instead we copy `(expert_bias, tokens_per_expert)` of every MoE
    # layer into a small RMP-backed CPU buffer first, then apply the
    # update.  Two markers split the operation:
    #
    #   EB_BACKUP_RUNNING : copy phase (live state untouched). On
    #                       recovery: re-do backup + update.
    #   EB_UPDATE_RUNNING : apply phase (live state mutated). On
    #                       recovery: restore live from CPU, then redo
    #                       update.
    #
    # If no layer has load_balance_coeff configured this whole subsystem
    # is a no-op — _moe_entries stays empty and the markers are skipped.

    def _collect_moe_entries(self) -> list[_MoEEntry]:
        """Walk model_parts for MoE blocks with load_balance_coeff set."""
        entries: list[_MoEEntry] = []
        for model_part in self._model_parts:
            layers = getattr(model_part, "layers", None)
            if layers is None:
                continue
            for transformer_block in layers.values():
                if not getattr(transformer_block, "moe_enabled", False):
                    continue
                moe = getattr(transformer_block, "moe", None)
                if moe is None or moe.expert_bias is None:
                    continue
                coeff = getattr(moe, "load_balance_coeff", None)
                if not coeff:
                    continue
                ac_enabled = (
                    getattr(transformer_block, "checkpoint_impl", None)
                    is CheckpointImpl.NO_REENTRANT
                )
                eb_local = _get_local(moe.expert_bias)
                tpe_local = _get_local(moe.tokens_per_expert)
                eb_bytes = eb_local.numel() * eb_local.element_size()
                tpe_bytes = tpe_local.numel() * tpe_local.element_size()
                entries.append(
                    _MoEEntry(
                        expert_bias=moe.expert_bias,
                        tokens_per_expert=moe.tokens_per_expert,
                        load_balance_coeff=float(coeff),
                        ac_enabled=ac_enabled,
                        eb_flat_uint8=eb_local.view(torch.uint8).reshape(-1),
                        tpe_flat_uint8=tpe_local.view(torch.uint8).reshape(-1),
                        eb_bytes=eb_bytes,
                        tpe_bytes=tpe_bytes,
                    )
                )
        return entries

    def _setup_eb_backup_buffer(self, rmp_client: RmpClient) -> None:
        """Allocate the RMP-backed CPU snapshot buffer for tpe + expert_bias."""
        total_bytes = sum(e.eb_bytes + e.tpe_bytes for e in self._moe_entries)
        # Tiny in absolute terms (a few KB for typical configs) but kept in
        # RMP CPU memory so it survives a TRANSIENT recovery.
        eb_storage, _ = rmp_client.get_or_allocate_cpu_memory(
            "resilient/eb_backup", total_bytes,
        )
        pin_memory(eb_storage.data_ptr(), eb_storage.nbytes())
        self._eb_backup_buffer = torch.empty(0, dtype=torch.uint8).set_(
            source=eb_storage, storage_offset=0, size=(total_bytes,),
        )
        self._eb_backup_total_bytes = total_bytes

    @torch.no_grad()
    def _backup_moe_state(self) -> None:
        """Copy live `(expert_bias, tokens_per_expert)` of every MoE layer to CPU."""
        offset = 0
        buf = self._eb_backup_buffer
        for entry in self._moe_entries:
            buf[offset : offset + entry.eb_bytes].copy_(
                entry.eb_flat_uint8, non_blocking=True
            )
            offset += entry.eb_bytes
            buf[offset : offset + entry.tpe_bytes].copy_(
                entry.tpe_flat_uint8, non_blocking=True
            )
            offset += entry.tpe_bytes

    @torch.no_grad()
    def _restore_moe_state(self) -> None:
        """Restore live `(expert_bias, tokens_per_expert)` from the CPU snapshot."""
        offset = 0
        buf = self._eb_backup_buffer
        for entry in self._moe_entries:
            entry.eb_flat_uint8.copy_(
                buf[offset : offset + entry.eb_bytes], non_blocking=True
            )
            offset += entry.eb_bytes
            entry.tpe_flat_uint8.copy_(
                buf[offset : offset + entry.tpe_bytes], non_blocking=True
            )
            offset += entry.tpe_bytes

    def _atomic_update_expert_bias(self) -> None:
        """Backup → marker transition → apply update.

        No-op when no MoE layer has load_balance_coeff configured.  When
        called as part of `step()`, this leaves the marker at
        EB_UPDATE_RUNNING; the caller is responsible for transitioning to
        PRE_DONE once the apply completes.
        """
        if not self._has_moe:
            return
        self._marker.fill_(_MARKER_EB_BACKUP_RUNNING)
        self._backup_moe_state()
        self._marker.fill_(_MARKER_EB_UPDATE_RUNNING)
        self._do_expert_bias_update()

    @torch.no_grad()
    def _do_expert_bias_update(self) -> None:
        """Compute and apply the per-layer expert_bias delta, then zero tpe.

        Mirrors `_update_expert_bias` from torchtitan.components.optimizer
        but operates on the precomputed `_MoEEntry` list — no model walk
        and no `_optimizer_step_pre_hooks` lookup.
        """
        if not self._moe_entries:
            return

        tpe_list = []
        for entry in self._moe_entries:
            tpe = entry.tokens_per_expert
            if entry.ac_enabled:
                # Selective AC double-counts in fwd+bwd-recompute, so halve.
                # We use *float* division (not `// 2`) so the half is exact:
                # the upstream pre-hook used `// 2` (integer floor), which
                # breaks `sign(mean(tpe) - tpe)` invariance under the
                # `(leftover + 2x) / 2` arithmetic that arises when a
                # mid-step kill leaves a per-layer leftover proportional
                # to one or two forward passes.  Floor rounding on odd
                # 3x perturbs `mean - per_expert` by up to ±0.5 and can
                # flip the sign for experts close to the mean.  Float /2
                # keeps every per-expert value at exactly `(c+2)/2 * x_i`
                # for c ∈ {0,1,2}, so `mean - per_expert` is a uniform
                # (c+2)/2 scaling of the fault-free expression and the
                # sign — and therefore the delta — matches the no-fault
                # path bit-identically.
                tpe = tpe.float() * 0.5
            tpe_list.append(tpe)

        tokens_per_expert_by_layer = torch.vstack(tpe_list)

        if self._loss_mesh is not None:
            if isinstance(
                tokens_per_expert_by_layer, torch.distributed.tensor.DTensor
            ):
                tokens_per_expert_by_layer = tokens_per_expert_by_layer.redistribute(
                    placements=[Replicate()]
                    * tokens_per_expert_by_layer.device_mesh.ndim
                )
            else:
                pg = self._loss_mesh.get_group()
                torch.distributed.all_reduce(
                    tokens_per_expert_by_layer,
                    group=pg,
                    op=torch.distributed.ReduceOp.SUM,
                )

        for layer_idx, entry in enumerate(self._moe_entries):
            tpe = tokens_per_expert_by_layer[layer_idx].float()
            delta = entry.load_balance_coeff * torch.sign(tpe.mean() - tpe)
            delta = delta - delta.mean()
            entry.expert_bias.add_(delta)
            entry.tokens_per_expert.zero_()

    def bind(self):
        """(Re)bind to the optimizer's current state and rebuild the schedule.

        Captures fresh references to ``optimizer.state[param][...]`` tensors
        and to each ``param_group`` dict, then rebuilds the chunk schedule.

        Must be called once after construction (so step()/maybe_recover() can
        run), and again after any operation that replaces
        ``optimizer.state`` or ``optimizer.param_groups`` — most importantly
        ``optimizer.load_state_dict``, which deep-copies state tensors and
        installs new param_group dicts. Without rebinding, cached references
        from a previous bind point at the now-detached objects (lr in
        particular stops tracking lr_scheduler.step()).
        """
        all_params: list[_ParamInfo] = []

        for optimizer in self._optimizers:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param not in optimizer.state:
                        continue
                    state = optimizer.state[param]
                    lp = _get_local(param)
                    lm = _get_local(state["exp_avg"])
                    lv = _get_local(state["exp_avg_sq"])
                    numel = lp.numel()
                    pes = lp.element_size()
                    bpe = pes + lm.element_size() + lv.element_size()
                    # Verify contiguity — flat views must not allocate.
                    assert lp.is_contiguous(), f"param not contiguous: {lp.shape}"
                    assert lm.is_contiguous(), f"exp_avg not contiguous: {lm.shape}"
                    assert lv.is_contiguous(), f"exp_avg_sq not contiguous: {lv.shape}"

                    all_params.append(
                        _ParamInfo(
                            param=param,
                            exp_avg=state["exp_avg"],
                            exp_avg_sq=state["exp_avg_sq"],
                            step=state["step"],
                            group=group,
                            numel=numel,
                            param_elem_size=pes,
                            bytes_per_elem=bpe,
                            param_bytes=numel * pes,
                            backup_bytes=numel * bpe,
                            flat_param=lp.view(-1),
                            flat_exp_avg=lm.view(-1),
                            flat_exp_avg_sq=lv.view(-1),
                        )
                    )

        # Sort by param_bytes descending — large params freed first.
        all_params.sort(key=lambda p: p.param_bytes, reverse=True)

        self._all_params = all_params
        self._num_params = len(all_params)
        self._schedule = self._build_schedule()

        if self._step_counter_needs_init:
            self._step_counter.fill_(int(all_params[0].step.item()))
            torch.cuda.current_stream().synchronize()
            self._step_counter_needs_init = False

        # Caches reference _SliceEntry / param-state tensor pointers and
        # captured CUDA graphs that bake in those addresses. bind() rebuilds
        # the schedule and re-views state tensors, so any prior cache is
        # stale.
        self._invalidate_chunk_caches()

        # Schedule (and thus per-chunk grad byte sizes) may have changed; force
        # a one-time re-alloc of the grad-scale scratch on the next step. Safe:
        # the step path is eager and chunk caches were just invalidated, so no
        # captured graph or cached lambda references the old scratch address.
        self._grad_scratch = None

        logger.info(
            f"[ResilientOpt] bound {self._num_params} params, "
            f"{len(self._schedule)} chunks, "
            f"init_chunk: {self._init_chunk_size_mb} MB, "
            f"max_chunk: {self._max_chunk_size_mb} MB, "
            f"CPU buffer: {self._cpu_buffer_bytes / (1024**2):.1f} MB"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_step(self) -> int:
        """Return the current step counter value."""
        return self._step_counter.item()

    @torch.no_grad()
    def set_clip_coef(self, clip_coef_clamped: torch.Tensor) -> None:
        """Stage this step's gradient-clip coefficient for the chunked step.

        Call this from the training loop right before ``step()``, passing the
        ``clip_coef_clamped`` returned by
        ``dist_utils.clip_grad_norm_(..., compute_only=True)`` (i.e. the
        coefficient computed but NOT applied in place).

        The value is copied **in place** into the fixed-address RMP scalar
        ``self._clip_coef`` (no realloc): it then (a) survives a transient
        restart so the recovery replay scales grads by the exact same factor
        as the no-fault step, and (b) is read via the stable-address
        ``_clip_coef_0d`` inside the per-chunk multiply — CUDA-graph-safe by
        construction (contents change per step, address does not), the same
        discipline as ``_step_counter_0d``.

        Written before ``step()``'s PRE_RUNNING marker / counter bump, so any
        fault during the chunked step finds the correct coefficient already
        persisted (mirrors the single-scalar step-counter atomicity
        assumption the rest of this class relies on).
        """
        self._clip_coef.copy_(
            clip_coef_clamped.detach().reshape(1).to(self._clip_coef.dtype),
            non_blocking=True,
        )

    @torch.no_grad()
    def persist_clip_locals(self, locals_list: list[torch.Tensor]) -> None:
        """Persist the pre-reduction local norm contribution(s) into RMP.

        Call this from the training loop with ``ClipLocals.locals`` from
        ``dist_utils.clip_compute_locals(...)``, **before** invoking
        ``dist_utils.clip_reduce_from_locals(...)`` (which runs the reduction
        collective). The pre-collective ordering is what makes the
        collective-completion witness sound: any rank that advances past the
        reducer (counter bumps) ⟹ every rank entered the reducer ⟹ every
        rank executed this persist just prior. So at recovery every rank's
        RMP holds the fresh local for the resume step, and the recovery
        recompute is grad-free and peer-free.

        Layout (must match ``ep_enabled`` for the run):
            EP:     locals_list = [ep_local, non_ep_local]
            Dense:  locals_list = [total_local]

        Each entry is the raw ``get_total_norm`` result — a 0-dim
        ``_NormPartial`` DTensor in the typical case, or a non-DTensor 0-dim
        tensor when this rank has no grads in that group (PP+EP corner case).
        ``.to_local()`` extracts the rank-local scalar from a DTensor.
        """
        for i, local in enumerate(locals_list):
            scalar = local.to_local() if isinstance(local, DTensor) else local
            self._clip_local_norms[i].copy_(
                scalar.detach().reshape(()).to(self._clip_local_norms.dtype),
                non_blocking=True,
            )
        # Mirror the GPU tensor into pinned RMP-CPU memory so the persisted
        # local survives a FATAL fault (RMP GPU server killed). Async DMA on
        # the current CUDA stream — pinned target makes `non_blocking=True`
        # an actual fire-and-forget transfer, not a hidden host sync. Stream
        # ordering ensures the subsequent reducer collective only starts
        # after this memcpy completes.
        self._clip_local_norms_cpu.copy_(
            self._clip_local_norms, non_blocking=True
        )

    @torch.no_grad()
    def recompute_clip_coef_from_locals(
        self,
        parameters: Iterable[torch.Tensor],
        max_norm: float,
        norm_type: float = 2.0,
        pp_mesh: DeviceMesh | None = None,
        *,
        ep_enabled: bool,
    ) -> None:
        """Recovery only: reconstruct ``_NormPartial`` DTensor(s) from the
        locals persisted in RMP, run the SAME reducer used at normal time
        (``dist_utils.clip_reduce_from_locals``), and write the resulting
        clip_coef into ``self._clip_coef`` for the chunked replay.

        **All-ranks lockstep collective.** Must be called from a code path
        that every rank enters unconditionally (the top of
        ``Trainer._resilient_opt_recover`` before ``maybe_recover``'s
        per-rank branches). The reducer issues the same multi-collective
        sequence as the normal path (per-mesh ``full_tensor()``, optional PP
        all-reduce); bit-identity rests on deterministic collective replay
        over identical persisted local inputs.

        Why no peer / grad reads: ``_NormPartial`` is reconstructed from a
        ``zeros_like(p)`` *template* — values are irrelevant, only the
        param's DTensor spec (mesh + placements) is consulted, and that spec
        is reconstructed identically on every (re)start. The persisted
        scalar carries this rank's own pre-reduction norm contribution,
        captured before the original collective when grads were valid.
        """
        # Lazy import to avoid a top-level cycle with torchtitan.distributed.utils
        from torchtitan.distributed import utils as dist_utils

        parameters = list(parameters)

        if ep_enabled:
            ep_only = [
                p for p in parameters
                if isinstance(p, DTensor) and "ep" in p.device_mesh.mesh_dim_names
            ]
            non_ep_only = [
                p for p in parameters
                if isinstance(p, DTensor) and "ep" not in p.device_mesh.mesh_dim_names
            ]
            ep_recon = self._reconstruct_local_norm_dtensor(
                ep_only, self._clip_local_norms[0], norm_type
            )
            non_ep_recon = self._reconstruct_local_norm_dtensor(
                non_ep_only, self._clip_local_norms[1], norm_type
            )
            total_norm, clip_coef = dist_utils.clip_reduce_from_locals(
                [ep_recon, non_ep_recon], max_norm, norm_type, pp_mesh,
                ep_enabled=True,
            )
        else:
            recon = self._reconstruct_local_norm_dtensor(
                parameters, self._clip_local_norms[0], norm_type
            )
            total_norm, clip_coef = dist_utils.clip_reduce_from_locals(
                [recon], max_norm, norm_type, pp_mesh, ep_enabled=False,
            )

        self._clip_coef.copy_(
            clip_coef.detach().reshape(1).to(self._clip_coef.dtype),
            non_blocking=True,
        )
        logger.info(
            f"[ResilientOpt] Recovery: clip_coef recomputed from persisted locals "
            f"(coef={clip_coef.item():.6e}, total_norm={total_norm.item():.6e})"
        )

    @torch.no_grad()
    def _reconstruct_local_norm_dtensor(
        self,
        group_params: list[torch.Tensor],
        persisted_scalar: torch.Tensor,
        norm_type: float,
    ) -> torch.Tensor:
        """Wrap ``persisted_scalar`` as the ``_NormPartial`` DTensor that
        ``get_total_norm(group_grads)`` would have produced at normal time.

        Derives the mesh + placements from a ``zeros_like`` template — the
        values don't matter, only the DTensor spec, and that spec is
        deterministic across (re)starts. Returns a non-DTensor scalar if
        the group is empty on this rank (stock path also returns
        ``tensor(0.)`` there, and the reducer handles either case).
        """
        if not group_params:
            return persisted_scalar.detach().reshape(()).to(torch.float32)
        zeros = [torch.zeros_like(p) for p in group_params]
        template = torch.nn.utils.get_total_norm(zeros, norm_type, False, None)
        del zeros  # template is a 0-dim norm; the zero shards can go.
        if isinstance(template, DTensor):
            local_shape = template.to_local().shape
            local_dtype = template.to_local().dtype
            local_scalar = persisted_scalar.detach().reshape(local_shape).to(local_dtype)
            return DTensor.from_local(
                local_scalar, template.device_mesh, template.placements
            )
        return persisted_scalar.detach().reshape(()).to(template.dtype)

    @torch.no_grad()
    def zero_moe_tokens_per_expert(self) -> None:
        """Zero the per-MoE-layer ``tokens_per_expert`` buffer on this rank.

        Intended to be called from the recovery driver in train.py *after*
        ``maybe_recover`` has finished — by that point every recovery branch
        that legitimately reads tpe (EB_BACKUP_RUNNING / EB_UPDATE_RUNNING /
        PRE_DONE / chunk markers) has already consumed it, and zeroing here
        only clobbers values the standby would otherwise carry into its
        first post-promotion forward pass.
        """
        for entry in self._moe_entries:
            entry.tokens_per_expert.zero_()

    def resync_after_external_load(self):
        """Re-sync RMP-backed scalars to the just-loaded optimizer state.

        Step counter and marker are kept in RMP so they survive faults;
        they are otherwise initialized only on fresh allocation. After an
        external load (e.g. gemini.load()) overwrites params and optim
        state but leaves these scalars at whatever the prior active left,
        the step counter would be ahead of params.step and the next AdamW
        update would use the wrong bias-correction step.  Must be called
        after bind().
        """
        self._step_counter.fill_(int(self._all_params[0].step.item()))
        self._marker.fill_(_MARKER_IDLE)
        torch.cuda.current_stream().synchronize()
        # Captured graphs reference the pre-load tensor addresses; while the
        # data may live in the same allocator slot, gemini.load() rewrites
        # bytes via foreach copies that aren't replay-safe. Force re-capture.
        self._invalidate_chunk_caches()

    def step(self):
        """Fault-tolerant optimizer step with 3-step doubling.

        Order:
            (claim) → counter bump → MoE expert-bias update → chunked AdamW.

        The marker walks through PRE_RUNNING → (EB_BACKUP_RUNNING →
        EB_UPDATE_RUNNING) → PRE_DONE → chunk markers → POST_RUNNING →
        IDLE so each phase is unambiguously identifiable on recovery.
        PRE_RUNNING is claimed *before* the counter bump so the
        post-completion (counter=N+1, mark=IDLE) state can't be confused
        with the next step's pre-bump window.

        We do NOT invoke `_optimizer_step_pre_hooks` /
        `_optimizer_step_post_hooks` from here.  The expert-bias balancing
        — the only pre-hook that matters in production — is implemented
        in-line via `_atomic_update_expert_bias()` so a fault mid-update
        can be rolled back to the CPU snapshot.  Post-hooks are
        intentionally skipped (the only registered one is the model
        converter post-hook, which is a no-op for bf16).
        """
        self._marker.fill_(_MARKER_PRE_RUNNING)
        self._step_counter += 1
        self._atomic_update_expert_bias()
        self._marker.fill_(_MARKER_PRE_DONE)
        self._run_chunks(resume_from=0)
        self._marker.fill_(_MARKER_POST_RUNNING)
        self._marker.fill_(_MARKER_IDLE)

    def _run_chunks(self, resume_from: int):
        """Execute chunks starting from resume_from.

        Chunks before resume_from are treated as committed — only their
        freed grad memory is harvested for scratch space.

        Caller is responsible for the post-chunks marker transition
        (POST_RUNNING) and the eventual IDLE.

        Fast path: when not in recovery, _backup_chunk_scatter is replayed
        from a per-chunk CUDA graph (captured on first call) and _step_chunk
        invokes per-chunk cached lambdas (built on first call). Both are
        skipped during recovery — see _recovery_in_progress — and any prior
        cache is invalidated on bind / resync_after_external_load /
        maybe_recover so stale tensor pointers can never be replayed.

        Last-chunk harvest is skipped: its freed grad memory has no
        consumer in this step and no carry across iterations.
        """
        freed_grad_segments: list[torch.Tensor] = []
        last_idx = len(self._schedule) - 1
        use_fast_path = not self._recovery_in_progress

        for chunk_idx, chunk in enumerate(self._schedule):
            if chunk_idx < resume_from:
                # Committed chunk: harvest freed grad memory for scratch space.
                self._resolve_grads(chunk)
                if chunk_idx < last_idx:
                    self._harvest_grads(chunk, freed_grad_segments)
                continue

            backup_dest = (
                [self._cpu_buffer] if chunk_idx == 0 else freed_grad_segments
            )

            self._marker.fill_(chunk_idx * 2)
            if use_fast_path:
                cache = self._get_chunk_cache(chunk_idx)
                if cache.backup_graph is None:
                    cache.backup_graph, cache.backup_capture_stream = (
                        self._capture_backup_graph(chunk, backup_dest)
                    )
                else:
                    cache.backup_graph.replay()
            else:
                self._backup_chunk_scatter(chunk, backup_dest)

            self._marker.fill_(chunk_idx * 2 + 1)
            if use_fast_path:
                cache = self._get_chunk_cache(chunk_idx)
                if cache.step_lambdas is None:
                    cache.step_lambdas = self._build_and_run_step_lambdas(chunk)
                else:
                    for lam in cache.step_lambdas:
                        lam()
            else:
                self._step_chunk(chunk)

            if chunk_idx < last_idx:
                self._harvest_grads(chunk, freed_grad_segments)

    def maybe_recover(self, resume_step: int) -> bool:
        """Detect mid-step fault and resume from the interrupted phase.

        Args:
            resume_step: the step number this recovery should produce.
                Must equal step_counter or step_counter + 1.

        Assumes optimizer states, gradients, and the per-MoE-layer
        ``(expert_bias, tokens_per_expert)`` buffers persist in RMP across
        faults.  The expert-bias update is rolled back via the dedicated
        CPU snapshot (``_eb_backup_buffer``) for the EB_UPDATE_RUNNING
        marker; for every other in-flight marker we re-do the relevant
        phase from scratch.  Committed chunks are not re-applied.

        Cross-rank consistency: the only branch that issues a collective
        is ``_atomic_update_expert_bias`` (its all-reduce over
        ``loss_mesh``).  We use the params-step witness on PRE_RUNNING to
        distinguish "claim before counter bump" (skip; no collective)
        from "counter bumped, fault in eb-update or pre-chunks window"
        (run the full pipeline including the collective).  Ranks at
        EB_BACKUP_RUNNING / EB_UPDATE_RUNNING / PRE_DONE / chunk markers
        all run the eb-update path during recovery (PRE_DONE re-runs only
        chunks; the others re-run eb-update + chunks), keeping
        loss-mesh peers in lockstep.

        Recovery runs on the eager backup/step path (never the fast-path
        graphs/lambdas): tensor pointers may have shifted across the fault
        boundary, so any captured graph or grad reference is stale. The
        cache is invalidated on exit so the next normal step rebuilds it.
        """
        self._recovery_in_progress = True
        try:
            return self._maybe_recover_impl(resume_step)
        finally:
            self._recovery_in_progress = False
            self._invalidate_chunk_caches()

    def _maybe_recover_impl(self, resume_step: int) -> bool:
        stored = self._step_counter.item()
        marker_val = self._marker.item()

        if stored + 1 == resume_step:
            # Counter not bumped on this rank → run a fresh step.  Marker
            # should be IDLE in the common case; under the fault model the
            # only other reachable value here is PRE_RUNNING (mark.fill_
            # landed but counter+=1 didn't, which the fault model excludes
            # because faults only inject at fill_).  step() unconditionally
            # re-fills PRE_RUNNING, so a stale value is harmless.
            assert marker_val in (_MARKER_IDLE, _MARKER_PRE_RUNNING), (
                f"unexpected marker={marker_val} with stored+1==resume_step"
            )
            self.step()
            logger.info(f"[ResilientOpt] Full step executed stored={stored}")
            return True

        assert stored == resume_step, (
            f"step counter mismatch: stored={stored}, resume_step={resume_step}"
        )

        if marker_val == _MARKER_IDLE:
            # Step fully done.  The pre-claim invariant (PRE_RUNNING is set
            # before the next step's counter bump) means IDLE at this
            # counter value can only come from the trailing fill_(IDLE) of
            # the just-completed step — no witness check needed.
            logger.info(
                f"[ResilientOpt] Step fully complete, no recovery needed stored={stored}"
            )
            return False

        if marker_val == _MARKER_PRE_RUNNING:
            # PRE_RUNNING is set *before* the counter bump, so its
            # presence with stored==resume_step is ambiguous:
            #   * counter never bumped (claim made, then SIGKILL hit
            #     before counter += 1 — narrow window).  params[0].step
            #     is at the prior step's value == stored.  No work to do
            #     beyond resetting the mark.
            #   * counter bumped, fault landed before EB_BACKUP_RUNNING
            #     was claimed.  params[0].step is at stored - 1 (chunks
            #     for the new step haven't run yet).  Run the full
            #     pipeline.
            #
            # This branch needs the params-step witness because the eb
            # update path issues an all_reduce that would deadlock with
            # peers at IDLE.
            params_step = int(self._all_params[0].step.item())
            if params_step == stored:
                self._marker.fill_(_MARKER_IDLE)
                logger.info(
                    f"[ResilientOpt] PRE_RUNNING claim with no counter "
                    f"bump; reset and skip recovery stored={stored}"
                )
                return False

            self._atomic_update_expert_bias()
            self._marker.fill_(_MARKER_PRE_DONE)
            self._run_chunks(resume_from=0)
            self._marker.fill_(_MARKER_POST_RUNNING)
            self._marker.fill_(_MARKER_IDLE)
            logger.info(f"[ResilientOpt] Full step executed stored={stored}")
            return True

        if marker_val == _MARKER_EB_BACKUP_RUNNING:
            # Backup was in flight (CPU snapshot may be partial); live
            # tpe + expert_bias have not yet been mutated by the update.
            # Re-do backup + apply, then continue with chunks.
            self._atomic_update_expert_bias()
            self._marker.fill_(_MARKER_PRE_DONE)
            self._run_chunks(resume_from=0)
            self._marker.fill_(_MARKER_POST_RUNNING)
            self._marker.fill_(_MARKER_IDLE)
            logger.info(
                f"[ResilientOpt] Recovered from EB backup; full step executed "
                f"stored={stored}"
            )
            return True

        if marker_val == _MARKER_EB_UPDATE_RUNNING:
            # Update was in flight: live tpe + expert_bias may have been
            # partially mutated; the CPU snapshot from the BACKUP phase is
            # complete and authoritative.  Restore live state, re-apply
            # the update (no need to re-do the backup — CPU already has
            # the pre-update state).
            self._restore_moe_state()
            self._marker.fill_(_MARKER_EB_UPDATE_RUNNING)
            self._do_expert_bias_update()
            self._marker.fill_(_MARKER_PRE_DONE)
            self._run_chunks(resume_from=0)
            self._marker.fill_(_MARKER_POST_RUNNING)
            self._marker.fill_(_MARKER_IDLE)
            logger.info(
                f"[ResilientOpt] Recovered from EB update; full step executed "
                f"stored={stored}"
            )
            return True

        if marker_val == _MARKER_PRE_DONE:
            # Eb-update done, chunks not yet started.
            self._run_chunks(resume_from=0)
            self._marker.fill_(_MARKER_POST_RUNNING)
            self._marker.fill_(_MARKER_IDLE)
            logger.info(
                f"[ResilientOpt] Resumed from chunk 0 after pre-phase stored={stored}"
            )
            return True

        if marker_val == _MARKER_POST_RUNNING:
            # Chunks done; nothing else to run (no post-hooks).
            self._marker.fill_(_MARKER_IDLE)
            logger.info(f"[ResilientOpt] Cleared POST_RUNNING stored={stored}")
            return True

        # marker_val >= 0: chunk in flight.
        fault_chunk = marker_val // 2
        backup_done = (marker_val % 2 == 1)
        logger.warning(
            f"[ResilientOpt] Fault detected (marker={marker_val}, "
            f"chunk={fault_chunk}, backup_done={backup_done}), recovering"
        )
        if backup_done:
            self._restore_faulted_chunk(fault_chunk)
        self._run_chunks(resume_from=fault_chunk)
        self._marker.fill_(_MARKER_POST_RUNNING)
        self._marker.fill_(_MARKER_IDLE)
        logger.info(f"[ResilientOpt] Recovery complete stored={stored}")
        return True

    def enable_fault_injection(self, prob: float):
        """Monkey-patch marker.fill_() to randomly crash with given probability.

        When triggered, synchronizes CUDA and raises RuntimeError to simulate
        a GPU fault.  When not triggered, calls the original fill_() with
        near-zero overhead (one random.random() call).

        Seeds RNG from time so each process launch gets a different sequence.
        """
        import time
        rng = random.Random(time.time_ns())
        original_fill = self._marker.fill_

        def _faulting_fill(value):
            if rng.random() < prob:
                logger.warning(
                    f"[ResilientOpt] Fault injection triggered at "
                    f"marker.fill_({value})"
                )
                torch.cuda.synchronize()
                raise RuntimeError("[ResilientOpt] Injected fault")
            return original_fill(value)

        self._marker.fill_ = _faulting_fill

    # ------------------------------------------------------------------
    # Internal: schedule, cursor, harvest
    # ------------------------------------------------------------------

    def _build_schedule(self) -> list[list[_SliceEntry]]:
        """Precompute the chunk schedule (deterministic from param layout)."""
        schedule: list[list[_SliceEntry]] = []
        param_cursor = 0
        elem_cursor = 0
        freed_grad_bytes = 0

        # CPU phase
        chunk, param_cursor, elem_cursor = self._take_chunk_by_param_bytes(
            param_cursor, elem_cursor, self._cpu_phase_param_bytes,
        )
        if chunk:
            schedule.append(chunk)
            freed_grad_bytes = sum(sl.param_bytes for sl in chunk)

        # GPU phase: 3 steps per level, doubling
        chunk_size = self._init_chunk_size
        while param_cursor < len(self._all_params):
            chunk_size = min(chunk_size, self._max_chunk_size)
            for _ in range(3):
                if param_cursor >= len(self._all_params):
                    break
                needed = chunk_size * 3
                if freed_grad_bytes < needed:
                    chunk_size = freed_grad_bytes // 3
                    if chunk_size <= 0:
                        break
                chunk, param_cursor, elem_cursor = self._take_chunk_by_param_bytes(
                    param_cursor, elem_cursor, chunk_size,
                )
                if not chunk:
                    break
                schedule.append(chunk)
                freed_grad_bytes += sum(sl.param_bytes for sl in chunk)
            chunk_size = min(chunk_size * 2, self._max_chunk_size)

        return schedule

    def _take_chunk_by_param_bytes(
        self, param_cursor: int, elem_cursor: int, param_bytes_budget: int,
    ) -> tuple[list[_SliceEntry], int, int]:
        """Take slices totalling up to param_bytes_budget of param data."""
        chunk: list[_SliceEntry] = []
        used = 0

        while param_cursor < len(self._all_params) and used < param_bytes_budget:
            p = self._all_params[param_cursor]
            remaining_elems = p.numel - elem_cursor
            space_elems = (param_bytes_budget - used) // p.param_elem_size
            if space_elems <= 0:
                break

            take_elems = min(remaining_elems, space_elems)
            start = elem_cursor
            end = elem_cursor + take_elems

            chunk.append(_SliceEntry(
                param_ref=p.param,
                step=p.step, group=p.group,
                start=start, end=end,
                is_first_slice=(start == 0),
                param_bytes=take_elems * p.param_elem_size,
                backup_bytes=take_elems * p.bytes_per_elem,
                flat_param=p.flat_param,
                flat_exp_avg=p.flat_exp_avg,
                flat_exp_avg_sq=p.flat_exp_avg_sq,
            ))
            used += take_elems * p.param_elem_size

            elem_cursor += take_elems
            if elem_cursor >= p.numel:
                param_cursor += 1
                elem_cursor = 0

        return chunk, param_cursor, elem_cursor

    def _resolve_grads(self, chunk: list[_SliceEntry]):
        """Set flat_grad references for slices (needed for harvesting)."""
        for sl in chunk:
            grad_local = _get_local(sl.param_ref.grad)
            sl.flat_grad = grad_local.view(-1)

    def _harvest_grads(
        self, chunk: list[_SliceEntry], freed_segments: list[torch.Tensor],
    ):
        """Collect freed gradient segments from a completed chunk."""
        for sl in chunk:
            seg = sl.flat_grad[sl.start : sl.end].view(torch.uint8).reshape(-1)
            freed_segments.append(seg)

    def _restore_faulted_chunk(self, fault_chunk: int):
        """Restore a chunk whose adam was interrupted from its backup."""
        chunk = self._schedule[fault_chunk]

        if fault_chunk == 0:
            # CPU phase backup is in cpu_buffer
            self._restore_chunk_scatter(chunk, [self._cpu_buffer])
        else:
            # GPU phase backup is in freed grad segments of preceding chunks.
            # Reconstruct the segment list by harvesting committed chunks' grads.
            freed_grad_segments: list[torch.Tensor] = []
            for i in range(fault_chunk):
                self._resolve_grads(self._schedule[i])
                self._harvest_grads(self._schedule[i], freed_grad_segments)
            self._restore_chunk_scatter(chunk, freed_grad_segments)

    # ------------------------------------------------------------------
    # Internal: backup / restore / step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _backup_chunk_scatter(self, chunk: list[_SliceEntry], segments: list[torch.Tensor]):
        """Scatter-write chunk's (param, exp_avg, exp_avg_sq) into segments."""
        seg_idx = 0
        seg_offset = 0

        for sl in chunk:
            for flat in (sl.flat_param, sl.flat_exp_avg, sl.flat_exp_avg_sq):
                src = flat[sl.start : sl.end].view(torch.uint8).reshape(-1)
                remaining = src.numel()
                src_offset = 0

                while remaining > 0:
                    seg = segments[seg_idx]
                    avail = seg.numel() - seg_offset
                    n = min(remaining, avail)
                    seg[seg_offset : seg_offset + n].copy_(
                        src[src_offset : src_offset + n], non_blocking=True
                    )
                    src_offset += n
                    seg_offset += n
                    remaining -= n
                    if seg_offset >= seg.numel():
                        seg_idx += 1
                        seg_offset = 0

    @torch.no_grad()
    def _restore_chunk_scatter(self, chunk: list[_SliceEntry], segments: list[torch.Tensor]):
        """Scatter-read from segments back into chunk's tensors."""
        seg_idx = 0
        seg_offset = 0

        for sl in chunk:
            for flat in (sl.flat_param, sl.flat_exp_avg, sl.flat_exp_avg_sq):
                dst = flat[sl.start : sl.end].view(torch.uint8).reshape(-1)
                remaining = dst.numel()
                dst_offset = 0

                while remaining > 0:
                    seg = segments[seg_idx]
                    avail = seg.numel() - seg_offset
                    n = min(remaining, avail)
                    dst[dst_offset : dst_offset + n].copy_(
                        seg[seg_offset : seg_offset + n], non_blocking=True
                    )
                    dst_offset += n
                    seg_offset += n
                    remaining -= n
                    if seg_offset >= seg.numel():
                        seg_idx += 1
                        seg_offset = 0

    @torch.no_grad()
    def _ensure_grad_scratch(self) -> None:
        """Lazily allocate the persistent scaled-grad scratch buffer once.

        Sized to the largest single chunk's total grad bytes (chunks run one
        at a time and reuse the buffer, so one chunk's worth suffices). Grads
        exist by first-step time, so element sizes are known here. Allocated
        eagerly (never inside a CUDA-graph capture region — the step path is
        not captured), never reallocated until bind() resets it.
        """
        if self._grad_scratch is not None:
            return
        max_bytes = 0
        for chunk in self._schedule:
            b = 0
            for sl in chunk:
                g = _get_local(sl.param_ref.grad)
                b += (sl.end - sl.start) * g.element_size()
            max_bytes = max(max_bytes, b)
        self._grad_scratch = torch.empty(
            max(max_bytes, 1), dtype=torch.uint8, device=self._device
        )

    def _scaled_grad_view(
        self, grad_slice: torch.Tensor, off: int
    ) -> "tuple[torch.Tensor, int]":
        """Write ``grad_slice * clip_coef`` into scratch at ``off``; return
        (scaled_view, next_off).

        Bit-identical to the stock path's
        ``torch._foreach_mul_(grad, clip_coef_clamped)`` followed by fused
        AdamW: same elementwise multiply, same fp32 coefficient tensor, same
        out dtype — only the destination differs (scratch, not the live grad),
        so the RMP-persisted gradient is never mutated and a mid-step fault is
        replayable.
        """
        nbytes = grad_slice.numel() * grad_slice.element_size()
        buf = self._grad_scratch[off : off + nbytes].view(grad_slice.dtype)
        torch.mul(grad_slice, self._clip_coef_0d, out=buf)
        return buf, off + nbytes

    def _step_chunk(self, chunk: list[_SliceEntry]):
        """Run fused AdamW on one chunk's slices."""
        # Mirror step_counter into each first-slice's per-param state["step"]
        # via a single foreach copy on the GPU. Replaces the prior
        # `step_val = self._step_counter.item(); sl.step.fill_(step_val)`,
        # which forced a CPU↔GPU sync once per chunk. Bit-identical to the
        # old path: both end up with state["step"] holding the int step
        # counter value (cast to float32 by the dtype-mismatched copy when
        # fused-style state tensors are used).
        first_slice_steps = [sl.step for sl in chunk if sl.is_first_slice]
        if first_slice_steps:
            torch._foreach_copy_(
                first_slice_steps,
                [self._step_counter_0d] * len(first_slice_steps),
                non_blocking=True,
            )

        self._ensure_grad_scratch()
        by_group: dict[int, list[_SliceEntry]] = defaultdict(list)
        for sl in chunk:
            by_group[id(sl.group)].append(sl)

        scratch_off = 0  # reused across this chunk's groups (disjoint regions)
        for group_slices in by_group.values():
            group = group_slices[0].group
            params = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            steps = []

            for sl in group_slices:
                # Resolve grad flat view (grad tensor may change between steps)
                grad_local = _get_local(sl.param_ref.grad)
                sl.flat_grad = grad_local.view(-1)  # keep original for harvest

                # Defer gradient clipping into the recoverable region: scale
                # grad*coef into scratch (grad itself is never mutated).
                scaled, scratch_off = self._scaled_grad_view(
                    sl.flat_grad[sl.start : sl.end], scratch_off
                )

                params.append(sl.flat_param[sl.start : sl.end])
                grads.append(scaled)
                exp_avgs.append(sl.flat_exp_avg[sl.start : sl.end])
                exp_avg_sqs.append(sl.flat_exp_avg_sq[sl.start : sl.end])
                steps.append(sl.step)

            torch._fused_adamw_(
                params, grads, exp_avgs, exp_avg_sqs,
                [], steps,
                amsgrad=False,
                lr=group["lr"],
                beta1=group["betas"][0],
                beta2=group["betas"][1],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=False,
                grad_scale=None,
                found_inf=None,
            )

    # ------------------------------------------------------------------
    # Per-chunk fast-path cache (CUDA-graphed backup + lambda-cached step)
    # ------------------------------------------------------------------

    def _invalidate_chunk_caches(self) -> None:
        """Drop all per-chunk caches.  Next normal step rebuilds them."""
        self._chunk_caches = []

    def _get_chunk_cache(self, chunk_idx: int) -> _ChunkCache:
        """Lazily allocate the per-chunk cache slot."""
        while len(self._chunk_caches) <= chunk_idx:
            self._chunk_caches.append(_ChunkCache())
        return self._chunk_caches[chunk_idx]

    def _capture_backup_graph(
        self, chunk: list[_SliceEntry], segments: list[torch.Tensor],
    ) -> "tuple[torch.cuda.CUDAGraph, torch.cuda.Stream]":
        """Capture _backup_chunk_scatter into a CUDA graph and run it once.

        The body issues async copies between fixed src views (param /
        exp_avg / exp_avg_sq slices, all stable from bind()) and fixed dst
        segments (cpu_buffer for chunk 0; freed_grad_segments for later
        chunks — view objects differ across iters but data pointers are
        stable as long as the grad shard isn't reallocated).

        CUDA stream capture only RECORDS kernels; it doesn't execute them.
        We replay() once on the default stream so the first iter's backup
        actually happens (otherwise cpu_buffer / freed grad memory stays
        un-initialized and a fault on this iter would restore garbage).
        """
        capture_stream = torch.cuda.Stream(device=self._device)
        capture_stream.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=capture_stream):
            self._backup_chunk_scatter(chunk, segments)
        torch.cuda.current_stream().wait_stream(capture_stream)
        g.replay()
        return g, capture_stream

    def _build_and_run_step_lambdas(
        self, chunk: list[_SliceEntry],
    ) -> list:
        """Build lambdas that re-issue _step_chunk's kernels, and run them once.

        Why lambdas not a CUDA graph: lr changes every iter (lr_scheduler
        ticks), and the previously-attempted lr-as-tensor capture broke
        bit-identity with the standard AdamW path. Lambdas keep lr on the
        Python-float fused_adamw_ overload while still skipping the per-iter
        list-rebuild and per-group dict iteration.

        Captured in closure (stable across iters):
          - params / exp_avgs / exp_avg_sqs / steps slice views
          - group dict (so group["lr"] picks up scheduler updates per call)
          - betas / weight_decay / eps (fixed per group)
        Re-resolved each call:
          - grads — param.grad may be reallocated by zero_grad(set_to_none=True)
            + backward; we also write back sl.flat_grad so a downstream
            _harvest_grads sees the right view. The grad is then scaled by
            the clip coefficient into the per-slice scratch view (fixed
            address, captured in the closure) — never in place.
        """
        self._ensure_grad_scratch()
        lambdas: list = []

        # Step counter mirroring (per-chunk, GPU-only foreach copy).
        first_slice_steps = [sl.step for sl in chunk if sl.is_first_slice]
        if first_slice_steps:
            step_counter_0d = self._step_counter_0d
            sources = [step_counter_0d] * len(first_slice_steps)
            lambdas.append(lambda: torch._foreach_copy_(
                first_slice_steps, sources, non_blocking=True,
            ))

        by_group: dict[int, list[_SliceEntry]] = defaultdict(list)
        for sl in chunk:
            by_group[id(sl.group)].append(sl)

        scratch_off = 0  # disjoint scratch regions across this chunk's groups
        for group_slices in by_group.values():
            group = group_slices[0].group
            params = [sl.flat_param[sl.start : sl.end] for sl in group_slices]
            exp_avgs = [sl.flat_exp_avg[sl.start : sl.end] for sl in group_slices]
            exp_avg_sqs = [sl.flat_exp_avg_sq[sl.start : sl.end] for sl in group_slices]
            steps = [sl.step for sl in group_slices]
            grad_specs = [(sl, sl.start, sl.end) for sl in group_slices]
            # Pre-carve a fixed-address scratch view per slice (sizes are
            # constant across iters). The lambda re-resolves the (possibly
            # moved) grad each call but always writes the scaled result into
            # these stable buffers — efficient and CUDA-graph-safe.
            scaled_views = []
            for sl in group_slices:
                g = _get_local(sl.param_ref.grad)
                nbytes = (sl.end - sl.start) * g.element_size()
                scaled_views.append(
                    self._grad_scratch[scratch_off : scratch_off + nbytes]
                    .view(g.dtype)
                )
                scratch_off += nbytes
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            lambdas.append(_make_group_step_lambda(
                group, params, exp_avgs, exp_avg_sqs, steps, grad_specs,
                beta1, beta2, weight_decay, eps,
                scaled_views, self._clip_coef_0d,
            ))

        for lam in lambdas:
            lam()
        return lambdas
