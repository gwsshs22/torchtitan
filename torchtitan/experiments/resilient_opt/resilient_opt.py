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
from dataclasses import dataclass

import torch
from torch.cuda._pin_memory_utils import pin_memory
from torch.distributed._tensor import DTensor

from leto.rmp.client import RmpClient, TensorSpec
from torchtitan.tools.logging import logger

_MARKER_IDLE = -1

# Hook state values stored in self._hook_state.
#   NONE     = pre-hook not yet run for the current step (or post-hook
#              completed and reset for the next step).
#   PRE_DONE = pre-hook completed; chunks may or may not have run yet.
# Faults inside hook execution itself are out of scope of the synthetic
# fault model (faults inject only at _marker.fill_); the existing real-RMP
# storage of expert_bias / tokens_per_expert handles persistence across
# crashes, but partial in-kernel hook execution is not recovered.
_HOOK_STATE_NONE = 0
_HOOK_STATE_PRE_DONE = 1


def _get_local(tensor):
    """Get underlying local tensor from DTensor, or return as-is."""
    if isinstance(tensor, DTensor):
        return tensor._local_tensor
    return tensor


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
    ):
        self._optimizers = optimizers
        self._device = device
        self._init_chunk_size = init_chunk_size_mb * 1024 * 1024
        self._max_chunk_size = max_chunk_size_mb * 1024 * 1024
        self._init_chunk_size_mb = init_chunk_size_mb
        self._max_chunk_size_mb = max_chunk_size_mb

        # Populated by bind(). step()/maybe_recover() require bind() first.
        self._all_params: list[_ParamInfo] = []
        self._num_params = 0
        self._schedule: list[list[_SliceEntry]] = []

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

        # -- Hook state tracker (separate allocation so existing deployments
        # without hook_state can roll forward without re-initing the marker).
        hook_state_tensors, hook_state_allocated = rmp_client.get_or_allocate_tensors(
            [TensorSpec(
                name="resilient/hook_state", shape=(1,),
                dtype=torch.int64, device=device_idx,
            )]
        )
        self._hook_state = hook_state_tensors["resilient/hook_state"]
        if hook_state_allocated:
            self._hook_state.fill_(_HOOK_STATE_NONE)
            torch.cuda.current_stream().synchronize()

        # -- Step counter (RMP-backed GPU) ------------------------------------
        step_cnt_tensors, step_cnt_allocated = rmp_client.get_or_allocate_tensors(
            [TensorSpec(
                name="resilient/step_counter", shape=(1,),
                dtype=torch.int64, device=device_idx,
            )]
        )
        self._step_counter = step_cnt_tensors["resilient/step_counter"]
        # Initialized from optim.state["step"] on the first bind() after a
        # fresh allocation; deferred because the source tensor lives in
        # optimizer state, which we don't traverse in __init__.
        self._step_counter_needs_init = step_cnt_allocated

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

    def step(self):
        """Fault-tolerant optimizer step with 3-step doubling.

        Order matches torch.optim.Optimizer.step():
            pre-hooks → param/state update → post-hooks.

        Hook state is recorded between phases so maybe_recover() can
        replay the right subset on resume.
        """
        self._step_counter += 1
        self._run_pre_hooks()
        # Mark AFTER pre-hook so a fault that prevents the fill_ leaves
        # state==NONE and recovery re-runs pre-hook (correct for synthetic
        # fault model where faults inject only at marker.fill_).
        self._hook_state.fill_(_HOOK_STATE_PRE_DONE)
        self._run_chunks(resume_from=0)
        self._run_post_hooks()
        self._hook_state.fill_(_HOOK_STATE_NONE)

    def _run_pre_hooks(self):
        container = self._optimizers
        if not hasattr(container, "_optimizer_step_pre_hooks"):
            return
        for hook in container._optimizer_step_pre_hooks.values():
            result = hook(container, (), {})
            if result is not None:
                raise RuntimeError(
                    "ResilientOptimizer does not support pre-hooks that "
                    "rewrite step() args/kwargs"
                )

    def _run_post_hooks(self):
        container = self._optimizers
        if not hasattr(container, "_optimizer_step_post_hooks"):
            return
        for hook in container._optimizer_step_post_hooks.values():
            hook(container, (), {})

    def _run_chunks(self, resume_from: int):
        """Execute chunks starting from resume_from.

        Chunks before resume_from are treated as committed — only their
        freed grad memory is harvested for scratch space.
        """
        freed_grad_segments: list[torch.Tensor] = []

        for chunk_idx, chunk in enumerate(self._schedule):
            if chunk_idx < resume_from:
                # Committed chunk: harvest freed grad memory for scratch space.
                self._resolve_grads(chunk)
                self._harvest_grads(chunk, freed_grad_segments)
                continue

            backup_dest = (
                [self._cpu_buffer] if chunk_idx == 0 else freed_grad_segments
            )

            self._marker.fill_(chunk_idx * 2)
            self._backup_chunk_scatter(chunk, backup_dest)
            self._marker.fill_(chunk_idx * 2 + 1)
            self._step_chunk(chunk)
            self._harvest_grads(chunk, freed_grad_segments)

        self._marker.fill_(_MARKER_IDLE)

    def maybe_recover(self, resume_step: int) -> bool:
        """Detect mid-step fault and resume from the interrupted phase.

        Args:
            resume_step: the step number this recovery should produce.
                Must equal step_counter or step_counter + 1.

        Assumes optimizer states, gradients, and hook-mutated buffers
        (e.g. expert_bias, tokens_per_expert) persist in RMP across faults.
        Committed chunks and completed hooks are not re-applied.

        Post-hook is assumed idempotent (current users are
        _model_converters.post_optimizer_hook, no-op for bf16 / safe to
        re-run for mxfp8 weight reconversion); it is re-invoked on any
        recovery that produces step `resume_step` to keep the path simple.
        Pre-hook (e.g. _update_expert_bias) is non-idempotent and is
        invoked only when hook_state confirms it has not yet run for the
        current step.
        """
        stored = self._step_counter.item()
        marker_val = self._marker.item()
        hook_state = self._hook_state.item()

        if stored + 1 == resume_step:
            # Counter hasn't been bumped → fresh step. If a previous step's
            # post-hook completed but the trailing hook_state.fill_(NONE)
            # never landed, hook_state is left at PRE_DONE; reset it so the
            # new step's pre-hook will run.
            assert marker_val == _MARKER_IDLE
            if hook_state == _HOOK_STATE_PRE_DONE:
                self._hook_state.fill_(_HOOK_STATE_NONE)
            self.step()
            logger.info(f"[ResilientOpt] Full step executed stored={stored}")
            return True

        # stored == resume_step: counter already bumped — step is in flight.
        assert stored == resume_step, (
            f"step counter mismatch: stored={stored}, resume_step={resume_step}"
        )

        params_step = int(self._all_params[0].step.item())

        if marker_val != _MARKER_IDLE:
            # Fault during chunk processing. Pre-hook must have run before
            # any chunk could have started touching _marker.
            assert hook_state == _HOOK_STATE_PRE_DONE, (
                f"chunk marker non-idle but hook_state={hook_state}"
            )
            fault_chunk = marker_val // 2
            backup_done = (marker_val % 2 == 1)
            logger.warning(
                f"[ResilientOpt] Fault detected (marker={marker_val}, "
                f"chunk={fault_chunk}, backup_done={backup_done}), recovering"
            )
            if backup_done:
                self._restore_faulted_chunk(fault_chunk)
            self._run_chunks(resume_from=fault_chunk)
            self._run_post_hooks()
            self._hook_state.fill_(_HOOK_STATE_NONE)
            logger.info(f"[ResilientOpt] Recovery complete stored={stored}")
            return True

        # marker IDLE — either step fully done, or counter bumped but
        # chunks haven't started, or chunks completed but post-hook didn't
        # run / didn't reset hook_state.
        if hook_state == _HOOK_STATE_NONE:
            if params_step == resume_step:
                # pre-hook ran, chunks ran, post-hook ran, hook_state reset:
                # full step is on disk. Nothing to do.
                logger.info(f"[ResilientOpt] Step fully complete, no recovery needed stored={stored}")
                return False
            # Counter bumped but pre-hook never landed (e.g. fault between
            # the counter increment and pre-hook execution). Run the full
            # post-bump pipeline.
            self._run_pre_hooks()
            self._hook_state.fill_(_HOOK_STATE_PRE_DONE)
            self._run_chunks(resume_from=0)
            self._run_post_hooks()
            self._hook_state.fill_(_HOOK_STATE_NONE)
            logger.info(f"[ResilientOpt] Full step executed stored={stored}")
            return True

        # hook_state == _HOOK_STATE_PRE_DONE
        if params_step == resume_step:
            # Chunks finished. Post-hook may or may not have run; idempotent
            # so re-running is safe.
            logger.info(
                "[ResilientOpt] Resuming after chunks complete: re-running post-hook"
            )
            self._run_post_hooks()
            self._hook_state.fill_(_HOOK_STATE_NONE)
            return True

        # Pre-hook done, chunks not yet started.
        self._run_chunks(resume_from=0)
        self._run_post_hooks()
        self._hook_state.fill_(_HOOK_STATE_NONE)
        logger.info("[ResilientOpt] Resumed from chunk 0 after pre-hook")
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
    def _step_chunk(self, chunk: list[_SliceEntry]):
        """Run fused AdamW on one chunk's slices."""
        step_val = self._step_counter.item()
        for sl in chunk:
            if sl.is_first_slice:
                sl.step.fill_(step_val)

        by_group: dict[int, list[_SliceEntry]] = defaultdict(list)
        for sl in chunk:
            by_group[id(sl.group)].append(sl)

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
                sl.flat_grad = grad_local.view(-1)

                params.append(sl.flat_param[sl.start : sl.end])
                grads.append(sl.flat_grad[sl.start : sl.end])
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
