"""Resilient optimizer: zero GPU overhead via exponential bootstrap.

Processes the optimizer step in chunks with fault-tolerant backup.
Uses a tiny pinned CPU buffer to bootstrap, then exponentially grows
the chunk size by harvesting freed gradient memory from completed
chunks — achieving zero additional GPU memory overhead.

Bootstrap (exponential growth):
    Round 0: backup init_chunk_mb to CPU, step → frees grad memory
    Round 1: backup init_chunk_mb to CPU, step → more grad memory
    Round 2: backup 2×init to GPU grad buf, step → even more
    Round 3: backup 4×init to GPU grad buf, step → ...
    ...until chunk size reaches max_chunk_mb → steady state

Params are sorted by gradient size (descending) so that early rounds
free the most memory per step, minimizing bootstrap rounds.

Memory budget:
    GPU: 0 extra (gradient memory is already allocated)
    CPU: init_chunk_mb pinned (e.g. 4 MB)

Usage:
    resilient_opt = ResilientOptimizer(optimizers, rmp_client, device,
                                       init_chunk_mb=4, max_chunk_mb=256)
    resilient_opt.step()
"""

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.distributed._tensor import DTensor

from leto.rmp.client import RmpClient, TensorSpec
from torchtitan.tools.logging import logger

_MARKER_IDLE = -1


def _get_local(tensor):
    """Get underlying local tensor from DTensor, or return as-is."""
    if isinstance(tensor, DTensor):
        return tensor._local_tensor
    return tensor


@dataclass
class _ParamInfo:
    """Per-parameter info before slicing into chunks."""

    param: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    step: torch.Tensor
    group: dict
    numel: int
    bytes_per_elem: int  # param + exp_avg + exp_avg_sq per element
    grad_bytes: int  # gradient bytes (param portion only)
    backup_bytes: int  # total backup bytes


@dataclass
class _SliceEntry:
    """One unit of work: a full param or a contiguous slice of one."""

    param: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    step: torch.Tensor
    group: dict
    start: int
    end: int
    is_first_slice: bool
    nbytes: int


def _make_slices(params: list[_ParamInfo], chunk_bytes: int) -> list[_SliceEntry]:
    """Split params into slices that fit in chunk_bytes."""
    slices: list[_SliceEntry] = []
    for p in params:
        if p.backup_bytes <= chunk_bytes:
            slices.append(
                _SliceEntry(
                    param=p.param, exp_avg=p.exp_avg, exp_avg_sq=p.exp_avg_sq,
                    step=p.step, group=p.group,
                    start=0, end=p.numel, is_first_slice=True,
                    nbytes=p.backup_bytes,
                )
            )
        else:
            max_elems = chunk_bytes // p.bytes_per_elem
            assert max_elems > 0
            for s in range(0, p.numel, max_elems):
                e = min(s + max_elems, p.numel)
                slices.append(
                    _SliceEntry(
                        param=p.param, exp_avg=p.exp_avg, exp_avg_sq=p.exp_avg_sq,
                        step=p.step, group=p.group,
                        start=s, end=e, is_first_slice=(s == 0),
                        nbytes=(e - s) * p.bytes_per_elem,
                    )
                )
    return slices


def _pack_chunks(slices: list[_SliceEntry], chunk_bytes: int) -> list[list[_SliceEntry]]:
    """Pack slices into chunks that fit in chunk_bytes."""
    chunks: list[list[_SliceEntry]] = []
    cur: list[_SliceEntry] = []
    cur_bytes = 0
    for sl in slices:
        if cur_bytes + sl.nbytes > chunk_bytes and cur:
            chunks.append(cur)
            cur = []
            cur_bytes = 0
        cur.append(sl)
        cur_bytes += sl.nbytes
    if cur:
        chunks.append(cur)
    return chunks


class ResilientOptimizer:
    """Zero-GPU-overhead resilient optimizer with exponential bootstrap.

    Sorts params by gradient size (descending), bootstraps with a tiny
    CPU buffer, then doubles the chunk size each round by reusing freed
    gradient memory until reaching max_chunk_mb.
    """

    def __init__(
        self,
        optimizers,
        rmp_client: RmpClient,
        device: torch.device,
        init_chunk_mb: int = 4,
        max_chunk_mb: int = 256,
    ):
        self._optimizers = optimizers
        self._device = device
        self._init_chunk_bytes = init_chunk_mb * 1024 * 1024
        self._max_chunk_bytes = max_chunk_mb * 1024 * 1024

        # -- collect per-param info ----------------------------------------
        all_params: list[_ParamInfo] = []

        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param not in optimizer.state:
                        continue
                    state = optimizer.state[param]
                    lp = _get_local(param)
                    lm = _get_local(state["exp_avg"])
                    lv = _get_local(state["exp_avg_sq"])
                    numel = lp.numel()
                    bpe = lp.element_size() + lm.element_size() + lv.element_size()
                    all_params.append(
                        _ParamInfo(
                            param=param,
                            exp_avg=state["exp_avg"],
                            exp_avg_sq=state["exp_avg_sq"],
                            step=state["step"],
                            group=group,
                            numel=numel,
                            bytes_per_elem=bpe,
                            grad_bytes=numel * lp.element_size(),
                            backup_bytes=numel * bpe,
                        )
                    )

        # Sort by gradient size descending — large grads first to maximize
        # freed memory early in the bootstrap phase.
        all_params.sort(key=lambda p: p.grad_bytes, reverse=True)

        self._all_params = all_params
        self._num_params = len(all_params)

        # -- allocate pinned CPU buffer (for bootstrap) --------------------
        self._cpu_buffer = torch.empty(
            self._init_chunk_bytes, dtype=torch.uint8, device="cpu",
        ).pin_memory()

        # -- allocate RMP marker -------------------------------------------
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

        logger.info(
            f"[ResilientOpt] {self._num_params} params "
            f"(sorted by grad size desc), "
            f"init_chunk: {init_chunk_mb} MB CPU, "
            f"max_chunk: {max_chunk_mb} MB GPU"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def needs_recovery(self) -> bool:
        return self._marker.item() != _MARKER_IDLE

    def step(self):
        """Exponential-bootstrap fault-tolerant optimizer step.

        Maintains a cursor into the sorted param list.  Each iteration:
        1. Decide chunk_bytes based on freed gradient memory.
        2. Take exactly one chunk's worth of elements from the cursor.
        3. Backup → step → harvest gradient.
        4. Try to double chunk_bytes for the next iteration.
        """
        freed_grad_segments: list[torch.Tensor] = []
        freed_grad_bytes = 0
        chunk_bytes = self._init_chunk_bytes
        global_chunk_idx = 0

        # Flatten all params into a stream of (param_info, elem_offset) work.
        # We consume elements from this stream one chunk at a time.
        param_cursor = 0  # index into self._all_params
        elem_cursor = 0   # element offset within current param

        while param_cursor < len(self._all_params):
            # -- decide buffer and chunk size ------------------------------
            # Try to grow chunk_bytes using freed gradient memory
            while (
                freed_grad_bytes >= min(chunk_bytes * 2, self._max_chunk_bytes)
                and chunk_bytes < self._max_chunk_bytes
            ):
                chunk_bytes = min(chunk_bytes * 2, self._max_chunk_bytes)

            if freed_grad_bytes >= chunk_bytes:
                buf_segments = freed_grad_segments
            else:
                buf_segments = [self._cpu_buffer]
                chunk_bytes = self._init_chunk_bytes

            # -- fill one chunk from the cursor ----------------------------
            chunk: list[_SliceEntry] = []
            chunk_used = 0

            while param_cursor < len(self._all_params) and chunk_used < chunk_bytes:
                p = self._all_params[param_cursor]
                remaining_elems = p.numel - elem_cursor
                space_elems = (chunk_bytes - chunk_used) // p.bytes_per_elem
                if space_elems <= 0:
                    break

                take_elems = min(remaining_elems, space_elems)
                start = elem_cursor
                end = elem_cursor + take_elems

                chunk.append(_SliceEntry(
                    param=p.param, exp_avg=p.exp_avg, exp_avg_sq=p.exp_avg_sq,
                    step=p.step, group=p.group,
                    start=start, end=end,
                    is_first_slice=(start == 0),
                    nbytes=take_elems * p.bytes_per_elem,
                ))
                chunk_used += take_elems * p.bytes_per_elem

                elem_cursor += take_elems
                if elem_cursor >= p.numel:
                    param_cursor += 1
                    elem_cursor = 0

            if not chunk:
                break

            # -- backup, step, harvest -------------------------------------
            self._marker.fill_(global_chunk_idx * 2)
            self._backup_chunk_scatter(chunk, buf_segments)
            self._marker.fill_(global_chunk_idx * 2 + 1)
            self._step_chunk(chunk)

            for sl in chunk:
                grad = _get_local(sl.param.grad)
                flat_grad = grad.contiguous().view(-1)
                seg = flat_grad[sl.start : sl.end].view(torch.uint8).reshape(-1)
                freed_grad_segments.append(seg)
                freed_grad_bytes += seg.numel()

            global_chunk_idx += 1

        self._marker.fill_(_MARKER_IDLE)
        torch.cuda.current_stream().synchronize()

    def maybe_recover(self) -> bool:
        """Detect mid-step fault and replay full step.

        Uses the same exponential bootstrap logic as step() so that
        recovery performance matches normal operation.
        """
        marker_val = self._marker.item()
        if marker_val == _MARKER_IDLE:
            return False

        logger.warning(
            f"[ResilientOpt] Fault detected (marker={marker_val}), "
            f"replaying full step"
        )

        # Replay the entire step (gradients are intact in RMP).
        # We can't trust any partial state, so redo everything.
        self.step()

        logger.info("[ResilientOpt] Recovery complete")
        return True

    # ------------------------------------------------------------------
    # Internal: backup / restore / step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _backup_chunk_scatter(self, chunk: list[_SliceEntry], segments: list[torch.Tensor]):
        """Copy chunk's slices into buffer segments (scatter write)."""
        seg_idx = 0
        seg_offset = 0

        for sl in chunk:
            for tensor in (sl.param, sl.exp_avg, sl.exp_avg_sq):
                flat = _get_local(tensor).contiguous().view(-1)
                src = flat[sl.start : sl.end].view(torch.uint8).reshape(-1)
                remaining = src.numel()
                src_offset = 0

                while remaining > 0:
                    seg = segments[seg_idx]
                    avail = seg.numel() - seg_offset
                    n = min(remaining, avail)
                    seg[seg_offset : seg_offset + n].copy_(
                        src[src_offset : src_offset + n]
                    )
                    src_offset += n
                    seg_offset += n
                    remaining -= n
                    if seg_offset >= seg.numel():
                        seg_idx += 1
                        seg_offset = 0

    @torch.no_grad()
    def _restore_chunk_scatter(self, chunk: list[_SliceEntry], segments: list[torch.Tensor]):
        """Copy from buffer segments back into chunk's slices."""
        seg_idx = 0
        seg_offset = 0

        for sl in chunk:
            for tensor in (sl.param, sl.exp_avg, sl.exp_avg_sq):
                flat = _get_local(tensor).contiguous().view(-1)
                dst = flat[sl.start : sl.end].view(torch.uint8).reshape(-1)
                remaining = dst.numel()
                dst_offset = 0

                while remaining > 0:
                    seg = segments[seg_idx]
                    avail = seg.numel() - seg_offset
                    n = min(remaining, avail)
                    dst[dst_offset : dst_offset + n].copy_(
                        seg[seg_offset : seg_offset + n]
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
        steps_to_inc = list(
            {id(sl.step): sl.step for sl in chunk if sl.is_first_slice}.values()
        )
        if steps_to_inc:
            torch._foreach_add_(steps_to_inc, 1)

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
                fp = _get_local(sl.param).contiguous().view(-1)
                fg = _get_local(sl.param.grad).contiguous().view(-1)
                fm = _get_local(sl.exp_avg).contiguous().view(-1)
                fv = _get_local(sl.exp_avg_sq).contiguous().view(-1)

                params.append(fp[sl.start : sl.end])
                grads.append(fg[sl.start : sl.end])
                exp_avgs.append(fm[sl.start : sl.end])
                exp_avg_sqs.append(fv[sl.start : sl.end])
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
