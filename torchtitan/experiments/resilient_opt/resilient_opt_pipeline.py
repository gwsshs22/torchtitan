"""Resilient optimizer (pipeline): double-buffered pipeline with copy/compute overlap.

Uses two RMP buffers (ping-pong) and a dedicated copy stream so that
backing up chunk[i+1] overlaps with stepping chunk[i] on the compute
stream.  This hides most of the backup copy latency behind compute.

Pipeline:

    copy_stream:    backup(0)─┐  backup(1)──┐  backup(2)──┐
                              │             │             │
    compute_stream:           └►step(0)     └►step(1)     └►step(2) ...

Marker lives on a GPU tensor, written only from the compute stream:

    i*2   → chunk i-1 done, backing up chunk i (buffer may be incomplete)
    i*2+1 → stepping chunk i (buffer[i%2] holds valid backup)
    -1    → idle

Usage:
    resilient_opt = ResilientOptimizerV2(optimizers, rmp_client, device)
    recovered = resilient_opt.maybe_recover()

    # In training loop (replaces optimizers.step()):
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


class ResilientOptimizerV2:
    """Double-buffered resilient optimizer with copy/compute overlap."""

    def __init__(
        self,
        optimizers,
        rmp_client: RmpClient,
        device: torch.device,
        chunk_size_mb: int = 32,
    ):
        self._optimizers = optimizers
        chunk_bytes = chunk_size_mb * 1024 * 1024

        # -- collect slice entries, splitting large params ------------------
        all_slices: list[_SliceEntry] = []
        num_params = 0

        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param not in optimizer.state:
                        continue
                    state = optimizer.state[param]
                    num_params += 1

                    lp = _get_local(param)
                    lm = _get_local(state["exp_avg"])
                    lv = _get_local(state["exp_avg_sq"])

                    numel = lp.numel()
                    bytes_per_elem = (
                        lp.element_size() + lm.element_size() + lv.element_size()
                    )
                    total_bytes = numel * bytes_per_elem

                    if total_bytes <= chunk_bytes:
                        all_slices.append(
                            _SliceEntry(
                                param=param,
                                exp_avg=state["exp_avg"],
                                exp_avg_sq=state["exp_avg_sq"],
                                step=state["step"],
                                group=group,
                                start=0,
                                end=numel,
                                is_first_slice=True,
                                nbytes=total_bytes,
                            )
                        )
                    else:
                        max_elems = chunk_bytes // bytes_per_elem
                        assert max_elems > 0
                        for s in range(0, numel, max_elems):
                            e = min(s + max_elems, numel)
                            all_slices.append(
                                _SliceEntry(
                                    param=param,
                                    exp_avg=state["exp_avg"],
                                    exp_avg_sq=state["exp_avg_sq"],
                                    step=state["step"],
                                    group=group,
                                    start=s,
                                    end=e,
                                    is_first_slice=(s == 0),
                                    nbytes=(e - s) * bytes_per_elem,
                                )
                            )

        self._num_params = num_params

        # -- pack slices into chunks ---------------------------------------
        self._chunks: list[list[_SliceEntry]] = []
        cur_chunk: list[_SliceEntry] = []
        cur_bytes = 0
        for sl in all_slices:
            if cur_bytes + sl.nbytes > chunk_bytes and cur_chunk:
                self._chunks.append(cur_chunk)
                cur_chunk = []
                cur_bytes = 0
            cur_chunk.append(sl)
            cur_bytes += sl.nbytes
        if cur_chunk:
            self._chunks.append(cur_chunk)

        # -- allocate RMP GPU tensors: 2 buffers + marker ------------------
        device_idx = device.index if hasattr(device, "index") else 0
        gpu_specs = [
            TensorSpec(
                name="resilient/buffer_0",
                shape=(chunk_bytes,),
                dtype=torch.uint8,
                device=device_idx,
            ),
            TensorSpec(
                name="resilient/buffer_1",
                shape=(chunk_bytes,),
                dtype=torch.uint8,
                device=device_idx,
            ),
            TensorSpec(
                name="resilient/marker",
                shape=(1,),
                dtype=torch.int64,
                device=device_idx,
            ),
        ]
        gpu_tensors, gpu_allocated = rmp_client.get_or_allocate_tensors(gpu_specs)
        self._buffers = [
            gpu_tensors["resilient/buffer_0"],
            gpu_tensors["resilient/buffer_1"],
        ]
        self._marker = gpu_tensors["resilient/marker"]

        if gpu_allocated:
            self._marker.fill_(_MARKER_IDLE)
            torch.cuda.current_stream().synchronize()

        # -- dedicated copy stream -----------------------------------------
        self._copy_stream = torch.cuda.Stream()

        logger.info(
            f"[ResilientOptV2] {num_params} params, "
            f"{len(all_slices)} slices, "
            f"{len(self._chunks)} chunks, "
            f"buffer: 2×{chunk_bytes / (1024**2):.1f} MB"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def needs_recovery(self) -> bool:
        return self._marker.item() != _MARKER_IDLE

    def step(self):
        """Pipelined fault-tolerant optimizer step.

        Overlaps backup(i+1) on copy_stream with step(i) on compute_stream.
        Marker is written only from compute_stream to avoid races.
        """
        n = len(self._chunks)
        if n == 0:
            return

        compute_stream = torch.cuda.current_stream()
        copy_stream = self._copy_stream

        # Kick off backup of chunk 0 on copy stream
        backup_ready = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            self._backup_chunk(0, buf_idx=0)
            backup_ready.record()

        for i in range(n):
            # Wait for current chunk's backup to land
            compute_stream.wait_event(backup_ready)

            # Mark stepping and run optimizer on compute stream
            self._marker.fill_(i * 2 + 1)
            self._step_chunk(i)

            if i + 1 < n:
                # Mark chunk i done / chunk i+1 backing up
                self._marker.fill_((i + 1) * 2)

                # Overlap: start backup of next chunk on copy stream
                backup_ready = torch.cuda.Event()
                with torch.cuda.stream(copy_stream):
                    self._backup_chunk(i + 1, buf_idx=(i + 1) % 2)
                    backup_ready.record()

        self._marker.fill_(_MARKER_IDLE)
        torch.cuda.current_stream().synchronize()

    def maybe_recover(self) -> bool:
        """Detect mid-step fault and replay (non-pipelined recovery)."""
        marker_val = self._marker.item()
        if marker_val == _MARKER_IDLE:
            return False

        if marker_val % 2 == 0:
            start = marker_val // 2
            logger.warning(
                f"[ResilientOptV2] Fault during backup of chunk {start}, "
                f"re-running from chunk {start}"
            )
        else:
            start = (marker_val - 1) // 2
            buf_idx = start % 2
            logger.warning(
                f"[ResilientOptV2] Fault during step of chunk {start}, "
                f"restoring from buffer {buf_idx} and re-running"
            )
            self._restore_chunk(start, buf_idx=buf_idx)

        # Recovery: sequential (no overlap) for simplicity
        for chunk_idx in range(start, len(self._chunks)):
            self._backup_chunk(chunk_idx, buf_idx=chunk_idx % 2)
            self._marker.fill_(chunk_idx * 2 + 1)
            self._step_chunk(chunk_idx)
        self._marker.fill_(_MARKER_IDLE)
        torch.cuda.current_stream().synchronize()
        logger.info("[ResilientOptV2] Recovery complete")
        return True

    # ------------------------------------------------------------------
    # Internal: backup / restore / step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _backup_chunk(self, chunk_idx, buf_idx):
        """Copy chunk's slices into buffer[buf_idx]."""
        chunk = self._chunks[chunk_idx]
        buf = self._buffers[buf_idx]
        offset = 0
        for sl in chunk:
            for tensor in (sl.param, sl.exp_avg, sl.exp_avg_sq):
                flat = _get_local(tensor).contiguous().view(-1)
                data = flat[sl.start : sl.end]
                nbytes = data.numel() * data.element_size()
                buf[offset : offset + nbytes].copy_(
                    data.view(torch.uint8).reshape(-1)
                )
                offset += nbytes

    @torch.no_grad()
    def _restore_chunk(self, chunk_idx, buf_idx):
        """Copy buffer[buf_idx] back into chunk's slices."""
        chunk = self._chunks[chunk_idx]
        buf = self._buffers[buf_idx]
        offset = 0
        for sl in chunk:
            for tensor in (sl.param, sl.exp_avg, sl.exp_avg_sq):
                flat = _get_local(tensor).contiguous().view(-1)
                data = flat[sl.start : sl.end]
                nbytes = data.numel() * data.element_size()
                data.view(torch.uint8).reshape(-1).copy_(
                    buf[offset : offset + nbytes]
                )
                offset += nbytes

    @torch.no_grad()
    def _step_chunk(self, chunk_idx):
        """Run fused AdamW on one chunk's slices."""
        chunk = self._chunks[chunk_idx]

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
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                [],
                steps,
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
