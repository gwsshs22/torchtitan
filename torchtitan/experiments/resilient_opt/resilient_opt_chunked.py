"""Resilient optimizer with fault-tolerant chunked step via RMP buffer.

Instead of shadowing all params (2x memory), uses a small RMP-persisted
buffer and processes the optimizer step in chunks.  For each chunk:

  1. Back up chunk's data (param, exp_avg, exp_avg_sq slices) into the buffer
  2. Run fused AdamW on just that chunk
  3. Advance to the next chunk

Large params that exceed the chunk size are split into intra-param slices
so the buffer size is always respected.

On fault, only the current chunk needs to be restored from the buffer
and replayed.  Previous chunks are already done; subsequent chunks are
untouched.  Extra memory = chunk_size (not total_params * 3).

Key insight: gradients are read-only input to optimizer.step() for both
Adam and AdamW.  So given the pre-step snapshot of one chunk and the
intact gradients (persisted in RMP), the chunk step is deterministic
and reproducible.

The marker lives on a GPU tensor (RMP-persisted).  All marker writes
are stream-ordered GPU ops (fill_), so no CPU–GPU sync is needed
between chunks — only one sync at the end of each step().

Backup copies + marker writes are captured into per-chunk CUDA graphs
at init time, eliminating Python-loop and kernel-launch overhead on
the backup path.

Usage:
    # After rmp_manager.maybe_init() (params + opt states on RMP):
    resilient_opt = ResilientOptimizer(optimizers, rmp_client, device)
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

# Marker encoding (stored as int64 on GPU):
#   -1           → idle (no fault)
#   chunk_idx*2  → backing up chunk_idx (buffer may be incomplete)
#   chunk_idx*2+1→ stepping chunk_idx  (buffer holds valid backup)
_MARKER_IDLE = -1


def _get_local(tensor):
    """Get underlying local tensor from DTensor, or return as-is."""
    if isinstance(tensor, DTensor):
        return tensor._local_tensor
    return tensor


@dataclass
class _SliceEntry:
    """One unit of work: a full param or a contiguous slice of one.

    All tensor fields are the *original* (unsliced) tensors.  The slice
    range [start, end) indexes into the flattened view.
    """

    param: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    step: torch.Tensor  # CPU scalar tensor (shared across slices of same param)
    group: dict  # mutable ref → optimizer param_group
    start: int  # element offset in flattened tensor
    end: int  # element end in flattened tensor
    is_first_slice: bool  # increment step only on first slice
    nbytes: int  # backup bytes for this slice (param+exp_avg+exp_avg_sq)


class ResilientOptimizer:
    """Fault-tolerant optimizer using a small RMP buffer and chunked steps.

    Recovery logic on restart (based on GPU marker value):
        * Even marker (backing up): buffer invalid, but chunk's params are
          still at pre-step values.  Re-run from that chunk.
        * Odd  marker (stepping):   restore chunk from buffer, then re-run
          from that chunk.
    """

    def __init__(
        self,
        optimizers,
        rmp_client: RmpClient,
        device: torch.device,
        chunk_size_mb: int = 32,
        use_cuda_graph: bool = True,
    ):
        self._optimizers = optimizers
        self._use_cuda_graph = use_cuda_graph
        # chunk_size_mb = param bytes to update per chunk
        # buffer needs 3x (param + exp_avg + exp_avg_sq)
        chunk_bytes = chunk_size_mb * 3 * 1024 * 1024

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
                        assert max_elems > 0, (
                            f"chunk_size_mb={chunk_size_mb} too small for even "
                            f"1 element ({bytes_per_elem} bytes/elem)"
                        )
                        for s in range(0, numel, max_elems):
                            e = min(s + max_elems, numel)
                            slice_bytes = (e - s) * bytes_per_elem
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
                                    nbytes=slice_bytes,
                                )
                            )

        self._num_params = num_params

        # -- pack slices into chunks that fit in chunk_bytes ---------------
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

        # -- allocate RMP GPU tensors: buffer + marker ---------------------
        device_idx = device.index if hasattr(device, "index") else 0
        gpu_specs = [
            TensorSpec(
                name="resilient/buffer",
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
        self._buffer = gpu_tensors["resilient/buffer"]
        self._marker = gpu_tensors["resilient/marker"]

        if gpu_allocated:
            self._marker.fill_(_MARKER_IDLE)
            torch.cuda.current_stream().synchronize()

        # -- capture per-chunk CUDA graphs for backup + marker -------------
        self._backup_graphs: list[torch.cuda.CUDAGraph] | None = None
        if use_cuda_graph and len(self._chunks) > 0:
            self._backup_graphs = self._capture_backup_graphs()

        logger.info(
            f"[ResilientOpt] {num_params} params, "
            f"{len(all_slices)} slices, "
            f"{len(self._chunks)} chunks, "
            f"buffer: {chunk_bytes / (1024**2):.1f} MB, "
            f"cuda_graph: {use_cuda_graph}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def needs_recovery(self) -> bool:
        return self._marker.item() != _MARKER_IDLE

    def step(self):
        """Chunked fault-tolerant optimizer step."""
        if self._backup_graphs is not None:
            for chunk_idx in range(len(self._chunks)):
                self._backup_graphs[chunk_idx].replay()
                self._step_chunk(chunk_idx)
        else:
            for chunk_idx in range(len(self._chunks)):
                self._marker.fill_(chunk_idx * 2)
                self._backup_chunk(chunk_idx)
                self._marker.fill_(chunk_idx * 2 + 1)
                self._step_chunk(chunk_idx)
        self._marker.fill_(_MARKER_IDLE)
        torch.cuda.current_stream().synchronize()

    def maybe_recover(self) -> bool:
        """Detect mid-step fault and replay from the faulting chunk.

        Call once on restart, after rmp_manager.maybe_init() has loaded
        params, optimizer states, and gradients from RMP.
        """
        marker_val = self._marker.item()  # GPU → CPU read
        if marker_val == _MARKER_IDLE:
            return False

        if marker_val % 2 == 0:
            start = marker_val // 2
            logger.warning(
                f"[ResilientOpt] Fault during backup of chunk {start}, "
                f"re-running from chunk {start}"
            )
        else:
            start = (marker_val - 1) // 2
            logger.warning(
                f"[ResilientOpt] Fault during step of chunk {start}, "
                f"restoring and re-running"
            )
            self._restore_chunk(start)

        # Recovery always uses eager path (graphs may not exist on restart)
        for chunk_idx in range(start, len(self._chunks)):
            self._marker.fill_(chunk_idx * 2)
            self._backup_chunk(chunk_idx)
            self._marker.fill_(chunk_idx * 2 + 1)
            self._step_chunk(chunk_idx)
        self._marker.fill_(_MARKER_IDLE)
        torch.cuda.current_stream().synchronize()
        logger.info("[ResilientOpt] Recovery complete")
        return True

    # ------------------------------------------------------------------
    # Internal: CUDA graph capture
    # ------------------------------------------------------------------

    def _capture_backup_graphs(self) -> list[torch.cuda.CUDAGraph]:
        """Capture one CUDA graph per chunk for backup + marker writes.

        Each graph contains:
          marker.fill_(chunk_idx * 2)       — mark "backing up"
          copy_ ops for all slices           — backup to buffer
          marker.fill_(chunk_idx * 2 + 1)   — mark "stepping"

        Tensor addresses are fixed (RMP), so graphs can be replayed
        every step with zero Python overhead on the backup path.
        """
        # CUDA graphs must be captured on a non-default stream
        capture_stream = torch.cuda.Stream()

        # Warmup on the capture stream
        with torch.cuda.stream(capture_stream):
            for chunk_idx in range(len(self._chunks)):
                self._marker.fill_(chunk_idx * 2)
                self._backup_chunk(chunk_idx)
                self._marker.fill_(chunk_idx * 2 + 1)
        torch.cuda.synchronize()

        # Capture
        graphs = []
        for chunk_idx in range(len(self._chunks)):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=capture_stream):
                self._marker.fill_(chunk_idx * 2)
                self._backup_chunk(chunk_idx)
                self._marker.fill_(chunk_idx * 2 + 1)
            graphs.append(g)

        logger.info(
            f"[ResilientOpt] Captured {len(graphs)} backup CUDA graphs"
        )
        return graphs

    # ------------------------------------------------------------------
    # Internal: backup / restore / step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _backup_chunk(self, chunk_idx):
        """Copy chunk's slices into the byte buffer."""
        chunk = self._chunks[chunk_idx]
        offset = 0
        for sl in chunk:
            for tensor in (sl.param, sl.exp_avg, sl.exp_avg_sq):
                flat = _get_local(tensor).contiguous().view(-1)
                data = flat[sl.start : sl.end]
                nbytes = data.numel() * data.element_size()
                self._buffer[offset : offset + nbytes].copy_(
                    data.view(torch.uint8).reshape(-1)
                )
                offset += nbytes

    @torch.no_grad()
    def _restore_chunk(self, chunk_idx):
        """Copy byte buffer back into chunk's slices."""
        chunk = self._chunks[chunk_idx]
        offset = 0
        for sl in chunk:
            for tensor in (sl.param, sl.exp_avg, sl.exp_avg_sq):
                flat = _get_local(tensor).contiguous().view(-1)
                data = flat[sl.start : sl.end]
                nbytes = data.numel() * data.element_size()
                data.view(torch.uint8).reshape(-1).copy_(
                    self._buffer[offset : offset + nbytes]
                )
                offset += nbytes

    @torch.no_grad()
    def _step_chunk(self, chunk_idx):
        """Run fused AdamW on one chunk's slices."""
        chunk = self._chunks[chunk_idx]

        # Increment step counts once per param (not per slice).
        # CPU op on CPU tensors — executes before GPU kernel launch.
        steps_to_inc = list(
            {id(sl.step): sl.step for sl in chunk if sl.is_first_slice}.values()
        )
        if steps_to_inc:
            torch._foreach_add_(steps_to_inc, 1)

        # Group slices by param_group (different groups may have different lr).
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
                [],  # max_exp_avg_sqs (no amsgrad)
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
