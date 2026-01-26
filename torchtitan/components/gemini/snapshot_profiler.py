import json
from pathlib import Path

import torch
import torch.distributed as dist

from torchtitan.tools.logging import logger


class SnapshotProfiler:
    """Profiles communication gaps for Gemini-style checkpointing.

    Records timing of FSDP all-gather/reduce-scatter operations to identify
    gaps where checkpoint traffic can be interleaved.

    Only records gap times - strategy computation is done by SnapshotExecutor
    which has access to actual tensor sizes and can broadcast the strategy.

    Each FSDP group writes its own gap times file (by FSDP rank 0).
    """

    def __init__(
        self,
        enable: bool = False,
        skip_first_k: int = 5,
        output_folder: str = "",
        fsdp_process_group: dist.ProcessGroup | None = None,
    ):
        self._enable = enable
        self._skip_first_k = skip_first_k
        self._output_folder = output_folder

        # FSDP group info
        self._fsdp_pg = fsdp_process_group

        # Profiling state
        self._step_count = 0
        self._profiling_stream = torch.cuda.Stream()
        self._events: list[list[torch.cuda.Event]] = []

    def _is_profiling(self) -> bool:
        """Check if we should profile this step."""
        return self._enable and self._step_count > self._skip_first_k

    def _get_fsdp_rank_info(self) -> tuple[int, int, int]:
        """Get FSDP group rank info.

        Returns:
            Tuple of (global_rank, fsdp_rank, fsdp_size)
        """
        global_rank = dist.get_rank() if dist.is_initialized() else 0

        if self._fsdp_pg is not None:
            fsdp_rank = dist.get_rank(self._fsdp_pg)
            fsdp_size = dist.get_world_size(self._fsdp_pg)
        else:
            fsdp_rank = global_rank
            fsdp_size = dist.get_world_size() if dist.is_initialized() else 1

        return global_rank, fsdp_rank, fsdp_size

    def begin_step(self):
        self._step_count += 1
        if not self._is_profiling():
            return
        event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self._profiling_stream):
            event.record()
        self._events.append([event])

    def begin_optimizer(self):
        if not self._is_profiling():
            return
        if not self._events:
            return
        event = torch.cuda.Event(enable_timing=True)
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._profiling_stream):
            self._profiling_stream.wait_stream(default_stream)
            event.record()
        self._events[-1].append(event)

    def begin_collective(self):
        if not self._is_profiling() or not self._events:
            return
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        comm_stream = torch.cuda.current_stream()

        with torch.cuda.stream(self._profiling_stream):
            self._profiling_stream.wait_stream(comm_stream)
            start_event.record()
        self._events[-1].append(start_event)
        self._events[-1].append(end_event)

    def end_collective(self, async_op, handle):
        if not self._is_profiling() or not self._events:
            return
        comm_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._profiling_stream):
            if async_op:
                handle.wait()
            else:
                self._profiling_stream.wait_stream(comm_stream)
            self._events[-1][-1].record()

    def compute_gaps(self):
        """Compute and save communication gaps.

        Only saves gap times - strategy computation is done by SnapshotExecutor
        at runtime using actual block sizes.

        Each FSDP group's rank 0 writes gap times to comm_gaps_rank_{global_rank}.json
        """
        if not self._enable or not self._events:
            return

        # Check if process group is still initialized (may be called from __del__)
        if not dist.is_initialized():
            return

        torch.cuda.synchronize()

        global_rank, fsdp_rank, fsdp_size = self._get_fsdp_rank_info()

        # Compute gaps for each iteration
        all_gaps: list[list[float]] = []
        for events in self._events:
            if len(events) < 2:
                continue
            iteration_gaps = []
            for i in range(0, len(events) - 1, 2):
                start_event = events[i]
                end_event = events[i + 1]
                gap_time = start_event.elapsed_time(end_event)
                iteration_gaps.append(gap_time)
            all_gaps.append(iteration_gaps)

        if not all_gaps:
            return

        num_gaps = len(all_gaps[0])
        assert all(
            len(gaps) == num_gaps for gaps in all_gaps
        ), "Inconsistent number of gaps across iterations"

        # Compute minimum gap time for each position (conservative)
        min_gaps: dict[int, float] = {}
        for gap_id in range(num_gaps):
            min_gaps[gap_id] = min(gaps[gap_id] for gaps in all_gaps)

        # Write raw gaps to file (all ranks for debugging)
        if self._output_folder:
            output_dir = Path(self._output_folder)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"rank_{global_rank}.txt"
            content = "\n".join(
                ",".join(f"{gap:.2f}" for gap in iteration_gaps)
                for iteration_gaps in all_gaps
            )
            output_file.write_text(content)

        # Write gap times JSON (only FSDP rank 0 in each group)
        if fsdp_rank == 0:
            gap_data = {
                "gap_times_ms": {str(k): v for k, v in min_gaps.items()},
                "num_gaps": num_gaps,
                "num_iterations_profiled": len(all_gaps),
                "fsdp_size": fsdp_size,
            }

            if self._output_folder:
                output_dir = Path(self._output_folder)
                gap_file = output_dir / f"comm_gaps_rank_{global_rank}.json"
                with open(gap_file, "w") as f:
                    json.dump(gap_data, f, indent=2)

            logger.info(f"[Gemini] Rank {global_rank} (FSDP rank 0): Profiled {num_gaps} gaps")
            logger.info(f"[Gemini] Gaps saved to {self._output_folder}/comm_gaps_rank_{global_rank}.json")
