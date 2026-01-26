from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

from torchtitan.tools.logging import logger


class SnapshotProfiler:
    def __init__(
        self,
        enable: bool = False,
        skip_first_k: int = 5,
        output_folder: str = "",
    ):
        self._enable = enable
        self._skip_first_k = skip_first_k
        self._output_folder = output_folder

        self._step_count = 0
        self._profiling_stream = torch.cuda.Stream()
        self._events = []  # List of iterations, each iteration is a list of events
        # self._mark = torch.tensor(0, device="cuda")

    def _is_profiling(self) -> bool:
        """Check if we should profile this step."""
        return self._enable and self._step_count > self._skip_first_k

    def start_step(self):
        self._step_count += 1
        if not self._is_profiling():
            return
        event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self._profiling_stream):
            event.record()
            # self._mark += 1
        self._events.append([event])

    def start_optimizer(self):
        if not self._is_profiling():
            return
        if not self._events:
            return
        event = torch.cuda.Event(enable_timing=True)
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._profiling_stream):
            self._profiling_stream.wait_stream(default_stream)
            event.record()
            # self._mark += 1
        self._events[-1].append(event)

    def maybe_profile_gap_begin(self):
        if not self._is_profiling() or not self._events:
            return
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        comm_stream = torch.cuda.current_stream()

        with torch.cuda.stream(self._profiling_stream):
            self._profiling_stream.wait_stream(comm_stream)
            start_event.record()
            # self._mark += 1
        self._events[-1].append(start_event)
        self._events[-1].append(end_event)

    def maybe_profile_gap_end(self, async_op, handle):
        if not self._is_profiling() or not self._events:
            return
        comm_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._profiling_stream):
            if async_op:
                handle.wait()
            else:
                self._profiling_stream.wait_stream(comm_stream)
            self._events[-1][-1].record()
            # self._mark += 1

    def compute_gaps(self):
        """Compute and save communication gaps to file."""
        if not self._enable or not self._events:
            return

        torch.cuda.synchronize()

        rank = dist.get_rank() if dist.is_initialized() else 0

        # Compute gaps for each iteration
        all_gaps = []
        for events in self._events:
            if len(events) < 2:
                continue
            iteration_gaps = []
            # Events are pairs: (start, end) for each communication gap
            # First event is from start_step, then pairs of (start, end) from maybe_profile_gap
            # Last event (if odd) is from start_optimizer
            for i in range(0, len(events) - 1, 2):
                start_event = events[i]
                end_event = events[i + 1]
                gap_time = start_event.elapsed_time(end_event)
                iteration_gaps.append(gap_time)
            all_gaps.append(iteration_gaps)
        num_gaps = len(all_gaps[0])
        assert all(
            len(gaps) == num_gaps for gaps in all_gaps
        ), "Inconsistent number of gaps across iterations"

        # Write to file
        if self._output_folder:
            output_dir = Path(self._output_folder)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"rank_{rank}.txt"
            content = "\n".join(
                ",".join(f"{gap:.2f}" for gap in iteration_gaps)
                for iteration_gaps in all_gaps
            )
            output_file.write_text(content)

        # Also print summary
        if rank == 0 and all_gaps:
            logger.info(f"[Gemini] Profiled {len(all_gaps)} iterations")
            logger.info(f"[Gemini] Gaps saved to {self._output_folder}/rank_*.txt")