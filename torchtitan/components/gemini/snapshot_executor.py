import json
import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor

from torchtitan.components.gemini.in_mem_state import InMemState
from torchtitan.components.gemini.snapshot_container import SnapshotContainer
from torchtitan.components.gemini.snapshot_group import (
    SnapshotGroup,
    CheckpointLoadAction
)
from torchtitan.components.gemini.snapshot_strategy import get_snapshot_strategy
from torchtitan.components.gemini.utils import InMemStateType
from torchtitan.tools.logging import logger


try:
    from leto.launch.worker_controller_client import (
        report_duration,
        get_checkpoint_loading_type,
        get_faulty_ranks,
        DURATION_CHECKPOINT_LOADING,
        DURATION_CHECKPOINT_ALLOC,
    )
    _LETO_AVAILABLE = True
except ImportError:
    _LETO_AVAILABLE = False

# InMemState types
LOCAL = 0
REMOTE = 1

# Double-buffer indices
CURR = 0
PREV = 1


def read_checkpoint_metadata(metadata_path: str) -> list:
    """Read a rank's checkpoint metadata; returns the index-aligned
    version_steps list ([] when absent).

    Tolerates a corrupt/truncated file (a container killed mid-PERSIST):
    a broken claim degrades this rank to "no checkpoint" — its pair still
    covers the step — instead of crashing every recovery attempt.
    """
    if not os.path.exists(metadata_path):
        return []
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        version_steps = metadata["version_steps"]
    except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
        logger.error(
            f"[Gemini] Corrupt checkpoint metadata at {metadata_path} "
            f"({e!r}); treating as no checkpoint on this rank"
        )
        return []
    return version_steps if isinstance(version_steps, list) else []


def discard_checkpoint_files(mem_fs_folder: str, global_rank: int) -> list:
    """Withdraw this rank's checkpoint claim: remove its metadata (first,
    so the claim disappears before the payloads) and version files.

    Called only on the election's fresh-start fallback: the surviving dumps
    belong to a superseded world, and leaving them behind would let a LATER
    fault elect a stale step against post-restart data. Returns the removed
    file names.
    """
    names = [f"rank_{global_rank}_metadata.json"]
    for version in (0, 1):
        for locality in ("local", "remote"):
            names.append(f"rank_{global_rank}_v{version}_{locality}.pt")
    removed = []
    for name in names:
        path = os.path.join(mem_fs_folder, name)
        if os.path.exists(path):
            try:
                os.remove(path)
                removed.append(name)
            except OSError as e:
                logger.warning(f"[Gemini] Failed to remove stale {path}: {e}")
    return removed

class SnapshotExecutor:

    def __init__(
        self,
        enable: bool,
        comm_gaps_folder: str,
        block_size: int = 4 * 1024 * 1024,  # 4M elements default
        mem_fs_folder: str = "",
        # Strategy computation parameters
        bandwidth_gbps: float = 50.0,  # Per-GPU Network bandwidth in Gbps
        gap_threshold_ms: float = 3.0,  # Minimum gap to consider (ms)
        min_p2p_time_ms: float = 0.2,
    ):
        self.enable = enable
        if not self.enable:
            return

        assert mem_fs_folder != ""
        self.block_size = block_size
        self.comm_gaps_folder = comm_gaps_folder
        self.mem_fs_folder = mem_fs_folder
        os.makedirs(self.mem_fs_folder, exist_ok=True)

        # Strategy parameters
        self._bandwidth_gbps = bandwidth_gbps
        self._gap_threshold_ms = gap_threshold_ms
        self._min_p2p_time_ms = min_p2p_time_ms

        self._curr_version = CURR
        self._cur_block_id = 0
        self._comm_gap_id = 0

        self._snapshot_strategy: dict[int, int] = {}
        self._have_strategy = False
        self._gpu_buffer_id = 0

        # Track if we're in a snapshot step
        self._is_snapshot_step = False
        self._snapshot_future: Future | None = None

        self._global_rank = int(os.environ["RANK"])
        self._metadata_path = os.path.join(self.mem_fs_folder, f"rank_{self._global_rank}_metadata.json")
        self.tmp_checkpoint_path = os.path.join(self.mem_fs_folder, f"rank_{self._global_rank}_tmp.pt")
        self.container_log_dir = os.path.join(self.mem_fs_folder, "logs")

        self.snapshot_container = SnapshotContainer(
            self.mem_fs_folder,
            self.container_log_dir,
            self._global_rank,
        )

        self._snapshot_thread_pool = ThreadPoolExecutor(max_workers=1)
        self._snapshot_thread_pool.submit(lambda: None).result()

    def lazy_init(
        self,
        states: dict,
        model_wrapper,
        optimizers,
        pp_process_group: dist.ProcessGroup | None = None,
        tp_process_group: dist.ProcessGroup | None = None,
        fsdp_process_group: dist.ProcessGroup | None = None,
    ) -> None:
        if not self.enable:
            return

        self.model_wrapper = model_wrapper
        self.optimizers = optimizers
        self.states = states
        self.pp_process_group = pp_process_group
        self.tp_process_group = tp_process_group

        self._fsdp_pg = fsdp_process_group
        self.snapshot_group = SnapshotGroup(self._fsdp_pg)

        sample_tensor = next(iter(model_wrapper.state_dict().values()))
        self._dtype_size = sample_tensor.element_size()

        # [LOCAL/REMOTE][CURR/PREV]
        self.in_mem_states: list[list[InMemState]] = [
            [
                InMemState(
                    state_id,
                    model_wrapper,
                    optimizers,
                    states,
                    state_type,
                    self.snapshot_container,
                    self.mem_fs_folder,
                ) for state_id in range(2)
            ] for state_type in [InMemStateType.LOCAL, InMemStateType.REMOTE]
        ]

        # Allocate as uint8 byte buffers for dtype-agnostic P2P transfer.
        # block_size is in elements but we need bytes for the largest possible block.
        max_block_bytes = self.block_size * sample_tensor.element_size()
        self._gpu_buffers = [
            torch.zeros(max_block_bytes, dtype=torch.uint8, device="cuda"),
            torch.zeros(max_block_bytes, dtype=torch.uint8, device="cuda"),
        ]

        self._local_copy_stream = torch.cuda.Stream()
        self._copy_stream = torch.cuda.Stream()
        self._p2p_stream = torch.cuda.Stream()
        self._local_copy_event = torch.cuda.Event()  # Reusable event for local GPU→CPU copy
        # One event per double-buffer slot: block N+2 (same buffer as N) waits for N's copy
        self._p2p_copy_events = [torch.cuda.Event(), torch.cuda.Event()]
        self._p2p_copy_event_recorded = [False, False]

        # Distributed setup - compute ranks within FSDP group
        self._peer_global_rank = self.snapshot_group.peer_global_rank

        self.rank = self.snapshot_group._global_rank
        self.model_wrapper.reset_cached_state_dict()

        t0 = time.monotonic()
        self.local_curr.init_cpu_tensors()
        logger.info(f"[Gemini R{self.rank}] local_curr.init_cpu_tensors: {time.monotonic() - t0:.3f}s")

        t0 = time.monotonic()
        self.local_prev.init_cpu_tensors()
        logger.info(f"[Gemini Load R{self.rank}] local_prev.init_cpu_tensors: {time.monotonic() - t0:.3f}s")

        t0 = time.monotonic()
        self.remote_curr.init_cpu_tensors()
        logger.info(f"[Gemini Load R{self.rank}] remote_curr.init_cpu_tensors: {time.monotonic() - t0:.3f}s")

        t0 = time.monotonic()
        self.remote_prev.init_cpu_tensors()
        logger.info(f"[Gemini Load R{self.rank}] remote_prev.init_cpu_tensors: {time.monotonic() - t0:.3f}s")

        t0 = time.monotonic()
        self._gpu_blocks = self.remote_curr.compute_tensor_blocks(self.block_size, return_gpu_blocks=True)
        self.remote_prev.compute_tensor_blocks(self.block_size)
        logger.info(f"[Gemini Load R{self.rank}] compute_tensor_blocks: {time.monotonic() - t0:.3f}s")

        self._total_blocks = len(self._gpu_blocks)
        self._block_sizes = [t.numel() for t in self._gpu_blocks]

        t0 = time.monotonic()
        self._load_gaps_and_compute_strategy()
        logger.info(f"[Gemini Load R{self.rank}] _load_gaps_and_compute_strategy: {time.monotonic() - t0:.3f}s")

        t0 = time.monotonic()
        self._init_sendrecv() # Warmup
        logger.info(f"[Gemini Load R{self.rank}] _init_sendrecv: {time.monotonic() - t0:.3f}s")

    @property
    def local_curr(self) -> InMemState:
        return self.in_mem_states[LOCAL][self._curr_version]

    @property
    def local_prev(self) -> InMemState:
        return self.in_mem_states[LOCAL][1 - self._curr_version]

    @property
    def remote_curr(self) -> InMemState:
        return self.in_mem_states[REMOTE][self._curr_version]

    @property
    def remote_prev(self) -> InMemState:
        return self.in_mem_states[REMOTE][1 - self._curr_version]

    def _version_path(self, version: int, locality: str) -> str:
        return os.path.join(self.mem_fs_folder, f"rank_{self._global_rank}_v{version}_{locality}.pt")

    def _read_checkpoint_metadata(self) -> list:
        """Read checkpoint metadata. Returns list of committed steps."""
        return read_checkpoint_metadata(self._metadata_path)

    def _check_has_checkpoint(self) -> bool:
        return os.path.exists(self._metadata_path)

    def _swap_buffers(self):
        self._curr_version = 1 - self._curr_version

    def _reset_for_new_step(self):
        self._cur_block_id = 0
        self._comm_gap_id = 0
        self._swap_buffers()

    def load(self, step) -> bool:
        if not self.enable:
            return False

        # Re-check for checkpoint files (may have appeared since init,
        # e.g. standby activated after active's SnapshotContainer dumped state)
        self.has_checkpoint = self._check_has_checkpoint()


        # --- ckpt_loading: load checkpoint from mem_fs ---
        loading_start = time.monotonic()
        t0 = time.monotonic()
        loaded = self._load_snapshot()
        logger.info(f"[Gemini Load R{self.rank}] _load_snapshot: {time.monotonic() - t0:.3f}s (loaded={loaded})")
        if not loaded:
            # Manually reset .step values in the optimizer states if not loaded.
            for k, v in self.optimizers.state_dict().items():
                if k.endswith(".step") and isinstance(v, torch.Tensor):
                    assert v.numel() == 1, f"Expected .step to be a single scalar tensor, got {v.shape}"
                    v.zero_()

        loading_duration = time.monotonic() - loading_start
        if _LETO_AVAILABLE:
            report_duration(DURATION_CHECKPOINT_LOADING, loading_duration,
                            checkpoint_loading_type=get_checkpoint_loading_type())

        logger.info(f"[Gemini Load R{self.rank}] total load time: {loading_duration:.3f}s")
        return loaded

    def _load_snapshot(self):
        rank = self.snapshot_group._global_rank

        # Read local metadata to find available versions/steps
        # version_steps is index-aligned: list[version] = step or None
        version_steps = self._read_checkpoint_metadata()
        available_steps = set(s for s in version_steps if s is not None) if self.has_checkpoint else set()
        version_for_step = {step: version for version, step in enumerate(version_steps) if step is not None}

        # Find consistent step across all ranks via all-gather. The election
        # chooses the step; when leto passed a faulty set, roles are
        # faulty-rank-driven (plan §3.7 retrofit) with the election as a
        # cross-check.
        faulty_ranks = get_faulty_ranks() if _LETO_AVAILABLE else []
        t0 = time.monotonic()
        target_step, load_action = self.snapshot_group.find_consistent_step_and_action(
            available_steps, faulty_ranks=faulty_ranks
        )
        logger.info(
            f"[Gemini Load R{rank}] find_consistent_step_and_action: {time.monotonic() - t0:.3f}s, "
            f"target_step={target_step}, action={load_action}, available={version_steps}, "
            f"faulty_ranks={faulty_ranks}"
        )

        if load_action == CheckpointLoadAction.NONE:
            if available_steps:
                # The election fell back to a fresh start (some pair lost
                # both replicas) while this rank still holds dumps from the
                # superseded world. Discard them so a later fault cannot
                # elect a stale step against post-restart training state.
                removed = discard_checkpoint_files(
                    self.mem_fs_folder, self._global_rank
                )
                logger.error(
                    f"[Gemini Load R{rank}] Fresh-start fallback: discarded "
                    f"stale checkpoint files {removed} "
                    f"(had steps {sorted(available_steps)})"
                )
            return False

        if load_action in (CheckpointLoadAction.LOCAL, CheckpointLoadAction.SEND):
            version = version_for_step[target_step]
            if load_action == CheckpointLoadAction.SEND:
                remote_path = self._version_path(version, "remote")
                t0 = time.monotonic()
                self.snapshot_group.send_checkpoint(remote_path)
                logger.info(f"[Gemini Load R{rank}] send_checkpoint: {time.monotonic() - t0:.3f}s")

            local_path = self._version_path(version, "local")
            t0 = time.monotonic()
            ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
            logger.info(f"[Gemini Load R{rank}] torch.load ({load_action.name}): {time.monotonic() - t0:.3f}s")

            t0 = time.monotonic()
            self.local_curr.load_state_dict(ckpt)
            logger.info(f"[Gemini Load R{rank}] load_state_dict ({load_action.name}): {time.monotonic() - t0:.3f}s")
        elif load_action == CheckpointLoadAction.RECV:
            t0 = time.monotonic()
            ckpt = self.snapshot_group.recv_checkpoint(self.tmp_checkpoint_path)
            logger.info(f"[Gemini Load R{rank}] recv_checkpoint: {time.monotonic() - t0:.3f}s")

            t0 = time.monotonic()
            self.local_curr.load_state_dict(ckpt)
            logger.info(f"[Gemini Load R{rank}] load_state_dict (RECV): {time.monotonic() - t0:.3f}s")

        loaded_step = self.states["train_state"].step
        logger.info(f"[Gemini Load R{rank}] Loaded checkpoint at step {loaded_step}, action={load_action}")
        return True

    def _load_gaps_and_compute_strategy(self):
        strategy = None

        # FSDP rank 0 loads gaps and computes strategy
        gap_times = self.snapshot_group.load_gaps(self.comm_gaps_folder)
        if gap_times is not None:
            # Compute strategy using actual block sizes
            strategy = get_snapshot_strategy(
                gap_times,
                self._block_sizes,
                self._dtype_size,
                self._bandwidth_gbps,
                self._gap_threshold_ms,
                self._min_p2p_time_ms,
            )

            logger.info(
                f"[Gemini] Rank {self.snapshot_group.global_rank} (FSDP rank 0): Computed strategy with "
                f"{len(strategy)} gaps, {len(self._block_sizes)} blocks"
            )
            logger.info(f"[Gemini] Strategy: {strategy}")

        strategy = self.snapshot_group.broadcast_strategy(strategy)
        if strategy is not None:
            self._snapshot_strategy = strategy
            self._have_strategy = True

    def _init_sendrecv(self):
        with torch.cuda.stream(self._p2p_stream):
            self.snapshot_group.warmup_p2p_pg(self._gpu_buffers[0], self._gpu_buffers[1])
            torch.cuda.current_stream().synchronize()

    def _snapshot_background(self, cpu_metadata_state_dict):
        """Background thread: GPU→CPU copy + Gloo metadata exchange."""
        torch.cuda.set_device(self._local_copy_stream.device)
        with torch.cuda.stream(self._local_copy_stream):
            self.local_curr.snapshot_gpu_state()
            self._local_copy_event.record()

        remote_cpu_metadata_state_dict = self.snapshot_group.exchange_object(cpu_metadata_state_dict)
        self.remote_curr.set_cpu_metadata_state_dict(remote_cpu_metadata_state_dict)
        self.local_curr.commit_cpu_metadata()
        self.remote_curr.commit_cpu_metadata()

    def snapshot(self, curr_step):
        if not self.enable:
            return

        self._is_snapshot_step = True
        self._snapshot_step = curr_step
        self._p2p_copy_event_recorded = [False, False]
        self._reset_for_new_step()

        self.snapshot_container.invalidate(self._curr_version)
        cpu_metadata_state_dict = self.local_curr.snapshot_cpu_metadata_state()
        self._snapshot_future = self._snapshot_thread_pool.submit(
            self._snapshot_background, cpu_metadata_state_dict
        )

    def begin_collective(self):
        torch.cuda.current_stream().wait_stream(self._p2p_stream)

    def end_collective(self, async_op: bool, handle):
        if not self.enable or not self._is_snapshot_step:
            return

        if not self._have_strategy:
            return

        self._comm_gap_id += 1
        if self._comm_gap_id in self._snapshot_strategy:
            num_blocks = self._snapshot_strategy[self._comm_gap_id]
            comm_stream = torch.cuda.current_stream()
            with torch.cuda.stream(self._p2p_stream):
                if async_op:
                    handle.wait()
                else:
                    self._p2p_stream.wait_stream(comm_stream)

                self._snapshot_blocks(num_blocks)

    def _snapshot_blocks(self, num_blocks: int):
        if num_blocks == -1:
            num_blocks = self._total_blocks - self._cur_block_id

        for _ in range(num_blocks):
            if self._cur_block_id >= self._total_blocks:
                break
            self._snapshot_block()

    def _snapshot_block(self):
        if self._cur_block_id >= self._total_blocks:
            return

        block_id = self._cur_block_id
        gpu_block = self._gpu_blocks[block_id]

        # View as uint8 for dtype-agnostic P2P (avoids NCCL size mismatch
        # when gpu_block dtype differs from peer's, e.g. bfloat16 vs float32).
        gpu_block_bytes = gpu_block.view(torch.uint8)
        block_bytes = gpu_block_bytes.numel()

        buf_id = self._gpu_buffer_id
        output_tensor = self._gpu_buffers[buf_id][:block_bytes]
        self._gpu_buffer_id = 1 - self._gpu_buffer_id

        # Wait only for the previous copy that used THIS buffer (2 blocks ago),
        # not the immediately preceding copy which uses the other buffer.
        if self._p2p_copy_event_recorded[buf_id]:
            self._p2p_stream.wait_event(self._p2p_copy_events[buf_id])

        self.snapshot_group.sendrecv_tensor(gpu_block_bytes, output_tensor)

        with torch.cuda.stream(self._copy_stream):
            cpu_block = self.remote_curr.get_block(block_id)
            # View cpu_block as uint8 to match output_tensor dtype (avoids GPU cast kernel)
            cpu_block_bytes = cpu_block.view(torch.uint8)

            self._copy_stream.wait_stream(self._p2p_stream)
            cpu_block_bytes.copy_(output_tensor, non_blocking=True)
            self._p2p_copy_events[buf_id].record()
            self._p2p_copy_event_recorded[buf_id] = True

        self._cur_block_id += 1

    def maybe_wait_for_staging(self):
        if not self.enable:
            return

        if self._is_snapshot_step:
            assert self._cur_block_id == self._total_blocks
            # Wait for background thread (Gloo exchange + local copy launch)
            self._snapshot_future.result()
            curr_stream = torch.cuda.current_stream()
            curr_stream.wait_stream(self._copy_stream)
            curr_stream.wait_stream(self._p2p_stream)
            curr_stream.wait_event(self._local_copy_event)

            # Durability barrier before commit. The wait_stream/wait_event calls
            # above only order *GPU* work on curr_stream — they do NOT block the
            # host, so the async D2H copies filling the LOCAL pool
            # (snapshot_gpu_state) and the REMOTE pool (the P2P block copies on
            # _copy_stream) may still be in flight when the CPU marks this
            # version committed below. On a FATAL fault the training process is
            # SIGKILLed; if the kill lands after commit but before those copies
            # land, tearing down the CUDA context aborts the DMAs mid-flight and
            # the "committed" pool is left with partial (torn) data. The
            # SnapshotContainer then persists that partial pool as a valid
            # checkpoint, and recovery loads stale/torn optimizer moments — the
            # loss diverges by a timing-dependent amount (the REMOTE pool is the
            # usual victim: its copies are staged during the *next* step's
            # forward, so they are the ones still in flight at commit, which is
            # why peer-replica RECV recovery on the failed worker is worst hit).
            # Block the host until the copies have actually landed so a committed
            # version is always durable across a SIGKILL.
            self._copy_stream.synchronize()
            self._p2p_stream.synchronize()
            self._local_copy_event.synchronize()

            self.snapshot_container.commit(self._curr_version, self._snapshot_step)
            # Mark snapshot step as complete
            self._is_snapshot_step = False

    def close(self):
        if not self.enable:
            return

        self._snapshot_thread_pool.shutdown(wait=True)
        self.snapshot_container.close()
