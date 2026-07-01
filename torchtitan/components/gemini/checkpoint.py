from typing import Any
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    DefaultAllGather,
    DefaultReduceScatter,
)
import torch.nn as nn

from torchtitan.components.checkpoint import (
    ModelWrapper,
    DATALOADER,
    LR_SCHEDULER,
)
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Checkpoint as CheckpointConfig
from torchtitan.distributed import ParallelDims

from torchtitan.components.gemini.snapshot_executor import SnapshotExecutor
from torchtitan.components.gemini.snapshot_profiler import SnapshotProfiler

class GeminiCheckpointManager:
    def __init__(
        self,
        dataloader: BaseDataLoader | None,
        states: dict[str, Any],
        checkpoint_config: CheckpointConfig,
        base_folder: str = "",
        **kwargs: Any,
    ) -> None:
        self.interval = checkpoint_config.interval
        self.enable = checkpoint_config.enable
        self.skip_last_save = checkpoint_config.skip_last_save

        self.states = states
        self.states[DATALOADER] = dataloader

        self._checkpoint_config = checkpoint_config

        comm_gaps_folder = os.path.join(
            base_folder, checkpoint_config.gemini_comm_gaps_folder
        )

        self._profiler = SnapshotProfiler(
            enable=checkpoint_config.gemini_profile_comm_gaps,
            skip_first_k=checkpoint_config.gemini_skip_first_k,
            output_folder=comm_gaps_folder,
        )
        self._executor = SnapshotExecutor(
            enable=not checkpoint_config.gemini_profile_comm_gaps,
            comm_gaps_folder=comm_gaps_folder,
            mem_fs_folder=checkpoint_config.gemini_mem_fs_folder,
        )

    def lazy_init(
        self,
        model_parts: list[nn.Module],
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        parallel_dims: ParallelDims | None = None,
        collective_manager=None,
    ) -> None:
        assert parallel_dims.fsdp_enabled, "Gemini needs FSDP enabled."

        self.model_wrapper = ModelWrapper(model_parts)
        self.optimizers = optimizers
        self.states[LR_SCHEDULER] = lr_schedulers

        def _get_pg(mesh_name: str) -> dist.ProcessGroup | None:
            if parallel_dims is None:
                return None
            mesh = parallel_dims.get_optional_mesh(mesh_name)
            return mesh.get_group() if mesh is not None else None

        fsdp_pg = _get_pg("fsdp")
        tp_pg = _get_pg("tp")
        pp_pg = _get_pg("pp")

        self._profiler.lazy_init(fsdp_process_group=fsdp_pg)
        self._executor.lazy_init(
            states=self.states,
            model_wrapper=self.model_wrapper,
            optimizers=self.optimizers,
            tp_process_group=tp_pg,
            fsdp_process_group=fsdp_pg,
            pp_process_group=pp_pg,
        )

        checkpoint_config = self._checkpoint_config
        callback = (
            self._profiler
            if checkpoint_config.gemini_profile_comm_gaps
            else self._executor
        )
        self._register_collectives(callback, collective_manager)

    def _register_collectives(self, callback, collective_manager=None):
        default_ag = DefaultAllGather()
        default_rs = DefaultReduceScatter()

        def ag_call(output_tensor, input_tensor, group, async_op=False):
            callback.begin_collective()
            handle = default_ag(
                output_tensor, input_tensor, group=group, async_op=async_op
            )
            callback.end_collective(async_op, handle)
            return handle

        def rs_call(output_tensor, input_tensor, group, op, async_op=False):
            callback.begin_collective()
            handle = default_rs(
                output_tensor=output_tensor, input_tensor=input_tensor,
                group=group, op=op, async_op=async_op,
            )
            callback.end_collective(async_op, handle)
            return handle

        if collective_manager is not None:
            collective_manager.set_all_gather_call(ag_call)
            collective_manager.set_reduce_scatter_call(rs_call)
        else:
            from torchtitan.components.fsdp_collective_manager import (
                FsdpCollectiveManager,
            )
            # Standalone: create a manager just for Gemini
            mgr = FsdpCollectiveManager()
            mgr.set_all_gather_call(ag_call)
            mgr.set_reduce_scatter_call(rs_call)

    @torch.no_grad()
    def begin_step(self, curr_step: int, last_step: bool = False) -> None:
        self._profiler.begin_step()

    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> None:
        if not self._should_save(curr_step, last_step):
            return
        self._executor.snapshot(curr_step)

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        if not self.enable:
            return False

        if last_step:
            # Skip last save if configured
            if self.skip_last_save:
                return False
            return True

        if curr_step % self.interval == 0:
            return True

        return False

    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        return self._executor.load(step)

    def maybe_wait_for_staging(self) -> None:
        self._profiler.begin_optimizer()
        self._executor.maybe_wait_for_staging()

    def wait_for_tracking(self) -> None:
        return

    def __del__(self):
        self.close()

    def close(self):
        self._profiler.compute_gaps()
        self._executor.close()

