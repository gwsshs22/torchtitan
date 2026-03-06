from typing import Any
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    DefaultAllGather,
    DefaultReduceScatter
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

class GeminiAllGather(DefaultAllGather):
    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> dist.Work | None:
        self._callback.begin_collective()
        handle = super().__call__(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )
        self._callback.end_collective(async_op, handle)
        return handle

class GeminiReduceScatter(DefaultReduceScatter):

    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: Any,
        op: Any,
        async_op: bool = False,
    ) -> dist.Work:
        self._callback.begin_collective()
        handle = super().__call__(
            output_tensor=output_tensor,
            input_tensor=input_tensor,
            group=group,
            op=op,
            async_op=async_op,
        )
        self._callback.end_collective(async_op, handle)
        return handle

class GeminiCheckpointManager:
    def __init__(
        self,
        dataloader: BaseDataLoader | None,
        states: dict[str, Any],
        checkpoint_config: CheckpointConfig,
        base_folder: str = "",
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
        rmp_restored: bool = False,
    ) -> None:
        self.model_wrapper = ModelWrapper(model_parts)
        self.optimizers = optimizers
        self.states[LR_SCHEDULER] = lr_schedulers

        # Get FSDP process group from parallel_dims
        fsdp_pg = None
        if parallel_dims is not None:
            fsdp_mesh = parallel_dims.get_optional_mesh("fsdp")
            if fsdp_mesh is not None:
                fsdp_pg = fsdp_mesh.get_group()

        self._profiler.lazy_init(fsdp_process_group=fsdp_pg)
        self._executor.lazy_init(
            states=self.states,
            model_wrapper=self.model_wrapper,
            optimizers=self.optimizers,
            fsdp_process_group=fsdp_pg,
            rmp_restored=rmp_restored,
        )

        checkpoint_config = self._checkpoint_config
        if checkpoint_config.gemini_profile_comm_gaps:
            self._gemini_all_gather = GeminiAllGather(self._profiler)
            self._gemini_reduce_scatter = GeminiReduceScatter(self._profiler)
        else:
            self._gemini_all_gather = GeminiAllGather(self._executor)
            self._gemini_reduce_scatter = GeminiReduceScatter(self._executor)
        self._register_collectives(model_parts)

    def _register_collectives(self, model_parts):
        for model_part in model_parts:
            for module in model_part.modules():
                if isinstance(module, FSDPModule):
                    module.set_custom_all_gather(self._gemini_all_gather)
                    module.set_custom_reduce_scatter(self._gemini_reduce_scatter)

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

