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

from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.ft import FTManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Checkpoint as CheckpointConfig, TORCH_DTYPE_MAP
from torchtitan.protocols import BaseStateDictAdapter
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import GarbageCollection

from torchtitan.components.gemini.snapshot_profiler import SnapshotProfiler

class GeminiAllGather(DefaultAllGather):
    def __init__(self, profiler):
        super().__init__()
        self._profiler = profiler

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> dist.Work | None:
        self._profiler.maybe_profile_gap_begin()
        handle = super().__call__(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )
        self._profiler.maybe_profile_gap_end(async_op, handle)

class GeminiReduceScatter(DefaultReduceScatter):

    def __init__(self, profiler):
        super().__init__()
        self._profiler = profiler

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: Any,
        op: Any,
        async_op: bool = False,
    ) -> dist.Work:
        self._profiler.maybe_profile_gap_begin()
        handle = super().__call__(
            output_tensor=output_tensor,
            input_tensor=input_tensor,
            group=group,
            op=op,
            async_op=async_op,
        )
        self._profiler.maybe_profile_gap_end(async_op, handle)

class GeminiCheckpointManager:
    def __init__(
        self,
        dataloader: BaseDataLoader | None,
        model_parts: list[nn.Module],
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        states: dict[str, Any],
        checkpoint_config: CheckpointConfig,
        sd_adapter: BaseStateDictAdapter | None,
        base_folder: str = "",
        ft_manager: FTManager | None = None,
    ) -> None:
        self.folder = os.path.join(base_folder, checkpoint_config.folder)
        self.interval = checkpoint_config.interval
        self.enable = checkpoint_config.enable

        # Setup profiler with config
        comm_gaps_folder = os.path.join(
            base_folder, checkpoint_config.gemini_comm_gaps_folder
        )
        self._profiler = SnapshotProfiler(
            enable=checkpoint_config.gemini_profile_comm_gaps,
            skip_first_k=checkpoint_config.gemini_skip_first_k,
            output_folder=comm_gaps_folder,
        )
        self._gemini_all_gather = GeminiAllGather(self._profiler)
        self._gemini_reduce_scatter = GeminiReduceScatter(self._profiler)
        self._register_collectives(model_parts)

    def _register_collectives(self, model_parts):
        for model_part in model_parts:
            for module in model_part.modules():
                if isinstance(module, FSDPModule):
                    module.set_custom_all_gather(self._gemini_all_gather)
                    module.set_custom_reduce_scatter(self._gemini_reduce_scatter)

    @torch.no_grad()
    def start_step(self, curr_step: int, last_step: bool = False) -> None:
        self._profiler.start_step()
        return

    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> None:
        return

    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        return False

    def maybe_wait_for_staging(self) -> None:
        self._profiler.start_optimizer()
        return

    def wait_for_tracking(self) -> None:
        return

    def __del__(self):
        self.close()

    def close(self):
        self._profiler.compute_gaps()

