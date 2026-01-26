from enum import Enum, auto
import itertools
import math
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor

from torchtitan.components.gemini.utils import (
    stateful_to_state_dict,
    state_dict_to_stateful
)
from torchtitan.tools.logging import logger

def _to_local_tensor(tensor: torch.Tensor | DTensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor

def _to_dtensor(
    local_tensor: torch.Tensor,
    reference_tensor: torch.Tensor | DTensor
) -> torch.Tensor | DTensor:
    if isinstance(reference_tensor, DTensor):
        return DTensor.from_local(
            local_tensor,
            device_mesh=reference_tensor.device_mesh,
            placements=reference_tensor.placements,
            run_check=False  # Skip global shape checks for efficiency
        )
    else:
        return local_tensor

class InMemStateType(Enum):
    LOCAL = auto()
    REMOTE = auto()

class InMemState(Stateful):

    def __init__(
        self,
        model_wrapper,
        optimizers,
        train_states,
        state_type
    ):
        self._model_wrapper = model_wrapper
        self._optimizers = optimizers
        self._train_states = train_states
        self._state_type = state_type

        self._model_gpu_tensors = None
        self._optim_gpu_tensors = None
        self._model_tensor_keys = None
        self._optim_tensor_keys = None

        self._model_cpu_tensors = None
        self._optim_cpu_tensors = None
        self._cpu_metadata_state_dict = None

        self._tensor_blocks = None

    def init_cpu_tensors(self):
        if self._model_cpu_tensors is not None:
            return

        self._model_cpu_tensors = []
        self._model_gpu_tensors = []
        self._model_tensor_keys = []
        self._optim_cpu_tensors = []
        self._optim_gpu_tensors = []
        self._optim_tensor_keys = []

        for k, tensor in self._model_wrapper.state_dict().items():
            assert isinstance(tensor, torch.Tensor)
            local_tensor = _to_local_tensor(tensor)
            cpu_tensor = torch.empty(
                local_tensor.shape,
                dtype=local_tensor.dtype,
                device='cpu',
                pin_memory=True
            )

            self._model_cpu_tensors.append(cpu_tensor)
            self._model_gpu_tensors.append(local_tensor)
            self._model_tensor_keys.append(k)

        for k, tensor in self._optimizers.state_dict().items():
            if isinstance(tensor, torch.Tensor):
                local_tensor = _to_local_tensor(tensor)
                cpu_tensor = torch.empty(
                    local_tensor.shape,
                    dtype=local_tensor.dtype,
                    device='cpu',
                    pin_memory=True
                )

                self._optim_cpu_tensors.append(cpu_tensor)
                self._optim_gpu_tensors.append(local_tensor)
                self._optim_tensor_keys.append(k)

    def compute_tensor_blocks(
        self,
        block_size,
        return_gpu_blocks=False
    ):
        assert self._state_type == InMemStateType.REMOTE
        self._tensor_blocks = self._compute_tensor_blocks(
            self._model_cpu_tensors, self._optim_cpu_tensors, block_size
        )

        if return_gpu_blocks:
            return self._compute_tensor_blocks(
                self._model_gpu_tensors, self._optim_gpu_tensors, block_size
            )
        else:
            return None

    def _compute_tensor_blocks(self, model_tensors, optim_tensors, block_size):
        blocks = []
        for tensor in itertools.chain(model_tensors, optim_tensors):
            numel = tensor.numel()
            num_blocks = math.ceil(numel / block_size)
            flat_tensor = tensor.view(-1)

            for i in range(num_blocks):
                start_pos = i * block_size
                end_pos = min((i + 1) * block_size, numel)
                blocks.append(flat_tensor[start_pos:end_pos])

        return blocks
    
    def snapshot_gpu_state(self):
        assert self._state_type == InMemStateType.LOCAL
        for cpu_tensor, gpu_tensor in zip(self._model_cpu_tensors, self._model_gpu_tensors):
            cpu_tensor.copy_(gpu_tensor, non_blocking=True)
        for cpu_tensor, gpu_tensor in zip(self._optim_cpu_tensors, self._optim_gpu_tensors):
            cpu_tensor.copy_(gpu_tensor, non_blocking=True)

    def snapshot_cpu_metadata_state(self):
        assert self._state_type == InMemStateType.LOCAL

        optim_cpu_metadata = {
            k: v for k, v in self._optimizers.state_dict().items()
            if not isinstance(v, torch.Tensor)
        }

        self._cpu_metadata_state_dict = {
            "TRAIN": stateful_to_state_dict(self._train_states),
            "OPTIM": optim_cpu_metadata
        }

        return self._cpu_metadata_state_dict

    def set_cpu_metadata_state_dict(self, cpu_metadata_state_dict):
        assert self._state_type == InMemStateType.REMOTE
        self._cpu_metadata_state_dict = cpu_metadata_state_dict

    def get_block(self, block_id: int) -> torch.Tensor:
        """Get a tensor block by ID from the CPU tensor blocks."""
        assert self._tensor_blocks is not None, "Must call compute_tensor_blocks first"
        return self._tensor_blocks[block_id]

    def state_dict(self) -> dict[str, Any]:
        state = {
            "_model_tensor_keys": self._model_tensor_keys,
            "_model_cpu_tensors": self._model_cpu_tensors,
            "_optim_tensor_keys": self._optim_tensor_keys,
            "_optim_cpu_tensors": self._optim_cpu_tensors,
            "_cpu_metadata": self._cpu_metadata_state_dict
        }

        return state

    def load_state_dict(self, state_dict: dict[str, Any]):
        cpu_metadata = state_dict["_cpu_metadata"]
        state_dict_to_stateful(self._train_states, cpu_metadata["TRAIN"])

        assert self._model_tensor_keys == state_dict["_model_tensor_keys"]
        assert self._optim_tensor_keys == state_dict["_optim_tensor_keys"]

        for gpu_tensor, cpu_tensor in zip(
            self._model_gpu_tensors, 
            state_dict["_model_cpu_tensors"]
        ):
            gpu_tensor.copy_(cpu_tensor, non_blocking=True)
        
        optim_new_state_dict = cpu_metadata["OPTIM"]
        optim_state_dict = self._optimizers.state_dict()

        for k, cpu_tensor in zip(
            self._optim_tensor_keys, state_dict["_optim_cpu_tensors"]
        ):
            optim_new_state_dict[k] = _to_dtensor(cpu_tensor, optim_state_dict[k])

        self._optimizers.load_state_dict(optim_new_state_dict)

