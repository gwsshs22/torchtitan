import itertools
import math
from typing import Any

import torch
from torch.cuda._pin_memory_utils import pin_memory
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor

from torchtitan.components.gemini.snapshot_container import SnapshotContainer
from torchtitan.components.gemini.utils import (
    InMemStateType,
    stateful_to_state_dict,
    state_dict_to_stateful
)
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import (
    _to_local_tensor,
    _to_dtensor
)

class InMemState:

    def __init__(
        self,
        state_id,
        model_wrapper,
        optimizers,
        train_states,
        state_type,
        snapshot_container,
    ):
        self._state_id = state_id
        self._model_wrapper = model_wrapper
        self._optimizers = optimizers
        self._train_states = train_states
        self._state_type = state_type
        self._snapshot_container = snapshot_container

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
            storage_size = local_tensor.numel() * local_tensor.element_size()
            cpu_storage = torch.UntypedStorage._new_shared(storage_size, device='cpu')
            pin_memory(cpu_storage.data_ptr(), cpu_storage.nbytes())
            cpu_tensor = torch.empty(0, dtype=local_tensor.dtype)
            cpu_tensor.set_(
                source=cpu_storage,
                storage_offset=0,
                size=local_tensor.shape,
                stride=local_tensor.stride()
            )

            self._model_cpu_tensors.append(cpu_tensor)
            self._model_gpu_tensors.append(local_tensor)
            self._model_tensor_keys.append(k)

        for k, tensor in self._optimizers.state_dict().items():
            if isinstance(tensor, torch.Tensor) and not k.endswith(".step"):
                local_tensor = _to_local_tensor(tensor)
                storage_size = local_tensor.numel() * local_tensor.element_size()
                cpu_storage = torch.UntypedStorage._new_shared(storage_size, device='cpu')
                pin_memory(cpu_storage.data_ptr(), cpu_storage.nbytes())
                cpu_tensor = torch.empty(0, dtype=local_tensor.dtype)
                cpu_tensor.set_(
                    source=cpu_storage,
                    storage_offset=0,
                    size=local_tensor.shape,
                    stride=local_tensor.stride()
                )

                self._optim_cpu_tensors.append(cpu_tensor)
                self._optim_gpu_tensors.append(local_tensor)
                self._optim_tensor_keys.append(k)

        self._snapshot_container.register(
            state_id=self._state_id,
            in_mem_state_type=self._state_type,
            model_tensor_keys=self._model_tensor_keys,
            model_cpu_tensors=self._model_cpu_tensors,
            optim_tensor_keys=self._optim_tensor_keys,
            optim_cpu_tensors=self._optim_cpu_tensors,
        )

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
        for cpu_tensor, gpu_tensor in zip(self._optim_cpu_tensors, self._optim_gpu_tensors):
            cpu_tensor.copy_(gpu_tensor, non_blocking=True)
        for cpu_tensor, gpu_tensor in zip(self._model_cpu_tensors, self._model_gpu_tensors):
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

        # Send metadata to SnapshotContainer if enabled
        self._snapshot_container.snapshot_cpu_metadata(
            state_id=self._state_id,
            in_mem_state_type=self._state_type,
            cpu_metadata=self._cpu_metadata_state_dict,
        )

        return self._cpu_metadata_state_dict

    def set_cpu_metadata_state_dict(self, cpu_metadata_state_dict):
        assert self._state_type == InMemStateType.REMOTE
        self._cpu_metadata_state_dict = cpu_metadata_state_dict
        self._snapshot_container.snapshot_cpu_metadata(
            state_id=self._state_id,
            in_mem_state_type=self._state_type,
            cpu_metadata=self._cpu_metadata_state_dict,
        )

    def get_block(self, block_id: int) -> torch.Tensor:
        """Get a tensor block by ID from the CPU tensor blocks."""
        assert self._tensor_blocks is not None, "Must call compute_tensor_blocks first"
        return self._tensor_blocks[block_id]

    def load_state_dict(self, state_dict: dict[str, Any]):
        cpu_metadata = state_dict["_cpu_metadata"]
        state_dict_to_stateful(self._train_states, cpu_metadata["TRAIN"])

        assert self._model_tensor_keys == state_dict["_model_tensor_keys"]
        assert self._optim_tensor_keys == state_dict["_optim_tensor_keys"]
        checkpointed_step = self._train_states["train_state"].step
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

        # Manually setting "*.step" values to avoid handle such a small tensors.
        for k, tensor in optim_state_dict.items():
            if k.endswith(".step"):
                cpu_tensor = torch.tensor(checkpointed_step + 1, dtype=tensor.dtype, device="cpu")
                optim_new_state_dict[k] = _to_dtensor(cpu_tensor, tensor)


        for k, tensor in optim_state_dict.items():
            if k.endswith(".step"):
                cpu_tensor = torch.tensor(checkpointed_step + 1, dtype=tensor.dtype, device="cpu")
                optim_new_state_dict[k] = _to_dtensor(cpu_tensor, tensor)

        self._optimizers.load_state_dict(optim_new_state_dict)

        # Reset gpu tensor references.
        self._optim_gpu_tensors = []
        optim_tensor_keys = []
        for k, tensor in self._optimizers.state_dict().items():
            if isinstance(tensor, torch.Tensor) and not k.endswith(".step"):
                local_tensor = _to_local_tensor(tensor)
                self._optim_gpu_tensors.append(local_tensor)
                optim_tensor_keys.append(k)
        assert optim_tensor_keys == self._optim_tensor_keys

