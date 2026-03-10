import itertools
import math
import os
import time
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor

from torchtitan.components.gemini.snapshot_container import SnapshotContainer
from torchtitan.components.gemini.utils import InMemStateType
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import (
    _to_local_tensor,
    _to_dtensor,
    stateful_to_state_dict,
    state_dict_to_stateful
)

try:
    from leto.rmp.client import RmpClient, CpuTensorSpec
    from leto.rmp.shm_tensor import (
        allocate_shm_tensor, import_shm_tensor,
        _sanitize_name, _compute_storage_size,
    )
    from torch.cuda._pin_memory_utils import pin_memory
    _RMP_AVAILABLE = True
except ImportError:
    _RMP_AVAILABLE = False

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

    def init_cpu_tensors(self, rank, mem_fs_folder):
        """Initialize CPU tensors using file-backed shared memory.

        Allocates tensors via allocate_shm_tensor() under mem_fs_folder/rmp/.
        Pinning is done at allocation time for DMA bandwidth.
        Passes lightweight file path metadata to SnapshotContainer.
        """
        if self._model_cpu_tensors is not None:
            return

        self._model_cpu_tensors = []
        self._model_gpu_tensors = []
        self._model_tensor_keys = []
        self._optim_cpu_tensors = []
        self._optim_gpu_tensors = []
        self._optim_tensor_keys = []

        model_file_infos = {}
        optim_file_infos = {}

        for k, tensor in self._model_wrapper.state_dict().items():
            assert isinstance(tensor, torch.Tensor)
            local_tensor = _to_local_tensor(tensor)
            name = self._make_rmp_name(rank, k, "model")

            cpu_tensor, file_info = allocate_shm_tensor(
                name=name,
                shape=tuple(local_tensor.shape),
                dtype=local_tensor.dtype,
                shm_dir=mem_fs_folder,
                pin=True,
            )

            self._model_cpu_tensors.append(cpu_tensor)
            self._model_gpu_tensors.append(local_tensor)
            self._model_tensor_keys.append(k)
            model_file_infos[k] = file_info

        for k, tensor in self._optimizers.state_dict().items():
            if isinstance(tensor, torch.Tensor) and not k.endswith(".step"):
                local_tensor = _to_local_tensor(tensor)
                name = self._make_rmp_name(rank, k, "optim")

                cpu_tensor, file_info = allocate_shm_tensor(
                    name=name,
                    shape=tuple(local_tensor.shape),
                    dtype=local_tensor.dtype,
                    shm_dir=mem_fs_folder,
                    pin=True,
                )

                self._optim_cpu_tensors.append(cpu_tensor)
                self._optim_gpu_tensors.append(local_tensor)
                self._optim_tensor_keys.append(k)
                optim_file_infos[k] = file_info

        self._snapshot_container.register(
            state_id=self._state_id,
            in_mem_state_type=self._state_type,
            model_tensor_keys=self._model_tensor_keys,
            model_file_infos=model_file_infos,
            optim_tensor_keys=self._optim_tensor_keys,
            optim_file_infos=optim_file_infos,
        )

    def _make_rmp_name(self, rank, key, category):
        return f"gemini/{self._state_type.name}/{self._state_id}/{category}/{rank}/{key}"

    def init_cpu_tensors_from_rmp(self, rmp_client, rank, mem_fs_folder=""):
        """Initialize CPU tensors by retrieving or allocating them via RMP server.

        RMP server allocates + touches file-backed shared memory (populating
        page cache). Training process imports via import_shm_tensor with
        touch + pin (cudaHostRegister) for DMA bandwidth (~24.5 GB/s D2H).
        """
        if self._model_cpu_tensors is not None:
            return

        self._model_cpu_tensors = []
        self._model_gpu_tensors = []
        self._model_tensor_keys = []
        self._optim_cpu_tensors = []
        self._optim_gpu_tensors = []
        self._optim_tensor_keys = []

        # Collect specs and entries
        cpu_tensor_specs = []
        model_entries = []
        optim_entries = []

        for k, tensor in self._model_wrapper.state_dict().items():
            assert isinstance(tensor, torch.Tensor)
            local_tensor = _to_local_tensor(tensor)
            name = self._make_rmp_name(rank, k, "model")
            cpu_tensor_specs.append(CpuTensorSpec(
                name=name,
                shape=tuple(local_tensor.shape),
                dtype=local_tensor.dtype,
            ))
            model_entries.append((k, local_tensor))

        for k, tensor in self._optimizers.state_dict().items():
            if isinstance(tensor, torch.Tensor) and not k.endswith(".step"):
                local_tensor = _to_local_tensor(tensor)
                name = self._make_rmp_name(rank, k, "optim")
                cpu_tensor_specs.append(CpuTensorSpec(
                    name=name,
                    shape=tuple(local_tensor.shape),
                    dtype=local_tensor.dtype,
                ))
                optim_entries.append((k, local_tensor))

        # Single batch RPC call — triggers server-side alloc + touch
        _, allocated = rmp_client.get_or_allocate_cpu_shared_tensors(
            cpu_tensor_specs, mem_fs_folder=mem_fs_folder,
        )
        status = "allocated" if allocated else "retrieved"
        logger.info(
            f"[Gemini] RMP CPU tensors {status} for "
            f"{self._state_type.name}/{self._state_id}: "
            f"{len(model_entries)} model + {len(optim_entries)} optim tensors"
        )

        # Import tensors from shared memory files and pin.
        # cudaHostRegister is per-process — server-side pinning is invisible
        # to us, so we must pin in the training process for DMA bandwidth.
        # Pages are already in page cache from server's touch, making our
        # pin ~2.5x faster than allocating from scratch.

        model_file_infos = {}
        optim_file_infos = {}
        rmp_dir = os.path.join(mem_fs_folder, "rmp")

        total_bytes = 0
        t_touch_total = 0.0
        t_pin_total = 0.0

        for k, local_tensor in model_entries:
            name = self._make_rmp_name(rank, k, "model")
            file_info = {
                'file_path': os.path.join(rmp_dir, _sanitize_name(name)),
                'storage_size': _compute_storage_size(
                    tuple(local_tensor.shape), local_tensor.dtype
                ),
                'dtype': local_tensor.dtype,
                'shape': tuple(local_tensor.shape),
                'stride': tuple(local_tensor.stride()),
                'storage_offset': 0,
            }
            cpu_tensor = import_shm_tensor(file_info, pin=False, touch=False)

            t0 = time.perf_counter()
            # cpu_tensor.fill_(0)
            t1 = time.perf_counter()
            pin_memory(cpu_tensor.untyped_storage().data_ptr(),
                       cpu_tensor.untyped_storage().nbytes())
            t2 = time.perf_counter()

            t_touch_total += t1 - t0
            t_pin_total += t2 - t1
            total_bytes += file_info['storage_size']

            self._model_cpu_tensors.append(cpu_tensor)
            self._model_gpu_tensors.append(local_tensor)
            self._model_tensor_keys.append(k)
            model_file_infos[k] = file_info

        for k, local_tensor in optim_entries:
            name = self._make_rmp_name(rank, k, "optim")
            file_info = {
                'file_path': os.path.join(rmp_dir, _sanitize_name(name)),
                'storage_size': _compute_storage_size(
                    tuple(local_tensor.shape), local_tensor.dtype
                ),
                'dtype': local_tensor.dtype,
                'shape': tuple(local_tensor.shape),
                'stride': tuple(local_tensor.stride()),
                'storage_offset': 0,
            }
            cpu_tensor = import_shm_tensor(file_info, pin=False, touch=False)

            t0 = time.perf_counter()
            # cpu_tensor.fill_(0)
            t1 = time.perf_counter()
            pin_memory(cpu_tensor.untyped_storage().data_ptr(),
                       cpu_tensor.untyped_storage().nbytes())
            t2 = time.perf_counter()

            t_touch_total += t1 - t0
            t_pin_total += t2 - t1
            total_bytes += file_info['storage_size']

            self._optim_cpu_tensors.append(cpu_tensor)
            self._optim_gpu_tensors.append(local_tensor)
            self._optim_tensor_keys.append(k)
            optim_file_infos[k] = file_info

        total_mb = total_bytes / (1024 * 1024)
        logger.info(
            f"[Gemini] init_cpu_tensors_from_rmp touch+pin timing: "
            f"touch={t_touch_total*1000:.1f}ms, pin={t_pin_total*1000:.1f}ms, "
            f"total={total_mb:.0f}MB "
            f"({total_mb/1024/(t_touch_total + t_pin_total):.1f} GB/s effective)"
        )

        self._snapshot_container.register(
            state_id=self._state_id,
            in_mem_state_type=self._state_type,
            model_tensor_keys=self._model_tensor_keys,
            model_file_infos=model_file_infos,
            optim_tensor_keys=self._optim_tensor_keys,
            optim_file_infos=optim_file_infos,
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

        return self._cpu_metadata_state_dict

    def set_cpu_metadata_state_dict(self, cpu_metadata_state_dict):
        assert self._state_type == InMemStateType.REMOTE
        self._cpu_metadata_state_dict = cpu_metadata_state_dict
    
    def commit_cpu_metadata(self):
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

        # Manually setting "*.step" values to avoid handling such small tensors.
        for k, tensor in optim_state_dict.items():
            if k.endswith(".step"):
                cpu_tensor = torch.tensor(checkpointed_step, dtype=tensor.dtype, device="cpu")
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

