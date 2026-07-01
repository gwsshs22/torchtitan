import itertools
import math
import time
from typing import Any

import torch
from torch.cuda._pin_memory_utils import pin_memory
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor

from torchtitan.components.gemini.snapshot_container import SnapshotContainer
from torchtitan.components.gemini.utils import InMemStateType
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import (
    _to_local_tensor,
    stateful_to_state_dict,
    state_dict_to_stateful
)

try:
    from leto.launch.worker_controller_client import (
        report_duration,
        DURATION_CHECKPOINT_ALLOC,
        DURATION_CHECKPOINT_INIT_TOTAL,
    )
    _LETO_AVAILABLE = True
except ImportError:
    _LETO_AVAILABLE = False

_POOL_ALIGNMENT = 64  # bytes


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

        self._pool_storage = None
        self._tensor_blocks = None

    def init_cpu_tensors(self):
        if self._model_cpu_tensors is not None:
            return

        t_init_start = time.perf_counter()

        # Phase 1: Collect tensor metadata (no allocation yet).
        model_entries = []
        for k, tensor in self._model_wrapper.state_dict().items():
            assert isinstance(tensor, torch.Tensor)
            local_tensor = _to_local_tensor(tensor)
            size_bytes = local_tensor.numel() * local_tensor.element_size()
            model_entries.append((k, local_tensor, local_tensor.dtype,
                                  local_tensor.shape, local_tensor.stride(),
                                  size_bytes))

        optim_entries = []
        for k, tensor in self._optimizers.state_dict().items():
            if isinstance(tensor, torch.Tensor) and not k.endswith(".step"):
                local_tensor = _to_local_tensor(tensor)
                size_bytes = local_tensor.numel() * local_tensor.element_size()
                optim_entries.append((k, local_tensor, local_tensor.dtype,
                                     local_tensor.shape, local_tensor.stride(),
                                     size_bytes))

        # Phase 2: Compute byte offsets with alignment.
        all_entries = model_entries + optim_entries
        byte_offsets = []
        total_bytes = 0
        for _, _, _, _, _, size_bytes in all_entries:
            aligned = (total_bytes + _POOL_ALIGNMENT - 1) // _POOL_ALIGNMENT * _POOL_ALIGNMENT
            byte_offsets.append(aligned)
            total_bytes = aligned + size_bytes
        total_bytes = (total_bytes + _POOL_ALIGNMENT - 1) // _POOL_ALIGNMENT * _POOL_ALIGNMENT

        # Phase 3: Allocate shared pool -> touch -> pin.
        
        t0 = time.perf_counter()
        # Allocate filename-backed shm directly to avoid the fd-to-filename copy
        # that _share_filename_cpu_() would do on fd-based storage. Gemini is
        # decoupled from RMP: the checkpoint pool is always process-owned shm.
        pool_storage = torch.UntypedStorage._new_using_filename_cpu(total_bytes)
        t1 = time.perf_counter()

        pool_share_info = pool_storage._share_filename_cpu_()
        t1b = time.perf_counter()

        logger.info(
            f"Pool: state_id={self._state_id}, type={self._state_type}, "
            f"size={total_bytes / (1024 * 1024):.2f}MB, "
            f"data_ptr=0x{pool_storage.data_ptr():x}, "
            f"shm_file={pool_share_info[0] if pool_share_info else 'N/A'}"
        )

        pool_view = torch.empty(0, dtype=torch.uint8)
        pool_view.set_(source=pool_storage, storage_offset=0, size=(total_bytes,))
        t2 = time.perf_counter()

        pin_memory(pool_storage.data_ptr(), pool_storage.nbytes())
        t3 = time.perf_counter()

        self._pool_storage = pool_storage

        # Phase 4: Create view tensors into the pool.
        t4_start = time.perf_counter()
        self._model_cpu_tensors = []
        self._model_gpu_tensors = []
        self._model_tensor_keys = []
        self._optim_cpu_tensors = []
        self._optim_gpu_tensors = []
        self._optim_tensor_keys = []

        model_metadata = {}
        optim_metadata = {}

        for i, (k, local_tensor, dtype, shape, stride, _) in enumerate(model_entries):
            elem_size = local_tensor.element_size()
            bo = byte_offsets[i]
            assert bo % elem_size == 0, (
                f"Byte offset {bo} not aligned to element size {elem_size} for {k}"
            )
            storage_offset = bo // elem_size

            cpu_tensor = torch.empty(0, dtype=dtype)
            cpu_tensor.set_(source=pool_storage, storage_offset=storage_offset,
                            size=shape, stride=stride)

            self._model_cpu_tensors.append(cpu_tensor)
            self._model_gpu_tensors.append(local_tensor)
            self._model_tensor_keys.append(k)
            model_metadata[k] = {
                'dtype': dtype,
                'shape': tuple(shape),
                'stride': tuple(stride),
                'storage_offset': storage_offset,
            }

        for i, (k, local_tensor, dtype, shape, stride, _) in enumerate(optim_entries):
            elem_size = local_tensor.element_size()
            bo = byte_offsets[len(model_entries) + i]
            assert bo % elem_size == 0, (
                f"Byte offset {bo} not aligned to element size {elem_size} for {k}"
            )
            storage_offset = bo // elem_size

            cpu_tensor = torch.empty(0, dtype=dtype)
            cpu_tensor.set_(source=pool_storage, storage_offset=storage_offset,
                            size=shape, stride=stride)

            self._optim_cpu_tensors.append(cpu_tensor)
            self._optim_gpu_tensors.append(local_tensor)
            self._optim_tensor_keys.append(k)
            optim_metadata[k] = {
                'dtype': dtype,
                'shape': tuple(shape),
                'stride': tuple(stride),
                'storage_offset': storage_offset,
            }

        t4_end = time.perf_counter()

        # Phase 5: Register with snapshot container (pool_share_info from Phase 3).
        t5_reg_start = time.perf_counter()
        self._snapshot_container.register(
            state_id=self._state_id,
            in_mem_state_type=self._state_type,
            model_tensor_keys=self._model_tensor_keys,
            model_tensor_metadata=model_metadata,
            optim_tensor_keys=self._optim_tensor_keys,
            optim_tensor_metadata=optim_metadata,
            pool_share_info=pool_share_info,
        )
        t5_reg_end = time.perf_counter()

        t_init_end = time.perf_counter()

        logger.info(
            f"Pool timings: "
            f"_new_shared={(t1 - t0) * 1000:.2f}ms, "
            f"_share_filename_cpu_={(t1b - t1) * 1000:.2f}ms, "
            f"touch={(t2 - t1b) * 1000:.2f}ms, "
            f"pin={(t3 - t2) * 1000:.2f}ms, "
            f"views={(t4_end - t4_start) * 1000:.2f}ms, "
            f"register={(t5_reg_end - t5_reg_start) * 1000:.2f}ms, "
            f"total={(t_init_end - t_init_start) * 1000:.2f}ms"
        )

        if _LETO_AVAILABLE:
            report_duration(DURATION_CHECKPOINT_ALLOC, t1b - t0)
            report_duration(DURATION_CHECKPOINT_INIT_TOTAL, t_init_end - t_init_start)

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

    @staticmethod
    def _reconstruct_tensors_from_pool(pool_bytes, keys, metadata):
        """Reconstruct typed tensor views from raw pool bytes."""
        pool_storage = pool_bytes.untyped_storage()
        tensors = []
        for key in keys:
            meta = metadata[key]
            tensor = torch.empty(0, dtype=meta['dtype'])
            tensor.set_(
                source=pool_storage,
                storage_offset=meta['storage_offset'],
                size=meta['shape'],
                stride=meta['stride']
            )
            tensors.append(tensor)
        return tensors

    def load_state_dict(self, state_dict: dict[str, Any]):
        cpu_metadata = state_dict["_cpu_metadata"]
        state_dict_to_stateful(self._train_states, cpu_metadata["TRAIN"])

        assert self._model_tensor_keys == state_dict["_model_tensor_keys"]
        assert self._optim_tensor_keys == state_dict["_optim_tensor_keys"]
        checkpointed_step = self._train_states["train_state"].step

        # Support both old format (direct tensor lists) and new format (pool bytes + metadata)
        if "_pool_bytes" in state_dict:
            pool_bytes = state_dict["_pool_bytes"]
            model_cpu_tensors = self._reconstruct_tensors_from_pool(
                pool_bytes, self._model_tensor_keys, state_dict["_model_metadata"]
            )
            optim_cpu_tensors = self._reconstruct_tensors_from_pool(
                pool_bytes, self._optim_tensor_keys, state_dict["_optim_metadata"]
            )
        else:
            model_cpu_tensors = state_dict["_model_cpu_tensors"]
            optim_cpu_tensors = state_dict["_optim_cpu_tensors"]

        # Model: write values *into* the existing GPU tensors (which may be
        # RMP-backed by RmpManager.maybe_init).  Reference identity must be
        # preserved so any other component holding the same tensor (e.g.
        # ResilientOptimizer's chunk schedule, RMP shared memory) keeps
        # observing the right storage.
        for gpu_tensor, cpu_tensor in zip(
            self._model_gpu_tensors, model_cpu_tensors
        ):
            gpu_tensor.copy_(cpu_tensor, non_blocking=True)

        # Optim non-step tensors: same in-place pattern as model tensors.
        # Previously this used optim.load_state_dict(...) with newly-built
        # CPU DTensors, which forced PyTorch's _cast to mint fresh CUDA
        # tensors via .to(device=cuda, ...) — severing the RMP backing of
        # exp_avg / exp_avg_sq / step that RmpManager.maybe_init had set
        # up.  Subsequent ResilientOptimizer._step_chunk writes then went
        # to non-RMP tensors and the RMP-backed copies stayed frozen at
        # the gemini-loaded values, breaking transient-fault recovery.
        for gpu_tensor, cpu_tensor in zip(
            self._optim_gpu_tensors, optim_cpu_tensors
        ):
            gpu_tensor.copy_(cpu_tensor, non_blocking=True)

        # Optim step tensors: fill the existing (RMP-backed) tensor in
        # place.  Same RMP-preservation reason as above.
        optim_state_dict = self._optimizers.state_dict()
        for k, tensor in optim_state_dict.items():
            if k.endswith(".step"):
                _to_local_tensor(tensor).fill_(checkpointed_step)

        # Apply non-tensor optim metadata (param_groups: lr, betas, ...).
        # Pass the *current* state_dict (RMP-backed tensor refs) to
        # load_state_dict and overlay the non-tensor entries from the
        # checkpoint — PyTorch's _cast does .to(dtype, device) which is a
        # no-op for already-on-device tensors, so the tensor refs survive
        # the round trip.
        optim_state_dict.update(cpu_metadata["OPTIM"])
        self._optimizers.load_state_dict(optim_state_dict)
