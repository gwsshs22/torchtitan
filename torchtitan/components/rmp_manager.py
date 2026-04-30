from itertools import chain
import pickle
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import _get_fqns

from torchtitan.components.checkpoint import (
    ModelWrapper,
    DATALOADER,
    LR_SCHEDULER,
)
from torchtitan.components.rmp_grad_alloc import RmpGradientAllocator
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import (
    _to_local_tensor,
    _to_dtensor,
    stateful_to_state_dict,
    state_dict_to_stateful
)
from leto.rmp.client import RmpClient, TensorSpec
from leto.rmp.flags import FLAG_KIND_GPU


class MetadataCircularBuffer:
    """Circular buffer for training metadata backed by RMP CPU shared memory.

    Allocates NUM_SLOTS fixed-size CPU shared memory regions via RMP.
    Each slot stores a pickled metadata dict with a step marker.
    Slots are written in round-robin order (step % NUM_SLOTS).

    Per-slot layout (SLOT_SIZE bytes):
        [0:8]    step      (int64) — commit marker, -1 = invalid
        [8:16]   data_len  (int64) — payload byte count
        [16:...] payload   (uint8) — pickled metadata
    """

    NUM_SLOTS = 3
    SLOT_SIZE = 1 * 1024 * 1024  # 1 MB
    HEADER_SIZE = 16  # 8 bytes step + 8 bytes data_len
    MAX_PAYLOAD = SLOT_SIZE - HEADER_SIZE
    _INVALID_STEP = -1

    def __init__(self, rmp_client, slot_key_prefix="metadata"):
        self._slots = []
        for i in range(self.NUM_SLOTS):
            storage, allocated = rmp_client.get_or_allocate_cpu_memory(
                f"{slot_key_prefix}_{i}", self.SLOT_SIZE,
            )
            step_view = torch.tensor([], dtype=torch.int64).set_(
                storage, 0, (1,),
            )
            len_view = torch.tensor([], dtype=torch.int64).set_(
                storage, 1, (1,),  # element offset 1 = byte offset 8
            )
            payload_view = torch.tensor([], dtype=torch.uint8).set_(
                storage, self.HEADER_SIZE, (self.MAX_PAYLOAD,),
            )
            self._slots.append((step_view, len_view, payload_view))
            if allocated:
                step_view.fill_(self._INVALID_STEP)

    def commit(self, step: int, metadata: dict) -> int:
        """Write metadata to the slot for *step*, using invalidation protocol.

        Returns the number of payload bytes written.
        """
        slot_idx = step % self.NUM_SLOTS
        step_view, len_view, payload_view = self._slots[slot_idx]

        payload = pickle.dumps(metadata)
        assert len(payload) <= self.MAX_PAYLOAD, (
            f"Metadata payload ({len(payload)} bytes) exceeds "
            f"slot capacity ({self.MAX_PAYLOAD} bytes)"
        )

        # 1. Invalidate slot
        step_view.fill_(self._INVALID_STEP)
        # 2. Write payload
        payload_tensor = torch.frombuffer(payload, dtype=torch.uint8)
        payload_view[: len(payload)].copy_(payload_tensor)
        # 3. Write length
        len_view.fill_(len(payload))
        # 4. Mark valid (commit point)
        step_view.fill_(step)
        return len(payload)

    def load_latest(self) -> tuple[int, dict] | None:
        """Return (step, metadata) from the most recent valid slot, or None."""
        best_step, best_idx = -1, -1
        for i, (step_view, _, _) in enumerate(self._slots):
            s = step_view.item()
            if s > best_step:
                best_step, best_idx = s, i
        if best_idx < 0:
            return None
        _, len_view, payload_view = self._slots[best_idx]
        data_len = len_view.item()
        payload_bytes = bytes(payload_view[:data_len].numpy())
        return best_step, pickle.loads(payload_bytes)

    def load_at_step(self, step: int) -> dict:
        """Return metadata committed at *step*. Raises if no slot matches."""
        for step_view, len_view, payload_view in self._slots:
            if step_view.item() != step:
                continue
            data_len = len_view.item()
            payload_bytes = bytes(payload_view[:data_len].numpy())
            return pickle.loads(payload_bytes)
        raise RuntimeError(
            f"No slot in metadata circular buffer holds step={step}"
        )

def get_fqns(model, name):
    fqns = _get_fqns(model, name)
    return next(iter(fqns))

class RmpManager:
    def __init__(
        self,
        leto_config,
        model_parts,
        optimizers,
        states,
        lr_schedulers,
        dataloader,
        device):
        self.enabled = leto_config.enable_rmp_gpu
        self.disable_resilient_opt = leto_config.disable_resilient_opt
        self.device = device

        if self.enabled or leto_config.enable_rmp_cpu:
            # RMP client configuration
            # Server address: localhost:{rmp_server_port + local_rank}
            base_port = leto_config.rmp_server_port
            local_rank = device.index if hasattr(device, 'index') else 0
            rmp_port = base_port + local_rank
            self.rmp_server_address = f"localhost:{rmp_port}"

            # Connect to RMP server
            self.rmp_client = RmpClient(self.rmp_server_address)

        self._grad_allocator: RmpGradientAllocator | None = None
        self._allocated: bool = True  # set by maybe_init

        if not self.enabled:
            return

        self.skip_commit = leto_config.enable_skip_commit
        self.model_parts = model_parts
        self.optimizers = optimizers
        self.states = states
        self.states.update({
            DATALOADER: dataloader,
            LR_SCHEDULER: lr_schedulers
        })
        self.lr_schedulers = lr_schedulers

    def maybe_init(self, buffer_device):
        if not self.enabled:
            return False

        tid_to_name = {}
        name_to_meta_tensor = {}
        tid_to_gpu_tensor = {}

        # Collect all tensors that need to be allocated
        tensor_specs = []

        for model_part in self.model_parts:
            for name, param in chain(
                model_part.named_parameters(),
                model_part.named_buffers()
            ):
                name = get_fqns(model_part, name)
                tid_to_name[id(param)] = name
                name_to_meta_tensor[name] = param

                # Get local tensor shape and dtype
                local_tensor = _to_local_tensor(param)
                tensor_specs.append(TensorSpec(
                    name=name,
                    shape=local_tensor.shape,
                    dtype=local_tensor.dtype,
                    device=self.device.index if hasattr(self.device, 'index') else 0
                ))

        for name, meta_tensor in self.optimizers.state_dict().items():
            if isinstance(meta_tensor, torch.Tensor):
                tid_to_name[id(meta_tensor)] = name
                name_to_meta_tensor[name] = meta_tensor

                # Get local tensor shape and dtype
                local_tensor = _to_local_tensor(meta_tensor)
                tensor_specs.append(TensorSpec(
                    name=name,
                    shape=local_tensor.shape,
                    dtype=local_tensor.dtype,
                    device=self.device.index if hasattr(self.device, 'index') else 0
                ))

        # Request all tensors from RMP server in one batch
        shared_tensors, allocated = self.rmp_client.get_or_allocate_tensors(tensor_specs)
        self._allocated = allocated

        # Map shared tensors to their IDs, wrapping with DTensor if needed
        for tid, name in tid_to_name.items():
            meta_tensor = name_to_meta_tensor[name]
            shared_tensor = shared_tensors[name]
            # Wrap with DTensor if the original was a DTensor
            tid_to_gpu_tensor[tid] = _to_dtensor(shared_tensor, meta_tensor)

        # Update optimizer states
        for optimizer in self.optimizers:
            optimizer.state = {
                param: {
                    name: tid_to_gpu_tensor[id(meta_tensor)]
                    for name, meta_tensor in param_state.items()
                }
                for param, param_state in optimizer.state.items()
            }

        # Apply function to replace tensors in model
        def _apply_fn(tensor):
            return tid_to_gpu_tensor[id(tensor)]

        self._meta_buffer = MetadataCircularBuffer(self.rmp_client)

        if allocated:
            logger.info(f"New tensors allocated on RMP server.")
        else:
            logger.info(f"Got existing tensors from RMP server.")

        for model_part in self.model_parts:
            model_part._apply(_apply_fn)
            if allocated:
                with torch.no_grad():
                    model_part.init_weights(buffer_device=buffer_device)
            model_part.train()

        self.rmp_client.set_allocation_flag(FLAG_KIND_GPU)

        return not allocated

    def maybe_commit(self, step: int):
        if not self.enabled or self.skip_commit:
            return
        t0 = time.perf_counter()

        # OPTIM is not required: load_cpu_metadata never applies it (see
        # comment in load_cpu_metadata). The non-tensor optim state is
        # reconstructed by the next lr_scheduler.step() call.
        metadata = {"TRAIN": stateful_to_state_dict(self.states)}
        num_bytes = self._meta_buffer.commit(step, metadata)

        # Diagnostic: log per-step commit wall time when running the
        # "only_resilient_commit" config of awsdev/performance/run_perf.sh
        # (= disable_resilient_opt=True).
        if self.disable_resilient_opt:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                f"_sync_commit step={step} bytes={num_bytes} "
                f"elapsed_ms={elapsed_ms:.2f}"
            )

    def has_committed_metadata(self) -> bool:
        """True if any rank-local metadata slot holds a valid commit."""
        if not self.enabled:
            return False
        return self._meta_buffer.load_latest() is not None

    def load_cpu_metadata(self, resume_step):
        if self.skip_commit:
            return
        committed_metadata = self._meta_buffer.load_at_step(resume_step)
        logger.info(f"Loaded metadata from circular buffer (step={resume_step})")
        state_dict_to_stateful(self.states, committed_metadata["TRAIN"])

        # Tensor states (params, exp_avg, exp_avg_sq, step) are already
        # correct in RMP-GPU from the previous active — do NOT route them
        # through optimizer.load_state_dict, whose set_optimizer_state_dict
        # path can mint fresh CUDA tensors and sever the RMP backing that
        # ResilientOptimizer's bind() then captures (same class of bug as
        # the gemini in_mem_state.py fix).  The non-tensor optim state
        # (param_groups: lr, betas, ...) is reconstructed by the next
        # lr_scheduler.step() call (the scheduler's last_epoch was just
        # restored above via state_dict_to_stateful), so we don't need to
        # apply committed_metadata["OPTIM"] either.

    def init_gradient_allocator(self, collective_manager, model_parts):
        """Register RMP gradient allocator on the collective manager."""
        if not self.enabled:
            return
        if self._grad_allocator is None:
            self._grad_allocator = RmpGradientAllocator(
                rmp_client=self.rmp_client,
                device=self.device,
                allocated=self._allocated,
            )
        self._grad_allocator.register(collective_manager, model_parts)

    def restore_param_gradients(self, model_parts):
        """Restore RMP-backed gradient tensors to param.grad.

        Creates the gradient allocator early (if needed) so that its
        prefetched tensors are available, then assigns them to param.grad.
        Must be called before init_gradient_allocator().register().
        """
        if not self.enabled:
            return
        if self._grad_allocator is None:
            self._grad_allocator = RmpGradientAllocator(
                rmp_client=self.rmp_client,
                device=self.device,
                allocated=self._allocated,
            )
        self._grad_allocator.restore_param_gradients(model_parts)

    def get_or_allocate_cpu_memory(self, name, num_bytes):
        return self.rmp_client.get_or_allocate_cpu_memory(name, num_bytes)

    def cleanup(self):
        """Clean up resources (e.g., close RMP client connection)."""
        if self.enabled and self.rmp_client is not None:
            self.rmp_client.close()
            self.rmp_client = None
