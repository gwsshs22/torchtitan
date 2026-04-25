from concurrent.futures import Future, ThreadPoolExecutor
from itertools import chain
import pickle
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import _get_fqns
from torch.distributed.tensor import _random as dtensor_random

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

def _ms(t_start, t_end):
    return (t_end - t_start) * 1000

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

        # CPU-only barrier group for metadata commit synchronization
        self._gloo_group = dist.new_group(backend="gloo")
        self._gloo_warmup = dist.barrier(group=self._gloo_group, async_op=True)

        self.rmp_commit_sync = leto_config.rmp_commit_sync
        self._commit_future: Future | None = None
        self._commit_executor: ThreadPoolExecutor | None = None

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

        if not self.skip_commit and not self.rmp_commit_sync:
            self._commit_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="rmp-commit",
            )
            # Warm up: spawn the worker thread eagerly so step 0 doesn't pay
            # thread-creation cost on the critical path.
            self._commit_executor.submit(lambda: None).result()

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

        if allocated:
            self._sync_commit(step=0)
        else:
            self._load_cpu_metadata()

        self.rmp_client.set_allocation_flag(FLAG_KIND_GPU)

        return not allocated

    def schedule_commit(self, step: int):
        """Kick off the async metadata commit on the background worker.

        Must be called from the main thread *after* every microbatch's
        forward/backward has been dispatched for this step. Captures the
        CUDA + DTensor rng state inline (cheap, main thread has the CUDA
        context already) and hands everything else to the worker.
        """
        if not self.enabled or self.skip_commit or self.rmp_commit_sync:
            return
        assert self._commit_future is None, (
            "schedule_commit called twice without an intervening maybe_commit"
        )

        cuda_idx = self.device.index if hasattr(self.device, "index") else 0
        cuda_gen = torch.cuda.default_generators[cuda_idx]
        # CPU ByteTensor under gen->mutex_; the Philox offset reflects every
        # host-side rng dispatch for this step.
        cuda_rng = cuda_gen.get_state()

        tracker = dtensor_random._rng_tracker
        dtensor_rng = None
        if tracker is not None and hasattr(tracker, "_get_device_state"):
            # Same underlying generator, but we capture via the tracker so the
            # tensor shape matches what Trainer.load_state_dict expects
            # (it calls tracker._set_device_state(state.to(device))).
            dtensor_rng = tracker._get_device_state().cpu()

        self._commit_future = self._commit_executor.submit(
            self._bg_commit, step, cuda_rng, dtensor_rng,
        )

    def _bg_commit(self, step: int, cuda_rng, dtensor_rng):
        """Runs on the background worker thread. Pure CPU — no CUDA calls."""
        t0 = time.perf_counter()

        # Trainer._include_rng_in_state_dict is False in async mode, so this
        # call does not read cuda_rng_state / dtensor_rng_state and therefore
        # makes no CUDA API calls.
        train_state = stateful_to_state_dict(self.states)
        optim_metadata = {
            name: v
            for name, v in self.optimizers.state_dict().items()
            if not isinstance(v, torch.Tensor)
        }
        # Inject rng bytes back under the keys Trainer.load_state_dict expects.
        train_state["cuda_rng_state"] = cuda_rng
        if dtensor_rng is not None:
            train_state["dtensor_rng_state"] = dtensor_rng
        metadata = {"TRAIN": train_state, "OPTIM": optim_metadata}

        t_snapshot = time.perf_counter()
        num_bytes = self._meta_buffer.commit(step, metadata)
        t_commit = time.perf_counter()

        if self._gloo_warmup is not None:
            self._gloo_warmup.wait()
            self._gloo_warmup = None
        dist.barrier(group=self._gloo_group)
        t_barrier = time.perf_counter()

        logger.info(
            f"Metadata commit (async): step={step}, {num_bytes} bytes, "
            f"snapshot={_ms(t0, t_snapshot):.2f} ms, "
            f"shm_write={_ms(t_snapshot, t_commit):.2f} ms, "
            f"barrier={_ms(t_commit, t_barrier):.2f} ms, "
            f"total={_ms(t0, t_barrier):.2f} ms"
        )

    def _sync_commit(self, step: int):
        """Inline commit on the main thread. Used when rmp_commit_sync=True
        and for the init-time bootstrap commit in maybe_init."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optim_metadata = {}
        metadata = {
            "TRAIN": stateful_to_state_dict(self.states),
            "OPTIM": optim_metadata,
        }

        for name, v in self.optimizers.state_dict().items():
            if not isinstance(v, torch.Tensor):
                optim_metadata[name] = v

        t_snapshot = time.perf_counter()

        num_bytes = self._meta_buffer.commit(step, metadata)

        t_commit = time.perf_counter()

        # Ensure all ranks have committed metadata before any rank proceeds
        if self._gloo_warmup is not None:
            self._gloo_warmup.wait()
            self._gloo_warmup = None
        dist.barrier(group=self._gloo_group)

        t_barrier = time.perf_counter()
        logger.info(
            f"Metadata commit: step={step}, {num_bytes} bytes, "
            f"snapshot={_ms(t0, t_snapshot):.2f} ms, "
            f"shm_write={_ms(t_snapshot, t_commit):.2f} ms, "
            f"barrier={_ms(t_commit, t_barrier):.2f} ms, "
            f"total={_ms(t0, t_barrier):.2f} ms"
        )

    def maybe_commit(self, step: int):
        if not self.enabled or self.skip_commit:
            return
        if self.rmp_commit_sync:
            self._sync_commit(step)
            return
        assert self._commit_future is not None, (
            "schedule_commit must be called before maybe_commit in async mode"
        )
        self._commit_future.result()  # propagates worker exceptions
        self._commit_future = None
        torch.cuda.synchronize()

    def _load_cpu_metadata(self):
        if self.skip_commit:
            return
        result = self._meta_buffer.load_latest()
        if result is None:
            raise RuntimeError("No committed metadata found in circular buffer")
        step, committed_metadata = result
        logger.info(f"Loaded metadata from circular buffer (step={step})")
        state_dict_to_stateful(self.states, committed_metadata["TRAIN"])

        optim_state_dict = self.optimizers.state_dict()
        optim_state_dict.update(committed_metadata["OPTIM"])
        self.optimizers.load_state_dict(optim_state_dict)

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
        if self._commit_future is not None:
            self._commit_future.result()
            self._commit_future = None
        if self._commit_executor is not None:
            self._commit_executor.shutdown(wait=True)
            self._commit_executor = None
        if self.enabled and self.rmp_client is not None:
            self.rmp_client.close()
            self.rmp_client = None
