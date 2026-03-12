from itertools import chain

import torch
from torch import nn
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import _get_fqns

from torchtitan.components.checkpoint import (
    ModelWrapper,
    DATALOADER,
    LR_SCHEDULER,
)
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import (
    _to_local_tensor,
    _to_dtensor,
    stateful_to_state_dict,
    state_dict_to_stateful
)
from leto.rmp.client import RmpClient, TensorSpec

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

        if not self.enabled:
            return

        self.model_parts = model_parts
        self.optimizers = optimizers
        self.states = states
        self.states.update({
            DATALOADER: dataloader,
            LR_SCHEDULER: lr_schedulers
        })

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
            self.maybe_commit()
        else:
            self._load_cpu_metadata()

        return not allocated

    def maybe_commit(self):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        optim_metadata = {}
        metadata = {
            "TRAIN": stateful_to_state_dict(self.states),
            "OPTIM": optim_metadata
        }

        for name, v in self.optimizers.state_dict().items():
            if isinstance(v, torch.Tensor):
                pass
            else:
                optim_metadata[name] = v
        self.rmp_client.commit_metadata(metadata)

    def _load_cpu_metadata(self):
        committed_metadata = self.rmp_client.get_committed_metadata()
        state_dict_to_stateful(self.states, committed_metadata["TRAIN"])

        optim_state_dict = self.optimizers.state_dict()
        optim_state_dict.update(committed_metadata["OPTIM"])
        self.optimizers.load_state_dict(optim_state_dict)

    def get_or_allocate_cpu_memory(self, name, num_bytes):
        return self.rmp_client.get_or_allocate_cpu_memory(name, num_bytes)

    def cleanup(self):
        """Clean up resources (e.g., close RMP client connection)."""
        if self.enabled and self.rmp_client is not None:
            self.rmp_client.close()
            self.rmp_client = None
