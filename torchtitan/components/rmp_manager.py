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
from torchtitan.tools.utils import (
    _to_local_tensor,
    _to_dtensor
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
        self.enabled = leto_config.enable_rmp
        if not self.enabled:
            return

        self.model_wrapper = ModelWrapper(model_parts)
        self.model_parts = model_parts
        self.optimizers = optimizers
        self.states = states
        self.states.update({
            DATALOADER: dataloader,
            LR_SCHEDULER: lr_schedulers
        })
        self.device = device

    def maybe_init(self, buffer_device):
        if not self.enabled:
            return

        tid_to_name = {}
        name_to_meta_tensor = {}
        tid_to_gpu_tensor = {}

        for model_part in self.model_parts:
            for name, param in chain(
                model_part.named_parameters(),
                model_part.named_buffers()
            ):
                name = get_fqns(model_part, name)
                tid_to_name[id(param)] = name
                name_to_meta_tensor[name] = param

        for name, meta_tensor in self.optimizers.state_dict().items():
            if isinstance(meta_tensor, torch.Tensor):
                tid_to_name[id(meta_tensor)] = name
                name_to_meta_tensor[name] = meta_tensor

        # for optimizer in self.optimizers:
        #     for param, param_state in optimizer.state.items():
        #         for name, meta_tensor in param_state.items():
        #             tid_to_name[id(meta_tensor)] = name
        #             name_to_meta_tensor[name] = param
        #             print(f"name={name}, meta_tensor={meta_tensor}")
        #             test_map[id(meta_tensor)]


        for tid, name in tid_to_name.items():
            meta_tensor = name_to_meta_tensor[name]
            tensor = torch.zeros_like(_to_local_tensor(meta_tensor), device=self.device)
            tid_to_gpu_tensor[tid] = _to_dtensor(tensor, meta_tensor)

        for optimizer in self.optimizers:
            optimizer.state = {
                param: {
                    name: tid_to_gpu_tensor[id(meta_tensor)]
                    for name, meta_tensor in param_state.items()
                }
                for param, param_state in optimizer.state.items()
            }

        def _apply_fn(tensor):
            return tid_to_gpu_tensor[id(tensor)]

        for model_part in self.model_parts:
            model_part._apply(_apply_fn)
            model_part.init_weights(buffer_device=buffer_device)
            model_part.train()

    def maybe_commit(self):
        if not self.enabled:
            return
