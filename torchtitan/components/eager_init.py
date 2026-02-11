import torch
import torch.distributed as dist
from torchtitan.tools.logging import logger


def maybe_eager_init(eager_init_list, parallel_dims, device: torch.device):
    if len(eager_init_list) == 0 or "none" in eager_init_list:
        logger.info("Eager init disabled")
        return

    logger.info(f"Eager init list={eager_init_list}")
    enable_all = "all" in eager_init_list

    if enable_all or "nccl" in eager_init_list:
        warmup_tensor = torch.zeros(1, device=device)
        names = []
        for name, mesh in parallel_dims.get_all_one_dimensional_meshes().items():
            pg = mesh.get_group()
            if pg is not None:
                dist.all_reduce(warmup_tensor, group=pg)
                names.append(name)
        logger.info(f"Eagerly initialized NCCL for '{names}' meshes")

    torch.cuda.synchronize()
