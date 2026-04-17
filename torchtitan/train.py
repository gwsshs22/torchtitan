# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import json
import os
import random
import time
from datetime import timedelta
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint.stateful
import torch.distributed.tensor._random as dtensor_random
from torch.distributed.elastic.multiprocessing.errors import record

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.ft import FTManager, maybe_semi_sync_training
from torchtitan.components.rmp_manager import RmpManager
from torchtitan.components.skip_shape_infer import maybe_record_stage_inputs
from torchtitan.config import ConfigManager, JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)

# Optional leto integration
try:
    from leto.launch.worker_controller_client import (
        register_training_process,
        report_event,
        report_duration,
        EVENT_PROCESS_STARTED,
        EVENT_TRAINING_STARTED,
        EVENT_STEP_DONE,
        DURATION_ITERATION,
    )
    _LETO_AVAILABLE = True
except ImportError:
    _LETO_AVAILABLE = False


class ExpertDistTracker:
    """Tracks per-step expert token distribution for the first MoE layer."""

    def __init__(self, model: torch.nn.Module, dump_folder: str, rank: int):
        self._moe_layer = None
        # Find first MoE layer
        from torchtitan.models.moe.moe import MoE
        for module in model.modules():
            if isinstance(module, MoE):
                self._moe_layer = module
                break
        if self._moe_layer is None:
            logger.warning("[ExpertDist] No MoE layer found, disabling tracker")
            return

        base_dir = os.environ.get("LETO_LOGS_DIR") or dump_folder
        experts_dir = os.path.join(base_dir, "experts")
        os.makedirs(experts_dir, exist_ok=True)
        self._csv_path = os.path.join(experts_dir, f"expert_dist_rank_{rank}.csv")
        num_experts = self._moe_layer.tokens_per_expert.numel()
        header = "step," + ",".join(f"expert_{i}" for i in range(num_experts))
        with open(self._csv_path, "w") as f:
            f.write(header + "\n")
        self._prev_tokens = self._moe_layer.tokens_per_expert.clone()

    def begin_step(self):
        if self._moe_layer is None:
            return
        self._prev_tokens.copy_(self._moe_layer.tokens_per_expert)

    def end_step(self, step: int):
        if self._moe_layer is None:
            return
        delta = self._moe_layer.tokens_per_expert - self._prev_tokens
        counts = delta.long().tolist()
        line = f"{step}," + ",".join(str(c) for c in counts)
        with open(self._csv_path, "a") as f:
            f.write(line + "\n")


class Trainer(torch.distributed.checkpoint.stateful.Stateful):
    # core configs
    job_config: JobConfig
    parallel_dims: ParallelDims
    train_spec: train_spec_module.TrainSpec

    # swappable training components in TrainSpec
    tokenizer: train_spec_module.BaseTokenizer | None
    dataloader: train_spec_module.BaseDataLoader
    model_parts: list[torch.nn.Module]
    loss_fn: train_spec_module.LossFunction
    optimizers: train_spec_module.OptimizersContainer
    lr_schedulers: train_spec_module.LRSchedulersContainer
    validator: train_spec_module.BaseValidator
    metrics_processor: train_spec_module.MetricsProcessor
    model_args: train_spec_module.BaseModelArgs

    # non-swappable training components
    checkpointer: CheckpointManager
    ft_manager: FTManager

    # runtime utilities
    device: torch.device
    gc_handler: utils.GarbageCollection
    train_context: dist_utils.TrainContext
    gradient_accumulation_steps: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    # additional training states
    step: int
    ntokens_seen: int

    # stage input recorder (for recording stage inputs/outputs)
    stage_input_recorder: Any = None

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, job_config: JobConfig):
        torch._C._log_api_usage_once("torchtitan.train")

        self.job_config = job_config

        logger.info(f"Starting job: {job_config.job.description}")

        if job_config.experimental.custom_import:
            importlib.import_module(job_config.experimental.custom_import)

        # Run the configurable initialization sequence (see init.py)
        from torchtitan.init import run_init_sequence

        ctx = run_init_sequence(job_config)
        ctx.apply_to(self)

        # Replace InitContext with Trainer as the stateful train_state object
        # so that future checkpoint saves/RMP commits use Trainer.state_dict().
        if hasattr(self.rmp_manager, "states"):
            self.rmp_manager.states["train_state"] = self
        if hasattr(self, "checkpointer") and self.checkpointer is not None:
            self.checkpointer.states["train_state"] = self

        # Resilient optimizer recovery requires Trainer methods (_resilient_opt_recover),
        # so it must happen after apply_to.
        if job_config.leto.enable_rmp_gpu and self.rmp_restored:
            self.rmp_manager.restore_param_gradients(self.model_parts)
            self._resilient_opt_recover()

    def maybe_inject_fault(self) -> bool:
        """Inject a fault at specific training steps (worker-side, step-based).

        When enabled, every `fault_injection_step_interval` steps, a deterministic
        hash selects a target rank. Depending on `fault_injection_step_rank_mode`:
          - "single": only the target rank faults
          - "tp": all ranks in the same TP group as the target rank fault
          - "fsdp": all ranks in the same FSDP group as the target rank fault

        If `fault_injection_step_barrier` is True, all ranks call dist.barrier()
        before the faulting rank(s) raise an exception.

        Returns:
            True if a fault was triggered (and raise_error is False), False otherwise.
        """
        import hashlib

        leto_cfg = self.job_config.leto

        if not leto_cfg.fault_injection_step_enabled:
            return False
        if leto_cfg.fault_injection_step_interval <= 0:
            return False

        # self.step is 1-indexed (incremented at start of training loop)
        step = self.step
        interval = leto_cfg.fault_injection_step_interval

        # Respect start_step and end_step bounds
        start_step = leto_cfg.fault_injection_start_step
        end_step = leto_cfg.fault_injection_end_step
        if start_step > 0 and step < start_step:
            return False
        if end_step > 0 and step > end_step:
            return False

        if step % interval != 0:
            return False

        # Skip fault on the first step after checkpoint restore to prevent
        # infinite loop (fault at step N -> restore to N-1 -> step N faults again)
        if self._restored_step > 0 and step == self._restored_step + 1:
            logger.info(
                f"[STEP FAULT INJECTION] Skipping fault at step {step} "
                f"(first step after restore from step {self._restored_step})"
            )
            return False

        # Determine target rank
        world_size = self.parallel_dims.world_size
        global_rank = int(os.environ["RANK"])

        # Use explicit target rank if set, otherwise select via deterministic hash
        if leto_cfg.fault_injection_target_rank >= 0:
            target_rank = leto_cfg.fault_injection_target_rank
        else:
            seed = leto_cfg.fault_injection_step_seed
            h = hashlib.sha256(f"{seed}:{step}".encode()).hexdigest()
            target_rank = int(h, 16) % world_size

        # Determine if this rank should fault based on rank_mode
        rank_mode = leto_cfg.fault_injection_step_rank_mode
        should_fault = False

        if rank_mode == "single":
            should_fault = (global_rank == target_rank)
        elif rank_mode == "tp":
            mesh_tensor = self.parallel_dims.get_mesh("tp").mesh
            if mesh_tensor.ndim == 1:
                mesh_tensor = mesh_tensor.unsqueeze(0)
            for group_idx in range(mesh_tensor.shape[0]):
                group_ranks = mesh_tensor[group_idx].tolist()
                if target_rank in group_ranks and global_rank in group_ranks:
                    should_fault = True
                    break
        elif rank_mode == "fsdp":
            mesh_tensor = self.parallel_dims.get_mesh("fsdp").mesh
            if mesh_tensor.ndim == 1:
                mesh_tensor = mesh_tensor.unsqueeze(0)
            for group_idx in range(mesh_tensor.shape[0]):
                group_ranks = mesh_tensor[group_idx].tolist()
                if target_rank in group_ranks and global_rank in group_ranks:
                    should_fault = True
                    break

        # All ranks mark the fault so nocommit reset happens everywhere
        self._prev_step_faulted = True

        if not should_fault:
            return False

        torch.cuda.synchronize()
        logger.info(
            f"[STEP FAULT INJECTION] step={step}, rank={global_rank}, "
            f"target_rank={target_rank}, mode={rank_mode}"
        )

        if leto_cfg.fault_injection_raise_error:
            time.sleep(3.0)
            raise RuntimeError(
                f"[STEP FAULT INJECTION] Fault at step {step} on rank {global_rank}"
            )

        # Skip optimizer.step() instead of raising
        logger.info(
            f"[STEP FAULT INJECTION] Skipping optimizer.step() at step {step} "
            f"on rank {global_rank}"
        )
        return True

    def maybe_reset_after_fault(self):
        """Reset dataloader and lr scheduler if the previous step faulted and nocommit is enabled."""
        if not self._prev_step_faulted:
            return

        logger.info(
            f"[NOCOMMIT] Resetting dataloader and lr scheduler "
            f"after fault at step {self.step - 1}"
        )
        self._data_iterator = self.batch_generator(self.dataloader)
        lr_steps = (
            self.job_config.training.max_steps
            if self.job_config.training.max_steps > 0
            else self.job_config.training.steps
        )
        self.lr_schedulers = self.train_spec.build_lr_schedulers_fn(
            self.optimizers, self.job_config.lr_scheduler, lr_steps
        )
        self._prev_step_faulted = False

    def _resilient_opt_recover(self):
        """Recover resilient optimizer state on RMP resume.

        Each rank reads its local step from the metadata buffer, then all
        ranks agree on the minimum step via an all-reduce.  The minimum
        is used as resume_step for maybe_recover().
        """
        result = self.rmp_manager._meta_buffer.load_latest()
        if result is None:
            raise RuntimeError("Cannot recover: no committed metadata")
        local_step = result[0]

        # MIN all-reduce across all ranks to find the globally consistent step
        step_tensor = torch.tensor([local_step], dtype=torch.int64, device=self.device)
        dist.all_reduce(step_tensor, op=dist.ReduceOp.MIN)
        resume_step = step_tensor.item()

        logger.info(
            f"[ResilientOpt] local_step={local_step}, "
            f"resume_step={resume_step} (global min)"
        )

        recovered = self._resilient_opt.maybe_recover(resume_step)
        if recovered:
            logger.info(f"[ResilientOpt] Recovery completed at step {resume_step}")
        else:
            logger.info(f"[ResilientOpt] No recovery needed at step {resume_step}")
        self.lr_schedulers.step()

    def maybe_check_step_consistency(self, data_iterator):
        if not self.job_config.leto.fault_injection_step_enabled:
            return

        if self.job_config.leto.fault_injection_step_barrier:
            return

        if getattr(self, "_step_consistency_checked", False):
            return
        self._step_consistency_checked = True

        # Gather steps from all ranks to check consistency
        local_step = torch.tensor([self.step], dtype=torch.long, device="cuda")
        world_size = dist.get_world_size()
        all_steps = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(world_size)]
        dist.all_gather(all_steps, local_step)

        max_step = max(s.item() for s in all_steps)
        my_step = local_step.item()

        if my_step < max_step:
            if my_step == max_step - 1:
                logger.warning(
                    f"Rank {dist.get_rank()} step {my_step} is behind max step {max_step} by 1. "
                    f"Advancing data_iterator and step to catch up."
                )
                # Advance data iterator by one training step (gradient_accumulation_steps microbatches)
                for _ in range(self.gradient_accumulation_steps):
                    next(data_iterator, None)
                self.step = max_step
            else:
                raise RuntimeError(
                    f"Rank {dist.get_rank()} step {my_step} is behind max step {max_step} "
                    f"by more than 1. Cannot recover automatically."
                )



    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Returns an iterator that processes batches from the data iterator."""
        device_type = utils.device_type
        data_iterator = iter(data_iterable)

        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderExhaustedError() from ex
            input_dict, labels = batch
            ntokens_batch = labels.numel()
            self.ntokens_seen += ntokens_batch
            self.metrics_processor.ntokens_since_last_log += ntokens_batch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            # Move tensors to the appropriate device
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(device_type)
            labels = labels.to(device_type)

            yield input_dict, labels

    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """
        Post-processing hook after data loading and before model forward pass.

        This method processes the raw data from the dataloader and prepares it for
        the model's forward pass. It separates the main input tensor from auxiliary
        inputs and constructs additional keyword arguments (e.g., attention masks).

        This method can be overridden in subclasses to customize data processing
        for different training strategies (e.g., converting tensors to DTensors,
        applying custom transformations, etc.).

        Args:
            input_dict: Dictionary containing tensors from the dataloader. Must
                contain an "input" key with the main input tensor. May contain
                additional keys for auxiliary inputs (e.g., position ids).
            labels: Target labels for the batch.

        Returns:
            A tuple of (inputs, labels, extra_inputs, extra_kwargs) where:
                - inputs: Main input tensor extracted from input_dict["input"].
                - labels: Target labels (unchanged from input parameter).
                - extra_inputs: Dict of auxiliary input tensors (all keys except
                    "input" from input_dict). These are passed to the model forward
                    but are NOT forwarded across pipeline parallel stages.
                - extra_kwargs: Dict of additional keyword arguments for model forward.
                    These ARE forwarded across pipeline parallel stages. Contains
                    attention_masks if flex attention is enabled.

        Note:
            The distinction between extra_inputs and extra_kwargs is important for
            pipeline parallelism: extra_kwargs are forwarded to all pipeline stages,
            while extra_inputs are only available to the first stage.
        """
        inputs = input_dict["input"]
        extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
        # For arguments, like attention_masks, we have to put them in a separate
        # dict as extra_inputs are not forwarded to other stages in PP, but
        # extra_kwargs are.
        extra_kwargs: dict[str, Any] = {}

        attn_type = getattr(self.model_args, "attn_type", "sdpa")
        if attn_type in ["flex", "varlen"]:
            # pyrefly: ignore [not-callable]
            extra_kwargs["attention_masks"] = self.model_parts[0].get_attention_masks(
                input_batch=inputs,
                tokenizer=self.tokenizer,
                extra_inputs=extra_inputs,
            )

        if self.parallel_dims.cp_enabled:
            inputs, labels, extra_kwargs = prepare_context_parallel_input(
                inputs,
                labels,
                extra_kwargs,
                self.parallel_dims.get_mesh("cp"),
                self.device,
                self.job_config.parallelism.context_parallel_load_balancer,
            )

        return inputs, labels, extra_inputs, extra_kwargs

    def forward_backward_step(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, labels
        )

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            with self.train_context():
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs,
                        **extra_inputs,
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )
                else:
                    self.pp_schedule.step(
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            loss = (
                # using sum instead of mean because we already rescale the
                # loss_fn down by a factor of n_microbatches in
                # torchtitan/distributed/pipeline_parallel.py
                torch.sum(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor([-1.0], device=self.device)
            )
        else:
            # Non-PP forward / backward
            assert len(model_parts) == 1
            with self.train_context():
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)
                    loss = self.loss_fn(pred, labels)
                # need to free pred before bwd to avoid peaking memory
                del pred
                loss.backward()

        return loss

    def maybe_dump_optimizer_info(self):
        """Dump optimizer setup (shapes, dtypes, hyperparams) to JSON.

        Controlled by leto.dump_optimizer_info config. Runs once on step 1,
        rank 0 only. The output is used by resilient optimizer tests.
        """
        if not self.job_config.leto.dump_optimizer_info:
            return
        if self.step != 1 or dist.get_rank() != 0:
            return

        import json
        dump = {"optimizers": []}
        for optimizer in self.optimizers:
            opt_info = {
                "class": optimizer.__class__.__name__,
                "defaults": {
                    k: v for k, v in optimizer.defaults.items()
                    if not callable(v) and not isinstance(v, torch.Tensor)
                },
                "param_groups": [],
            }
            for pg in optimizer.param_groups:
                pg_info = {
                    "num_params": len(pg["params"]),
                    "lr": pg.get("lr"),
                    "betas": pg.get("betas"),
                    "eps": pg.get("eps"),
                    "weight_decay": pg.get("weight_decay"),
                    "fused": pg.get("fused"),
                    "foreach": pg.get("foreach"),
                    "params": [],
                }
                for param in pg["params"]:
                    local_p = param._local_tensor if hasattr(param, '_local_tensor') else param
                    p_info = {
                        "shape": list(local_p.shape),
                        "dtype": str(local_p.dtype),
                        "device": str(local_p.device),
                        "requires_grad": param.requires_grad,
                    }
                    if param in optimizer.state:
                        state = optimizer.state[param]
                        state_info = {}
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                local_v = v._local_tensor if hasattr(v, '_local_tensor') else v
                                state_info[k] = {
                                    "shape": list(local_v.shape),
                                    "dtype": str(local_v.dtype),
                                }
                            else:
                                state_info[k] = v
                        p_info["state"] = state_info

                    if param.grad is not None:
                        local_g = param.grad._local_tensor if hasattr(param.grad, '_local_tensor') else param.grad
                        p_info["grad"] = {
                            "shape": list(local_g.shape),
                            "dtype": str(local_g.dtype),
                        }

                    pg_info["params"].append(p_info)
                opt_info["param_groups"].append(pg_info)
            dump["optimizers"].append(opt_info)

        dump_path = os.path.join(
            self.job_config.job.dump_folder, "optimizer_info.json"
        )
        with open(dump_path, "w") as f:
            json.dump(dump, f, indent=2)
        logger.info(f"Dumped optimizer info to {dump_path}")

    def train_step(
        self, data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        if self._cpu_snapshot_opt is not None:
            self._cpu_snapshot_opt.begin_snapshot()
        # Save the current step learning rate for logging
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        if self._expert_dist_tracker is not None:
            self._expert_dist_tracker.begin_step()

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        accumulated_losses = []
        # If data runs out during gradient accumulation, that
        # entire step will not be executed.
        for _microbatch in range(self.gradient_accumulation_steps):
            # pyrefly: ignore [no-matching-overload]
            input_dict, labels = next(data_iterator)
            loss = self.forward_backward_step(input_dict, labels)
            accumulated_losses.append(loss.detach())

        # All microbatches' forward/backward have been dispatched. Kick off the
        # async RMP metadata commit now so pickle/shm_write/barrier overlap
        # with clip_grad_norm_ and in-flight GPU work. No-op in sync mode.
        self.rmp_manager.schedule_commit(self.step)

        if self._expert_dist_tracker is not None:
            self._expert_dist_tracker.end_step(self.step)

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()

        fault_triggered = self.maybe_inject_fault()
        # Barrier: all ranks participate (faulting ranks barrier then crash,
        # non-faulting ranks barrier then continue normally)
        if self.job_config.leto.fault_injection_step_barrier:
            dist.barrier()
        self.rmp_manager.maybe_commit(self.step)

        if not fault_triggered:
            if self._resilient_opt is not None:
                self._resilient_opt.step()
            elif self._cpu_snapshot_opt is not None:
                self._cpu_snapshot_opt.step()
            else:
                self.optimizers.step()

        self.maybe_dump_optimizer_info()

        self.lr_schedulers.step()



        # Reduce the data collected over gradient accumulation steps.
        loss = torch.sum(torch.stack(accumulated_losses))

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            ft_pg = self.ft_manager.loss_sync_pg
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_mean(loss, loss_mesh, ft_pg),
                dist_utils.dist_max(loss, loss_mesh, ft_pg),
                dist_utils.dist_sum(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                    ft_pg,
                ),
            )
        else:
            global_avg_loss = global_max_loss = loss.detach().item()
            global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
        }
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
            extra_metrics=extra_metrics,
        )

    @record
    def train(self):
        job_config = self.job_config

        self.checkpointer.load(step=job_config.checkpoint.load_step)
        self._restored_step = self.step

        global_batch_size = job_config.training.global_batch_size
        if global_batch_size < 0:
            global_batch_size = job_config.training.local_batch_size * self._batch_degree

        logger.info(
            "Trainer is initialized with "
            f"local batch size {job_config.training.local_batch_size}, "
            f"global batch size {global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {job_config.training.seq_len}, "
            f"total steps {job_config.training.steps} "
            f"(warmup {job_config.lr_scheduler.warmup_steps})"
        )

        if _LETO_AVAILABLE:
            report_event(EVENT_TRAINING_STARTED)
        logger.info(f"Training starts at step {self.step + 1}")

        leaf_folder = (
            ""
            if not self.ft_manager.enabled
            else f"replica_{self.ft_manager.replica_id}"
        )
        with (
            maybe_enable_profiling(
                job_config.profiling,
                global_step=self.step,
                base_folder=job_config.job.dump_folder,
                leaf_folder=leaf_folder,
            ) as torch_profiler,
            maybe_enable_memory_snapshot(
                job_config.profiling,
                global_step=self.step,
                base_folder=job_config.job.dump_folder,
                leaf_folder=leaf_folder,
            ) as memory_profiler,
            maybe_semi_sync_training(
                # pyrefly: ignore [bad-argument-type]
                job_config.fault_tolerance,
                ft_manager=self.ft_manager,
                model=self.model_parts[0],
                n_layers=(
                    self.model_args.n_layers
                    if hasattr(self.model_args, "n_layers")
                    else 0
                ),
                optimizer=self.optimizers,
                fragment_fn=(
                    self.train_spec.fragment_fn
                    if hasattr(self.train_spec, "fragment_fn")
                    else None
                ),
            ),
        ):
            # pyrefly: ignore [bad-argument-type]
            self._data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1
                _iter_start = time.monotonic()

                self.maybe_reset_after_fault()

                # Run validation if validator is available
                if (
                    self.job_config.validation.enable
                    and self.validator.should_validate(self.step)
                ):
                    # pyrefly: ignore [missing-attribute]
                    with self.loss_fn.no_rescale():
                        # pyrefly: ignore [bad-argument-count]
                        self.validator.validate(self.model_parts, self.step)

                self.maybe_check_step_consistency(self._data_iterator)
                # Handle stage input/output recording (before/after first iteration)
                self.stage_input_recorder = maybe_record_stage_inputs(
                    self.model_parts,
                    job_config,
                    self.step,
                    self.stage_input_recorder,
                )

                self.gc_handler.run(self.step)
                self.checkpointer.begin_step(
                    self.step, last_step=(self.step == job_config.training.steps)
                )
                try:
                    self.train_step(self._data_iterator)
                except DataloaderExhaustedError:
                    logger.warning("Ran out of data; last step was canceled.")
                    break

                self.checkpointer.save(
                    self.step, last_step=(self.step == job_config.training.steps)
                )

                # signal the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(
                            seconds=job_config.comm.train_timeout_seconds
                        ),
                        parallel_dims=self.parallel_dims,
                    )

                if _LETO_AVAILABLE:
                    report_duration(DURATION_ITERATION, time.monotonic() - _iter_start, step=self.step)
                    report_event(EVENT_STEP_DONE, step=self.step)


        # Wait for any pending checkpoint tracking to complete
        if hasattr(self, "checkpointer") and self.checkpointer:
            self.checkpointer.wait_for_tracking()

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def should_continue_training(self) -> bool:
        return self.step < self.job_config.training.steps

    def state_dict(self) -> dict[str, Any]:
        state = {
            "step": self.step,
            "ntokens_seen": self.ntokens_seen,
            # RNG states for reproducibility
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        # CUDA and DTensor rng state are omitted when the RmpManager async
        # commit path captures them on the main thread and injects them into
        # the metadata dict directly (see RmpManager.schedule_commit).
        if self._include_rng_in_state_dict:
            state["cuda_rng_state"] = torch.cuda.get_rng_state(self.device)
            rng_tracker = dtensor_random._rng_tracker
            if rng_tracker is not None and hasattr(rng_tracker, "_get_device_state"):
                state["dtensor_rng_state"] = rng_tracker._get_device_state().cpu()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]
        # Restore RNG states if present (for backward compatibility)
        if "torch_rng_state" in state_dict:
            torch.set_rng_state(state_dict["torch_rng_state"])
        if "cuda_rng_state" in state_dict:
            torch.cuda.set_rng_state(state_dict["cuda_rng_state"], self.device)
        if "numpy_rng_state" in state_dict:
            np.random.set_state(state_dict["numpy_rng_state"])
        if "python_rng_state" in state_dict:
            random.setstate(state_dict["python_rng_state"])
        # Restore DTensor RNG tracker state if available
        if "dtensor_rng_state" in state_dict:
            rng_tracker = dtensor_random._rng_tracker
            if rng_tracker is not None and hasattr(rng_tracker, "_set_device_state"):
                rng_tracker._set_device_state(state_dict["dtensor_rng_state"].to(self.device))

    def close(self) -> None:
        if hasattr(self, "_data_iterator") and self._data_iterator is not None:
            self._data_iterator.close()
            self._data_iterator = None
        if hasattr(self, "checkpointer") and self.checkpointer:
            self.checkpointer.close()
        if hasattr(self, "metrics_processor") and self.metrics_processor:
            self.metrics_processor.close()


def main(trainer_class: type[Trainer]) -> None:
    """Main entry point for training with a specified trainer class.

    Args:
        trainer_class: The trainer class to instantiate (e.g., Trainer, FluxTrainer, TorchCommsTrainer)
    """
    init_logger()
    import torchtitan

    logger.info(
        "torchtitan version: %s (0.0.0 means __version__ is not defined correctly).",
        torchtitan.__version__,
    )

    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer: Trainer | None = None

    try:
        trainer = trainer_class(config)

        # TODO(local_tensor): Remove this special case once LocalTensor supports
        # init_weights() and foreach_allgather. In local tensor mode, skip
        # training/checkpointing as the # model is not fully initialized
        if config.comm.mode == "local_tensor":
            logger.info("Local tensor mode enabled - skipping training execution")
            return

        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    if _LETO_AVAILABLE:
            register_training_process(rank=int(os.environ["RANK"]))
            report_event(EVENT_PROCESS_STARTED)
    main(Trainer)
