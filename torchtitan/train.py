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
from typing import Any, Iterable, Optional

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
        kill_standby_for_oom_safeguard,
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

        self._oom_safeguard_installed = False

        # State for kernel_trap fault injection. Updated each iter regardless
        # of whether the fault fires, so the next fault step has a recent
        # launches-per-iter measurement to randomize against.
        self._kernel_trap_prev_count: Optional[int] = None
        self._kernel_trap_last_k: int = 0

    @staticmethod
    def _seeded_offset(seed: int, *parts) -> int:
        """Stable hash-based RNG. Same inputs always yield the same int."""
        import hashlib
        s = f"{seed}|" + "|".join(str(p) for p in parts)
        return int(hashlib.sha256(s.encode()).hexdigest(), 16)

    def _kernel_trap_update_k(self) -> int:
        """Update ``self._kernel_trap_last_k`` from the previous iter's
        matching-kernel launches, and return the current matching-launch
        count. Returns 0 if the NVBit tool isn't loaded."""
        try:
            from leto.fault_injection import kernel_trap
        except Exception:
            return 0
        if not kernel_trap.is_available():
            return 0
        current = kernel_trap.get_count()
        if self._kernel_trap_prev_count is not None:
            self._kernel_trap_last_k = max(0, current - self._kernel_trap_prev_count)
        self._kernel_trap_prev_count = current
        return current

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

        # Track per-iter matching-kernel launch count for kernel_trap mode. We
        # call this every step so that on a fault step we have a recent K.
        pre_count = self._kernel_trap_update_k() if leto_cfg.fault_injection_kernel_trap else 0

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
        elif rank_mode == "prob":
            # Each rank independently decides whether to fault this step,
            # deterministic per (seed, step, global_rank) so the schedule is
            # reproducible across restarts. target_rank is unused in this mode;
            # with p over `world_size` ranks, ~p*world_size ranks fault per step.
            prob = leto_cfg.fault_injection_step_prob
            if prob >= 1.0:
                should_fault = True
            elif prob > 0.0:
                rank_seed = leto_cfg.fault_injection_step_seed
                draw = (
                    self._seeded_offset(rank_seed, step, global_rank, "prob")
                    % 1_000_000
                ) / 1_000_000.0
                should_fault = draw < prob
        elif rank_mode == "random":
            # Per (seed, step) deterministic 50/50 between two fault kinds:
            #   kind == 0 -> "single" NVBit kernel_trap on target_rank
            #   kind == 1 -> target_rank's whole FSDP group os._exit(1)
            #               (hard process death -> master restarts from the
            #               last checkpoint; os._exit so NCCL peers notice in
            #               ~8s instead of the ~180s heartbeat timeout).
            rand_seed = leto_cfg.fault_injection_step_seed
            kind = self._seeded_offset(rand_seed, step, "kind") % 2
            if kind == 0:
                should_fault = (global_rank == target_rank)
            else:
                mesh_tensor = self.parallel_dims.get_mesh("fsdp").mesh
                if mesh_tensor.ndim == 1:
                    mesh_tensor = mesh_tensor.unsqueeze(0)
                for group_idx in range(mesh_tensor.shape[0]):
                    group_ranks = mesh_tensor[group_idx].tolist()
                    if target_rank in group_ranks and global_rank in group_ranks:
                        logger.info(
                            f"[STEP FAULT INJECTION] random->fsdp os._exit(1): "
                            f"step={step}, rank={global_rank}, "
                            f"target_rank={target_rank}"
                        )
                        os._exit(1)

        # kernel_trap lets optimizer.step() run normally (the armed launch
        # traps inside the fused-AdamW kernel and kills the process). No
        # dataloader/lr-scheduler reset is needed and setting the flag here
        # would race against the async trap and produce spurious NOCOMMIT
        # logs before the CUDA error surfaces.
        if not leto_cfg.fault_injection_kernel_trap:
            # All ranks mark the fault so nocommit reset happens everywhere
            self._prev_step_faulted = True

        if not should_fault:
            return False

        torch.cuda.synchronize()
        logger.info(
            f"[STEP FAULT INJECTION] step={step}, rank={global_rank}, "
            f"target_rank={target_rank}, mode={rank_mode}"
        )

        if leto_cfg.fault_injection_kernel_trap:
            from leto.fault_injection import kernel_trap
            if not kernel_trap.is_available():
                raise RuntimeError(
                    "fault_injection_kernel_trap requires adam_trap.so to be loaded "
                    "via CUDA_INJECTION64_PATH. Set fault_injection.mode='kernel_trap' "
                    "in your job YAML so leto wires the env automatically."
                )
            k = max(1, self._kernel_trap_last_k)
            seed = leto_cfg.fault_injection_step_seed
            # Include global_rank so concurrently-faulting ranks (prob/random
            # modes) trap at *different* launch positions within the optimizer
            # step — a diffuse blast radius rather than every rank tearing the
            # same chunk. Deterministic per (seed, step, rank).
            launch_offset = self._seeded_offset(seed, step, global_rank, "launch") % k
            # arm() is 1-indexed; pre_count is the count BEFORE this iter's
            # optimizer step starts. The (launch_offset+1)-th matching launch
            # in this iter will trap.
            target_count = int(pre_count) + int(launch_offset) + 1
            kernel_trap.arm(target_count)
            logger.info(
                f"[KERNEL_TRAP] armed: step={step}, rank={global_rank}, "
                f"pre_count={pre_count}, K={k}, launch_offset={launch_offset}, "
                f"target_count={target_count}"
            )
            # Do NOT return True — we want optimizer.step() to run so the
            # adam kernel actually launches and trips the trap.
            return False

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

    def maybe_alloc_for_oom_test(self) -> None:
        """OOM-safeguard test hook: at the configured step, allocate
        `oom_test_alloc_mb` MiB of GPU memory in 64 MiB chunks (a list of
        fresh tensors). Each fresh-size 64 MiB tensor causes a cache-miss
        expansion, so the reservation-aware FreeMemoryCallback should fire on
        each one whenever the allocation would push effective free GPU memory
        below `progressive_reservation_margin_mb`.

        The tensors are held in `self._oom_tensors` so they stay alive
        for the rest of the step. They're freed naturally when the
        attribute goes out of scope or is overwritten on a future call."""
        leto_cfg = self.job_config.leto
        if leto_cfg.oom_test_alloc_step <= 0 or leto_cfg.oom_test_alloc_mb <= 0:
            return
        if self.step != leto_cfg.oom_test_alloc_step:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return

        free_b, total_b = torch.cuda.mem_get_info()
        free_mb_before = int(free_b // (1024 * 1024))
        total_mb = int(total_b // (1024 * 1024))
        alloc_mb = int(leto_cfg.oom_test_alloc_mb)

        chunk_mb = 64
        chunk_floats = (chunk_mb * 1024 * 1024) // 4  # float32 = 4 bytes
        n_chunks = (alloc_mb + chunk_mb - 1) // chunk_mb  # ceil
        logger.info(
            f"[oom_test] rank={rank} step={self.step}: free={free_mb_before}MiB "
            f"total={total_mb}MiB; allocating {n_chunks} x {chunk_mb}MiB "
            f"(target {alloc_mb}MiB total, reservation margin="
            f"{leto_cfg.progressive_reservation_margin_mb}MiB)"
        )

        from torchtitan.components.mem import get_num_kill_standby_called

        self._oom_tensors: list[torch.Tensor] = []
        safeguard_disabled = False
        for i in range(n_chunks):
            try:
                t = torch.empty(chunk_floats, dtype=torch.float32, device="cuda")
            except torch.cuda.OutOfMemoryError as e:
                free_b_after, _ = torch.cuda.mem_get_info()
                logger.error(
                    f"[oom_test] rank={rank} OOM at chunk {i + 1}/{n_chunks} "
                    f"despite safeguard: "
                    f"free_now={int(free_b_after // (1024 * 1024))}MiB; "
                    f"got={len(self._oom_tensors) * chunk_mb}MiB of "
                    f"{alloc_mb}MiB target; err={e}"
                )
                return
            self._oom_tensors.append(t)
            # Once the safeguard has fired, it self-disarms: the kill voids
            # the grant (reset_granted), so with nothing reserved and no
            # standby memory held the reclaim gate cannot fire again —
            # remaining allocations of this test surface OOM cleanly.
            logger.info(f"get_num_kill_standby_called()={get_num_kill_standby_called()}")
            if not safeguard_disabled and get_num_kill_standby_called() >= 1:
                safeguard_disabled = True
                logger.info(
                    f"[oom_test] rank={rank} safeguard fired during chunk "
                    f"{i + 1}/{n_chunks}; self-disarmed for the rest of "
                    f"the test"
                )

        free_b_after, _ = torch.cuda.mem_get_info()
        logger.info(
            f"[oom_test] rank={rank} step={self.step}: alloc OK "
            f"({len(self._oom_tensors)} tensors x {chunk_mb}MiB = "
            f"{len(self._oom_tensors) * chunk_mb}MiB); "
            f"free_after={int(free_b_after // (1024 * 1024))}MiB"
        )
        self._oom_tensors = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()

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

    def _resilient_opt_recover(self, resume_step: int):
        """Recover resilient optimizer state on RMP resume.

        ``resume_step`` is the globally agreed-on step (MAX of every rank's
        ``_step_counter``); train() computes it alongside the metadata-vote
        so we don't need a second all-reduce here.
        """
        self.rmp_manager.load_cpu_metadata(resume_step)
        self._resilient_opt.bind()

        logger.info(f"[ResilientOpt] resume_step={resume_step} (global max)")

        # Recompute clip_coef from the per-rank pre-reduction locals persisted
        # in RMP — runs the SAME reducer the normal path uses (and the SAME
        # multi-collective sequence: per-mesh full_tensor + optional PP/EP
        # all-reduce + clamp). All-ranks lockstep; placed before
        # ``maybe_recover`` so the chunked replay scales by the correct
        # coefficient. Grad-free + peer-free: the local came from this rank's
        # valid grads at compute time and was persisted before the original
        # collective, so collective completion ⟹ every rank's RMP holds its
        # fresh local for the resume step.
        self._resilient_opt.recompute_clip_coef_from_locals(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            norm_type=2.0,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            ep_enabled=self.parallel_dims.ep_enabled,
        )

        recovered = self._resilient_opt.maybe_recover(resume_step)
        if recovered:
            logger.info(f"[ResilientOpt] (rank={dist.get_rank()}) Recovery completed at step {resume_step}")
        else:
            logger.info(f"[ResilientOpt] (rank={dist.get_rank()}) No recovery needed at step {resume_step}")

        self._resilient_opt.zero_moe_tokens_per_expert()
        self.lr_schedulers.step()
        self.step = resume_step

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

    def _opt_fingerprint(self, tag: str):
        """Compact, env-gated fingerprint of optimizer state for debugging
        FATAL-recovery divergence. Enable with LETO_OPT_FP=1. Logs lr + the
        L2 norms of param / exp_avg / exp_avg_sq + the step tensor for the
        first two params that have optimizer state, so a normal run and a
        faulted run can be diffed at the recovery boundary to see which
        quantity (lr / moments / step) is wrong."""
        if os.environ.get("LETO_OPT_FP") != "1":
            return
        try:
            lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
        except Exception:
            lr = float("nan")
        def _loc(t):
            return t.to_local() if hasattr(t, "to_local") else t
        # Order-independent, rank-stable global checksums (double precision so
        # tiny per-element differences accumulate visibly): total squared L2 of
        # params / exp_avg / exp_avg_sq across ALL this-rank shards.
        psum = msum = vsum = 0.0
        sv0 = None
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    st = optimizer.state.get(p)
                    if not st or "exp_avg" not in st:
                        continue
                    psum += _loc(p).detach().double().pow(2).sum().item()
                    msum += _loc(st["exp_avg"]).detach().double().pow(2).sum().item()
                    vsum += _loc(st["exp_avg_sq"]).detach().double().pow(2).sum().item()
                    if sv0 is None:
                        s = st["step"]
                        sv0 = s.item() if torch.is_tensor(s) else s
        rank = int(os.environ.get("RANK", "0"))
        logger.info(
            f"[OPTFP] tag={tag} rank={rank} step={self.step} lr={lr:.12e} "
            f"pstep={sv0} P2={psum:.15e} M2={msum:.15e} V2={vsum:.15e}"
        )

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

        self.rmp_manager.maybe_commit(self.step)

        if self._expert_dist_tracker is not None:
            self._expert_dist_tracker.end_step(self.step)

        if self._resilient_opt is not None:
            # Defer the in-place grad scaling AND persist the pre-reduction
            # local norm so a mid-step fault is fully recoverable:
            #   1. compute_locals  — per-rank `_NormPartial` (no collective)
            #   2. persist_clip_locals — write locals to RMP **before** the
            #      reducer so collective completion witnesses universal
            #      persistence (any rank that advances ⟹ every rank
            #      persisted; recovery re-runs the same reducer over the
            #      persisted locals, no peer reads, no grad reads).
            #   3. reduce_from_locals — full_tensor/EP/PP all-reduce + clamp
            #      (the collective(s)); shared with the stock and recovery
            #      paths → bit-identical by construction.
            #   4. set_clip_coef — stage coef for the chunked replay.
            # grad_norm is still the pre-clip total norm, so the logged value
            # is unchanged.
            params_for_clip = [p for m in self.model_parts for p in m.parameters()]
            pp_mesh = parallel_dims.get_optional_mesh("pp")
            clip_locals = dist_utils.clip_compute_locals(
                params_for_clip,
                norm_type=2.0,
                foreach=True,
                ep_enabled=parallel_dims.ep_enabled,
            )
            self._resilient_opt.persist_clip_locals(clip_locals.locals)
            grad_norm, clip_coef = dist_utils.clip_reduce_from_locals(
                clip_locals.locals,
                self.job_config.training.max_norm,
                norm_type=2.0,
                pp_mesh=pp_mesh,
                ep_enabled=parallel_dims.ep_enabled,
            )
            self._resilient_opt.set_clip_coef(clip_coef)
        else:
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
        

        self._opt_fingerprint("pre_step")
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

    def _maybe_install_oom_safeguard(self):
        if self._oom_safeguard_installed:
            return
        self._oom_safeguard_installed = True
        # OOM safeguard: install only now that we've reached training. By
        # construction this rank is active (original or post-promotion) —
        # pre-activation standbys never get here because they're parked
        # in run_init_sequence polling for activation. This guarantees
        # KillStandby is only ever issued by an active rank.
        from torchtitan.components.init.progressive import get_free_mb

        leto_cfg = self.job_config.leto
        _oom_rank = int(os.environ.get("RANK", -1))

        if leto_cfg.progressive_init and leto_cfg.enable_standby:
            # Reservation broker: the active process is the broker for the
            # standby's GPU memory. progressive_reservation_margin_mb is the
            # master switch: > 0 installs the broker (grant + reclaim, i.e.
            # kill the standby before the active grows into reserved memory);
            # == 0 skips the broker entirely, so the standby runs its init
            # tasks ungated (grant-only control that OOMs under pressure) and
            # the active never reclaims.
            margin_mb = int(leto_cfg.progressive_reservation_margin_mb)
            if margin_mb <= 0:
                logger.info(
                    f"OOM safeguard disabled: reservation margin=0 on "
                    f"rank={_oom_rank} — standby runs reordered init ungated, "
                    f"no reclaim (grant-only OOM control)"
                )
                return

            from torchtitan.components.mem import (
                install_reservation_broker,
                reset_granted,
            )
            from torchtitan.components.init.progressive import _shm_path

            ledger_path = _shm_path(_oom_rank)

            def _on_oom() -> tuple[bool, int]:
                freed, pid = kill_standby_for_oom_safeguard(
                    get_free_mb(), 0, rank=_oom_rank
                )
                if freed:
                    reset_granted()
                return (freed, pid)

            _grant_only = str(leto_cfg.progressive_protocol) == "grant_only"
            logger.info(
                f"Installing reservation broker: rank={_oom_rank} "
                f"margin={margin_mb}MiB reclaim=on "
                f"protocol={leto_cfg.progressive_protocol} "
                f"ledger={ledger_path}"
            )
            install_reservation_broker(
                ledger_path, margin_mb, _on_oom, grant_only=_grant_only,
                est_ttl_ms=int(leto_cfg.fmcb_est_ttl_ms),
                timing=bool(getattr(leto_cfg, "fmcb_timing", False)),
            )

            # RMP-server allocations (e.g. lazy gradient-persistence tensors
            # at the first backward) happen in a different process, so a CUDA
            # OOM there never reaches the FreeMemoryCallback above. Route
            # those failures through the same standby reclaim + one retry.
            try:
                from leto.rmp.client import set_oom_reclaim_hook

                set_oom_reclaim_hook(lambda: _on_oom()[0])
            except ImportError:
                pass


    def _resume_vote_pg(self):
        """CPU (gloo) group for the rmp resume vote. Collective-ordering-safe:
        every rank on the rmp path calls this at the same point in train()."""
        if getattr(self, "_vote_pg", None) is None:
            self._vote_pg = dist.new_group(backend="gloo")
        return self._vote_pg

    def _moevement_replay_pending(self) -> bool:
        """True when this rank's predecessor died MID-REPLAY (duck-typed;
        only method=moevement defines the hook).

        A moevement restore arms a replay spanning w_sparse-1 real training
        steps; until save(S+w) the live — and therefore RMP-persisted —
        params/optimizer state are a sparse-replay INTERMEDIATE: operators
        that have not been activated yet hold their bf16 compute copy and
        unrestored Adam moments. That state is bit-exact for the forward
        (which casts to bf16 anyway) but wrong for the optimizer, so a
        promoted standby that adopts it produces bit-identical losses for
        the interrupted step and diverges from the NEXT one on (2026-07
        gptoss_tp2fsdp4ep4 M8 full_ftft: transient at replayed step 23 →
        step 24 loss 10.6507 vs 10.6925 baseline).

        Voting it into ``has_md`` routes the whole world down the
        checkpointer.load() fallback, which restores the window and replays
        it from the top when the dump survived, and otherwise fresh-starts
        via _fresh_start_reinit_from_seed — never a half-restored resume.
        """
        pending = getattr(self.checkpointer, "has_pending_replay", None)
        if pending is None or not pending():
            return False
        logger.warning(
            "[moevement] this rank's predecessor died mid-replay — voting "
            "the RMP resume down; recovery falls back to checkpointer.load()"
        )
        return True

    def _fresh_start_reinit_from_seed(self):
        """True seed fresh start for a trainer attached to pre-existing RMP
        GPU pools (rmp_restored=True) whose fallback load found no
        checkpoint.

        maybe_init() skips init_weights() when the pools were retrieved
        rather than allocated, on the assumption that a recovery source will
        overwrite them. When BOTH recovery sources come up empty — no
        committed RMP metadata (the vote's any_lacks path) AND
        checkpointer.load() returned False (e.g. moevement's
        window-agreement vote said FRESH-START) — nothing ever overwrites
        the pools, and the run would silently continue on whatever the dead
        predecessor left in them (2026-07 gptoss_tp2fsdp4ep4 full_ftft
        forensics: a standby promoted on a transient fault that hit before
        the post-fatal active's first metadata commit trained the active's
        half-restored step-22 weights with train_state reset to step 1).
        Plan §3.9: a fresh start must never be RMP-sourced — so make it a
        real one:

        - reseed RNG exactly as the initial init sequence did, then re-run
          init_weights() into the RMP-backed params/buffers (bit-identical
          to a from-scratch start's weights);
        - zero all optimizer state tensors (RMP pools are zero-filled on
          allocation, so this reproduces fresh-alloc semantics);
        - drop restored param gradients (restore_param_gradients may have
          re-attached the predecessor's stale grads).

        Every rank takes this path together: rmp_restored and the two vote
        outcomes are world-uniform, so no collective divergence.
        """
        logger.warning(
            "[fresh-start] RMP pools were retrieved but neither RMP metadata "
            "nor the checkpointer had recoverable state — re-initializing "
            "model/optimizer state from seed (plan §3.9: fresh start must "
            "never be RMP-sourced)"
        )
        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            self.job_config.debug,
            distinct_seed_mesh_dims=["pp"],
        )
        with torch.no_grad():
            for model_part in self.model_parts:
                model_part.init_weights(buffer_device=self.buffer_device)
                model_part.train()
            for optimizer in self.optimizers:
                for param_state in optimizer.state.values():
                    for value in param_state.values():
                        if torch.is_tensor(value):
                            value.zero_()
        for model_part in self.model_parts:
            for param in model_part.parameters():
                param.grad = None

    @record
    def train(self):
        job_config = self.job_config
        if job_config.leto.enable_rmp_gpu and not job_config.leto.disable_resilient_opt:
            # One all_reduce instead of two: pack the metadata-vote and
            # resume_step into a single MAX-reduced tensor.
            #   buf[0] = 1 if this rank lacks metadata else 0. After MAX,
            #     buf[0]==1 means at least one rank lacks metadata, so every
            #     rank must fall back to gemini to avoid divergence.
            #   buf[1] = _step_counter when this rank has metadata, else 0.
            #     After MAX, equals the global-max step counter (= resume
            #     step) when every rank has metadata; ignored otherwise.
            #   buf[2] = 1 if this rank re-attached to pre-existing RMP
            #     pools (rmp_restored). After MAX, world-uniform "any rank
            #     holds a predecessor's pool state" — gates the stale-pool
            #     fresh-start reinit below. Must be voted, not read locally:
            #     a lone organically-relaunched RMP server would otherwise
            #     split the decision (and set_determinism can broadcast).


            has_md = (
                self.rmp_manager.has_committed_metadata()
                and not self._moevement_replay_pending()
            )
            local_step = self._resilient_opt.get_step() if has_md else 0
            buf = torch.tensor(
                [0 if has_md else 1, local_step, 1 if self.rmp_restored else 0],
                dtype=torch.int64,
            )
            # Vote over gloo: an all_reduce on the default PG would create an
            # extra world-size NCCL communicator (the no-fault path never
            # creates one), which measurably degrades forward-collective
            # arrival timing for the rest of the run (~1.3% steady state).
            dist.all_reduce(buf, op=dist.ReduceOp.MAX, group=self._resume_vote_pg())
            any_lacks = bool(buf[0].item())
            resume_step = int(buf[1].item())
            any_retrieved = bool(buf[2].item())

            if not any_lacks:
                self._resilient_opt_recover(resume_step)
                # RMP recovery bypassed checkpointer.load() entirely — on
                # the full-leto transient path the in-memory checkpoint
                # state is NOT consulted (plan §3.9). method=moevement must
                # still reseed its deferred window-start ring from the
                # restored state and start a fresh sparse window (plan
                # §8-R6); duck-typed — gemini defines no such hook.
                notify = getattr(self.checkpointer, "notify_rmp_restored", None)
                if notify is not None:
                    notify(resume_step)
            else:
                loaded = self.checkpointer.load(step=job_config.checkpoint.load_step)
                if not loaded and any_retrieved:
                    # Retrieved pools + no recovery source anywhere: the
                    # pools hold a dead predecessor's partial state. Reinit
                    # BEFORE bind() so the resilient optimizer captures the
                    # seed-fresh tensors.
                    self._fresh_start_reinit_from_seed()
                self._resilient_opt.bind()
                # Pass the trainer's restored step explicitly: under
                # method=moevement's sparse restore only the window's first
                # slot ops carry restored per-param Adam steps (still-frozen
                # params hold 0 until their replay activation), so the
                # param-derived fallback inside resync would be wrong. Under
                # gemini/fresh-start self.step equals that fallback.
                self._resilient_opt.resync_after_external_load(self.step)
        elif job_config.leto.enable_rmp_gpu and job_config.leto.disable_resilient_opt:
            # No ResilientOptimizer replay, but rmp_manager.maybe_commit has
            # been writing CPU-side metadata (step counter, dataloader, lr
            # scheduler) every step. Load that metadata so the restart
            # picks up where the last successful step left off; the
            # RMP-GPU-backed params/optim tensors are already correct.
            has_md = (
                self.rmp_manager.has_committed_metadata()
                and not self._moevement_replay_pending()
            )
            local_step = self.rmp_manager.latest_committed_step() or 0
            # Same 3-slot packing as the resilient-opt branch above; buf[2]
            # makes the stale-pool fresh-start decision world-uniform.
            buf = torch.tensor(
                [0 if has_md else 1, local_step, 1 if self.rmp_restored else 0],
                dtype=torch.int64,
            )
            dist.all_reduce(buf, op=dist.ReduceOp.MAX, group=self._resume_vote_pg())
            any_lacks = bool(buf[0].item())
            resume_step = int(buf[1].item())
            any_retrieved = bool(buf[2].item())

            if not any_lacks:
                self.rmp_manager.load_cpu_metadata(resume_step)
                # The committed metadata captures lr_scheduler state from the
                # START of step=resume_step (before optimizer.step). To match
                # the no-fault path's state at the end of resume_step (so the
                # next iter starts with the same lr as normal step resume_step+1
                # would), advance the scheduler by one — same compensation
                # _resilient_opt_recover does.
                self.lr_schedulers.step()
                self.step = resume_step
                # Same duck-typed reseed as the resilient-opt branch above.
                notify = getattr(self.checkpointer, "notify_rmp_restored", None)
                if notify is not None:
                    notify(resume_step)
            else:
                loaded = self.checkpointer.load(step=job_config.checkpoint.load_step)
                if not loaded and any_retrieved:
                    # Same stale-pool hazard as the resilient-opt branch.
                    self._fresh_start_reinit_from_seed()
        else:
            self.checkpointer.load(step=job_config.checkpoint.load_step)

        self._restored_step = self.step
        self._opt_fingerprint("post_recovery")

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

                self.maybe_alloc_for_oom_test()
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

                self._maybe_install_oom_safeguard()

                # Miss-rate observability: the stop_broker atexit line never
                # survives the crash-style exits real runs use, so surface
                # the counters in-band (one line per rank every 32 steps).
                if self._oom_safeguard_installed and self.step % 32 == 0:
                    try:
                        from torchtitan.components.mem import _module as _mem_mod
                        if _mem_mod is not None:
                            misses, fast = _mem_mod.get_fmcb_miss_stats()
                            logger.info(
                                f"fmcb stats step={self.step}: misses={misses} "
                                f"fast_exits={fast}"
                            )
                    except Exception:
                        pass

                # Per-component callback timing (leto.fmcb_timing): dump the
                # cumulative counters every step so the offline analyzer can
                # diff consecutive steps and bucket by step%8. Cheap (a few
                # atomic loads); only active when fmcb_timing is set.
                if (self._oom_safeguard_installed
                        and getattr(self.job_config.leto, "fmcb_timing", False)):
                    try:
                        import json as _json
                        from torchtitan.components.mem import _module as _mem_mod
                        if _mem_mod is not None:
                            _d = _mem_mod.get_fmcb_timing()
                            try:
                                _r = torch.distributed.get_rank()
                            except Exception:
                                _r = -1
                            logger.info(
                                f"fmcb_timing rank={_r} step={self.step} "
                                f"{_json.dumps(_d, sort_keys=True)}"
                            )
                    except Exception:
                        pass


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
        # The DTensor RNG tracker is a thin wrapper over
        # torch.cuda.default_generators, so cuda_rng_state alone covers it
        # — no separate dtensor_rng_state.
        return {
            "step": self.step,
            "ntokens_seen": self.ntokens_seen,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state": torch.cuda.get_rng_state(self.device),
        }

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

    rank = int(os.environ.get("RANK", -1))
    logger.info(f"Process rank={rank}, pid={os.getpid()}")
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
        elif config.leto.profile_init:
            # A profile_init run exists only to capture the init-cost profile;
            # the active group must not run the training loop, whose
            # activation/optimizer allocations would OOM on top of the standby
            # group's still-resident init memory. When a standby is present it is
            # the one writing the profile (init_profile/<mode>/...), so the active
            # group must stay alive until those files land — exiting first trips
            # the master's shutdown, which SIGKILLs the standby mid-profile. Once
            # the profile is on disk the wait returns and we fall through to the
            # clean close()/destroy_process_group() in the `else` clause below.
            logger.info("leto.profile_init enabled - skipping training execution")
            if config.leto.enable_standby:
                from torchtitan.init import wait_for_standby_init_profile

                wait_for_standby_init_profile(config)
        else:
            trainer.train()
    except Exception:
        # Crash-style exit: log the failure and _exit WITHOUT graceful
        # teardown. Attempting teardown here is what wedged post-OOM ranks
        # (2026-07 hang forensics): trainer.close() blocks on in-flight
        # gemini snapshot exchanges with dead peers (up to the ~30 min gloo
        # PG timeout), and normal interpreter exit joins the non-daemon
        # SnapshotContainer child, which polls CheckServiceAction forever
        # because the controller answers WORKING until teardown. A rank
        # wedged here never exits, torchrun never reports failure, and the
        # launcher sees a healthy job indefinitely. os._exit makes a crash
        # look like a crash: the SnapshotContainer is left alive as an
        # orphan, exactly as after a SIGKILL, so the controller's
        # PERSIST/CLOSE handshake still owns its lifecycle (gemini fatal
        # recovery depends on that).
        import sys
        import traceback

        logger.error(
            "Training failed; exiting crash-style without graceful teardown:\n"
            + traceback.format_exc()
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    if _LETO_AVAILABLE:
            rank = int(os.environ["RANK"])
            register_training_process(rank=rank)
            report_event(EVENT_PROCESS_STARTED)
    main(Trainer)
