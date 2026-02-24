# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import importlib
import json
import os
import random
import time
from datetime import timedelta
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed.checkpoint.stateful
import torch.distributed.tensor._random as dtensor_random
from torch.distributed.elastic.multiprocessing.errors import record

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.eager_init import maybe_eager_init
from torchtitan.components.ft import FTManager, maybe_semi_sync_training
from torchtitan.components.gemini.checkpoint import GeminiCheckpointManager
from torchtitan.components.loss import rescale_accumulated_loss
from torchtitan.components.metrics import (
    build_metrics_processor,
    ensure_pp_loss_visible,
)
from torchtitan.components.rmp_manager import RmpManager
from torchtitan.components.skip_shape_infer import (
    maybe_record_stage_inputs,
    maybe_warmup_stages
)
from torchtitan.config import ConfigManager, JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)

# Optional leto integration for measurements
try:
    from leto.launch.worker_controller_client import (
        get_client as get_leto_client,
        report_event,
        report_duration,
        register_training_process,
        get_checkpoint_loading_type,
        EVENT_TRAINING_STARTED,
        EVENT_CHECKPOINT_LOADING_DONE,
        EVENT_STEP_DONE,
        DURATION_CHECKPOINT_LOADING,
        DURATION_ITERATION,
        DURATION_WEIGHT_ALLOCATION,
    )
    _LETO_AVAILABLE = True
except ImportError:
    _LETO_AVAILABLE = False


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

        device_module, device_type = utils.device_module, utils.device_type
        # pyrefly: ignore [read-only]
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes
        self.parallel_dims = parallel_dims = self.init_distributed()
        logger.info(f"Init distributed.")

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            batch_degree, batch_rank = 1, 0

        # pyrefly: ignore [bad-argument-type]
        self.ft_manager = FTManager(job_config.fault_tolerance)
        batch_degree, batch_rank = self.ft_manager.get_dp_info(batch_degree, batch_rank)

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=job_config.training.gc_freq, debug=job_config.training.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            parallel_dims,
            self.device,
            job_config.debug,
            distinct_seed_mesh_dims=["pp"],
        )
        self.train_spec = train_spec_module.get_train_spec(job_config.model.name)

        # build tokenizer and dataloader
        self.tokenizer = (
            self.train_spec.build_tokenizer_fn(job_config)
            if self.train_spec.build_tokenizer_fn is not None
            else None
        )

        self.dataloader = self.train_spec.build_dataloader_fn(
            dp_world_size=batch_degree,
            dp_rank=batch_rank,
            tokenizer=self.tokenizer,
            job_config=job_config,
        )

        # build model (using meta init)
        model_args = self.train_spec.model_args[job_config.model.flavor]
        # set the model args from training job configs
        model_args.update_from_config(job_config)
        self.model_args = model_args

        logger.info(
            f"Building {job_config.model.name} {job_config.model.flavor}"
            f"with {json.dumps(dataclasses.asdict(model_args), indent=2, ensure_ascii=False)}"
        )
        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
        ):
            # pyrefly: ignore[bad-instantiation]
            model = self.train_spec.model_cls(model_args)

        # Build the collection of model converters. No-op if `model.converters` empty
        model_converters = build_model_converters(job_config, parallel_dims)
        model_converters.convert(model)

        # metrics logging
        build_metrics_processor_fn = (
            build_metrics_processor
            if self.train_spec.build_metrics_processor_fn is None
            else self.train_spec.build_metrics_processor_fn
        )
        self.metrics_processor = build_metrics_processor_fn(
            job_config, parallel_dims, model_args
        )
        color = self.metrics_processor.color

        # calculate model size and flops per token
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_args.get_nparams_and_flops(model, job_config.training.seq_len)

        logger.info(
            f"{color.blue}Model {job_config.model.name} {job_config.model.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        if job_config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            self.buffer_device = None
        elif job_config.training.enable_cpu_offload:
            init_device = "cpu"
            self.buffer_device = device_type
        else:
            init_device = device_type
            self.buffer_device = None

        self.loss_fn = self.train_spec.build_loss_fn(
            job_config, parallel_dims=parallel_dims, ft_manager=self.ft_manager
        )

        # verify batch sizes
        global_batch_size = job_config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            global_batch_size = job_config.training.local_batch_size * batch_degree
        assert global_batch_size > 0
        assert (
            global_batch_size % (job_config.training.local_batch_size * batch_degree)
            == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({global_batch_size} "
            f"% ({job_config.training.local_batch_size} * {batch_degree}) != 0)"
        )

        # calculate gradient accumulation steps
        self.gradient_accumulation_steps = global_batch_size // (
            job_config.training.local_batch_size * batch_degree
        )
        assert self.gradient_accumulation_steps > 0
        self.loss_fn = rescale_accumulated_loss(
            self.loss_fn, self.gradient_accumulation_steps
        )

        # apply parallelisms and initialization
        # [Leto] Start timing weight allocation and init
        _weight_alloc_start = time.time()

        if parallel_dims.pp_enabled:
            if not self.train_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {job_config.model.name} "
                    f"does not support pipelining"
                )

            # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = self.train_spec.pipelining_fn(
                model,
                parallel_dims,
                job_config,
                self.device,
                model_args,
                self.train_spec.parallelize_fn,
                self.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model

            if not job_config.leto.enable_rmp:
                for m in self.model_parts:
                    m.to_empty(device=init_device)
                    with torch.no_grad():
                        # pyrefly: ignore [not-callable]
                        m.init_weights(buffer_device=buffer_device)
                    m.train()

            # confirm that user will be able to view loss metrics on the console
            # pyrefly: ignore [bad-argument-type]
            ensure_pp_loss_visible(parallel_dims, job_config, color)
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            model = self.train_spec.parallelize_fn(model, parallel_dims, job_config)

            if not job_config.leto.enable_rmp:
                model.to_empty(device=init_device)
                with torch.no_grad():
                    # pyrefly: ignore [not-callable]
                    model.init_weights(buffer_device=buffer_device)
                model.train()

            self.model_parts = [model]

        # [Leto] Record weight allocation and init time
        if _LETO_AVAILABLE:
            report_duration(DURATION_WEIGHT_ALLOCATION, time.time() - _weight_alloc_start)

        self.ft_manager.maybe_set_all_reduce_hook(self.model_parts)

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        self.optimizers = self.train_spec.build_optimizers_fn(
            self.model_parts, job_config.optimizer, parallel_dims, self.ft_manager
        )
        self.lr_schedulers = self.train_spec.build_lr_schedulers_fn(
            self.optimizers, job_config.lr_scheduler, job_config.training.steps
        )
        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
        # where it issues a single all-reduce for all parameters at once for better performance
        self.optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(
                self.model_parts
            )
        )
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.ntokens_seen = 0

        ckpt_cls = CheckpointManager
        if job_config.checkpoint.use_gemini:
            ckpt_cls = GeminiCheckpointManager
            assert parallel_dims.fsdp_enabled, "Gemini needs FSDP enabled."

        ckpt_extra_kwargs = {}
        if job_config.checkpoint.use_gemini:
            ckpt_extra_kwargs["parallel_dims"] = parallel_dims

        self.checkpointer = ckpt_cls(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            checkpoint_config=job_config.checkpoint,
            sd_adapter=(
                # pyrefly: ignore[bad-instantiation]
                self.train_spec.state_dict_adapter(
                    model_args, job_config.model.hf_assets_path
                )
                if self.train_spec.state_dict_adapter
                else None
            ),
            base_folder=job_config.job.dump_folder,
            ft_manager=self.ft_manager,
            **ckpt_extra_kwargs,
        )

        self.rmp_manager = RmpManager(
            leto_config=job_config.leto,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            states={"train_state": self},
            lr_schedulers=self.lr_schedulers,
            dataloader=self.dataloader,
            device=self.device,
        )

        loss_parallel_enabled = (
            parallel_dims.tp_enabled
            and not job_config.parallelism.disable_loss_parallel
        )
        self.train_context = dist_utils.get_train_context(loss_parallel_enabled)
        self.maybe_enable_amp = dist_utils.maybe_enable_amp(
            parallel_dims,
            job_config.training.mixed_precision_param,
            device_type,
        )

        # Build validator if validation is configured
        if job_config.validation.enable:
            assert self.train_spec.build_validator_fn is not None

            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = self.train_spec.build_validator_fn(
                job_config=job_config,
                dp_world_size=batch_degree,
                dp_rank=batch_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=self.loss_fn,
                validation_context=self.train_context,
                maybe_enable_amp=self.maybe_enable_amp,
                metrics_processor=self.metrics_processor,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        logger.info(
            "Trainer is initialized with "
            f"local batch size {job_config.training.local_batch_size}, "
            f"global batch size {global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {job_config.training.seq_len}, "
            f"total steps {job_config.training.steps} "
            f"(warmup {job_config.lr_scheduler.warmup_steps})"
        )

    def init_distributed(self) -> ParallelDims:
        job_config = self.job_config
        world_size = dist_utils.init_distributed(
            job_config.comm,
            enable_cpu_backend=job_config.training.enable_cpu_offload,
            base_folder=job_config.job.dump_folder,
        )

        parallelism_config = job_config.parallelism
        return ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            etp=parallelism_config.expert_tensor_parallel_degree,
            world_size=world_size,
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

    def train_step(
        self, data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        # Save the current step learning rate for logging
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

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

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
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

        self.rmp_manager.maybe_init(self.buffer_device)
        maybe_eager_init(job_config.leto.eager_init_list, self.parallel_dims, self.device)

        # [Leto] Report training started event
        if _LETO_AVAILABLE:
            report_event(EVENT_TRAINING_STARTED)

        # [Leto] Time checkpoint loading
        _ckpt_load_start = time.time()
        self.checkpointer.load(step=job_config.checkpoint.load_step)
        if _LETO_AVAILABLE:
            _ckpt_loading_type = get_checkpoint_loading_type()
            report_duration(DURATION_CHECKPOINT_LOADING, time.time() - _ckpt_load_start,
                            checkpoint_loading_type=_ckpt_loading_type)
            report_event(EVENT_CHECKPOINT_LOADING_DONE)

        maybe_warmup_stages(
            self.model_parts,
            job_config
        )

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
            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                # [Leto] Time each iteration
                _iter_start = time.time()

                self.step += 1

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
                    self.train_step(data_iterator)
                except DataloaderExhaustedError:
                    logger.warning("Ran out of data; last step was canceled.")
                    break

                self.checkpointer.save(
                    self.step, last_step=(self.step == job_config.training.steps)
                )

                self.rmp_manager.maybe_commit()

                # Run validation if validator is available
                if (
                    self.job_config.validation.enable
                    and self.validator.should_validate(self.step)
                ):
                    # pyrefly: ignore [missing-attribute]
                    with self.loss_fn.no_rescale():
                        # pyrefly: ignore [bad-argument-count]
                        self.validator.validate(self.model_parts, self.step)

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

                # [Leto] Record iteration time and report step done
                _iter_time = time.time() - _iter_start
                if _LETO_AVAILABLE:
                    report_duration(DURATION_ITERATION, _iter_time, step=self.step)
                    report_event(EVENT_STEP_DONE, step=self.step)


        # [Leto] Wait for any pending checkpoint tracking to complete
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
            "cuda_rng_state": torch.cuda.get_rng_state(self.device),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        # Save DTensor RNG tracker state if available
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

        # Register training process with leto worker controller for fault injection and metrics
        if _LETO_AVAILABLE:
            register_training_process(rank=torch.distributed.get_rank())

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
    main(Trainer)
