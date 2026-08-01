# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch

from torchtitan.tools.logging import logger


@dataclass
class Job:
    config_file: str | None = None
    """File to read job configs from"""

    dump_folder: str = "./outputs"
    """Folder to dump job outputs"""

    description: str = "default job"
    """Description of the job"""

    print_config: bool = False
    """Print the job configs to terminal"""

    save_config_file: str | None = None
    """Path to save job config into"""

    custom_config_module: str = ""
    """
    This option allows users to extend the existing JobConfig with a customized
    JobConfig dataclass. Users need to ensure that the path can be imported.
    """


@dataclass
class Profiling:
    enable_profiling: bool = False
    """Whether to enable pytorch profile"""

    save_traces_folder: str = "profile_traces"
    """Trace files location"""

    profile_freq: int = 10
    """How often to collect profile traces, in iterations"""

    profiler_active: int = 1
    """
    The steps profiler is active for.

    This is used to configure torch.profile.schedule.
    """

    profiler_warmup: int = 3
    """
    The number of warmup steps before the active step in each profiling cycle.

    This is used to configure torch.profile.schedule.
    """

    enable_memory_snapshot: bool = False
    """Whether to dump memory snapshot"""

    save_memory_snapshot_folder: str = "memory_snapshot"
    """Memory snapshot files location"""



@dataclass
class Metrics:
    log_freq: int = 10
    """How often to log metrics to TensorBoard, in iterations"""

    enable_tensorboard: bool = False
    """Whether to log metrics to TensorBoard"""

    disable_color_printing: bool = False
    """Whether to disable color printing in logs"""

    save_tb_folder: str = "tb"
    """Folder to dump TensorBoard states"""

    save_for_all_ranks: bool = False
    """
    Whether to save TensorBoard/Wandb metrics only for rank 0 or for all ranks.
    When this option is False and pipeline_parallel_degree is > 1, the metrics
    component uses the 0th rank of the last stage pipeline group, which is the
    only stage that computes loss metrics.
    """

    enable_wandb: bool = False
    """Whether to log metrics to Weights & Biases"""

    wandb_project: str = ""
    """WandB project name. Overrides WANDB_PROJECT env var if set."""

    wandb_run_name: str = ""
    """WandB run name. Overrides WANDB_RUN_NAME env var if set."""

    save_expert_dist: bool = False
    """Save per-step expert token distribution (first MoE layer) to CSV per rank."""


@dataclass
class Model:
    name: str = "llama3"
    """Which model to train"""

    flavor: str = "debugmodel"
    """Which model config to train"""

    hf_assets_path: str = "./tests/assets/tokenizer"
    """
    Path to HF assets folder. This folder contains local copies of Hugging Face assets,
    including model weights in .safetensors format, the model.safetensor.index.json file
    (fqn to file mapping), the config.json file, generation_config.json, and tokenizer files.
    """

    tokenizer_path: str | None = None
    """DEPRECATED: Use hf_assets_path instead."""
    """Tokenizer path"""

    converters: list[str] = field(default_factory=list)
    """
    Comma separated list of converters to apply to the model.
    For instance, the `float8` converter swaps `torch.nn.Linear`
    with `Float8Linear`. This feature requires you to install 'torchao'
    which can be found here: https://github.com/pytorch/ao
    """

    print_after_conversion: bool = False
    """
    If true, model definition will be printed to stdout after all model
    converters have been applied.
    """


@dataclass
class Optimizer:
    name: str = "AdamW"
    """Optimizer to use"""

    lr: float = 8e-4
    """Learning rate to use"""

    beta1: float = 0.9
    beta2: float = 0.95
    """Exponential moving average hyperparameters to use"""

    eps: float = 1e-8
    """Epsilon value to use"""

    weight_decay: float = 0.1
    """Weight decay to use"""

    implementation: Literal["for-loop", "foreach", "fused"] = "fused"
    """
    Specify which optimizer implementation to use:
    - 'fused': Use fused implementation (CUDA only) for best performance.
    - 'foreach': Use some horizontal fusion of tensors for better performance.
    - 'for-loop': Use the default implementation for the optimizer (slowest).
    - more info: https://pytorch.org/docs/stable/optim.html
    """

    early_step_in_backward: bool = False
    """
    Whether to apply optimizer in the backward. Caution, optimizer_in_backward
    is not compatible with gradients clipping, users should not call
    register_post_accumulate_grad_hook after the optimizer is built.
    """


@dataclass
class LRScheduler:
    warmup_steps: int = 200
    """
    Steps for lr scheduler warmup, normally 1/5 of --training.steps
    """

    decay_ratio: float | None = None
    """
    Controls the proportion of the training steps allocated to the learning rate decay phase.
    If `None`, the learning rate will begin decaying immediately after the warmup period.
    Otherwise, the learning rate will remain stable after the warmup period and
    only start decaying during the last `decay_ratio` portion of the total training steps.
    This is known as the Warmup-Stable-Decay (WSD) schedule, as described in https://arxiv.org/abs/2404.06395.
    """

    decay_type: Literal["linear", "sqrt", "cosine"] = "linear"
    """
    Learning rate decay type to use during training:
    - 'linear': linearly decays learning rate from initial to final value
    - 'sqrt': decays learning rate following a 1 minus square root curve
    - 'cosine': smoothly decays learning rate following a cosine curve
    """

    min_lr_factor: float = 0.0
    """
    Min lr ratio for lr scheduler.
    If provided, the range of decay factor is scaled from 1 to `min_lr_factor`
    to ensure the learning rate does not drop below `optimizer.lr * lr_scheduler.min_lr_factor`.
    """


@dataclass
class DataLoader:
    """
    Configuration for PyTorch DataLoader settings.

    These settings are passed directly to StatefulDataLoader.

    Note:
        persistent_workers and prefetch_factor are only valid if num_workers > 0.

    Example (TOML config file):
        [training.dataloader]
        num_workers = 4
        pin_memory = true
        persistent_workers = true
        prefetch_factor = 2
    """

    num_workers: int = 0
    """Number of worker processes for data loading."""

    persistent_workers: bool = False
    """Keep workers alive between epochs. Only valid when num_workers > 0."""

    pin_memory: bool = False
    """Copy tensors to CUDA pinned memory before returning them."""

    prefetch_factor: int | None = None
    """
    Number of batches loaded in advance by each worker. Only valid when num_workers > 0.
    Default is 2 when num_workers > 0, otherwise None.
    """


@dataclass
class Training:
    dataset: str = "c4_test"
    """Dataset to use"""

    dataset_path: str | None = None
    """
    Path to the dataset in the file system. If provided, data will be
    loaded from this path instead of downloaded.
    """

    local_batch_size: int = 8
    """Local batch size (i.e., per-device batch size)"""

    global_batch_size: int = -1
    """
    Global batch size (defaults to `training.local_batch_size * data-parallel degree`)
    """

    seq_len: int = 2048
    """Sequence length"""

    max_norm: float | int = 1.0
    """Max norm for gradient clipping"""

    steps: int = 10000
    """How many train steps to run"""

    max_steps: int = -1
    """Max steps for LR scheduler decay. If -1, defaults to training.steps."""

    enable_cpu_offload: bool = False
    """
    Whether to apply CPU offloading of parameters, gradients, and optimizer states in FSDP
    """

    dtype: Literal["bfloat16", "float32"] = "float32"
    """
    torch dtype for training. In contrast to mixed precision training, setting training_dtype=bfloat16 will
    put all parameters, gradients, and optimizer states in bfloat16, without an extra copy of fp32 weights.
    In the case of full bf16 training, RoPE calculations and logits will still be in fp32.
    """

    mixed_precision_param: Literal["bfloat16", "float32"] = "bfloat16"
    """
    torch dtype to use for parameters when applying mixed precision via fully_shard or torch.autocast.
    This feature takes effect via fully_shard when data_parallel_shard_degree > 1 or
    context_parallel_degree > 1; it takes effect via torch.autocast when data_replicate_degree >= 1
    and no other parallelism is enabled, i.e. under DDP or single-device training.
    """

    mixed_precision_reduce: Literal["float32"] = "float32"
    """
    torch dtype to use for reductions when applying mixed precision via FSDP.
    This feature only takes effect when data_parallel_shard_degree > 1
    """

    gc_freq: int = 50
    """Python garbage control scheduling interval, in steps"""

    gc_debug: bool = False
    """
    Enable GC debugging mode. This will perform gc.collect() at every step to
    detect if there is a reference cycle that includes a CUDA Tensor.
    Note that you may want to lower the training steps to avoid generating too
    many temporary files.
    """

    dataloader: DataLoader = field(default_factory=DataLoader)
    """DataLoader configuration"""


@dataclass
class Parallelism:
    data_parallel_replicate_degree: int = 1
    """
    The `data_parallel_replicate_degree` argument specifies the degree of
    data parallelism for weight replication. When this value is greater
    than 1, weights will be replicated across `data_parallel_replicate_degree`
    ranks. If `data_parallel_shard_degree` is also greater than 1, the parallelism
    method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
    parallelism method used is DDP (Distributed Data Parallelism).
    1 means disabled.
    """

    data_parallel_shard_degree: int = -1
    """
    The `data_parallel_shard_degree` argument specifies the degree of data
    parallelism for weight sharding. When this value is greater than 1, weights
    will be sharded across `data_parallel_shard_degree` ranks. If
    `data_parallel_replicate_degree` is also greater than 1, the parallelism
    method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
    parallelism method used is FSDP (Fully Sharded Data Parallelism).
    -1 means leftover ranks will be used (After DP_REPLICATE/SP/PP). Note that
    only `data_parallel_shard_degree` can be negative. 1 means disabled.
    """

    fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default"
    """
    `reshard_after_forward` specifies the policy for applying `reshard_after_forward`
    within an FSDP setup. `reshard_after_forward` controls parameter behavior after forward,
    trading off memory and communication. See torch's `fully_shard` API for more documentation
    on `reshard_after_forward`.

    The supported policies include "default", "always" and "never":

    - "default" applies default resharding behavior, implementing "smart defaults" for known optimal
      scenarios.
    - "always" will enable `reshard_after_forward` for all forward passes.
    - "never" will disable `reshard_after_forward` for all forward passes.
    """

    tensor_parallel_degree: int = 1
    """Tensor Parallelism degree. 1 means disabled."""

    disable_loss_parallel: bool = False
    """Whether to apply loss parallel when sequence parallel is enabled"""

    enable_async_tensor_parallel: bool = False
    """Whether to apply async tensor parallel (currently only effective when compile is enabled)"""

    pipeline_parallel_degree: int = 1
    """
    Pipeline Parallelism degree, or number of ranks. 1 means disabled.
    If using looped schedules, this still specifies the number of physical ranks, not the number
    of stages. Stages per rank are inferred from split points degree, and schedule.
    """

    module_fqns_per_model_part: list[list[str]] | None = None
    """
    Specify a list of lists containing the FQNs (Fully Qualified Names) of modules for each model chunk.
    Each inner list represents one model chunk and contains the module names that belong to that chunk.
    e.g. [['tok_embeddings', 'layers.0'], ['layers.1', 'layers.2'], ['layers.3', 'layers.4']]
    will create 3 chunks: the first containing tok_embeddings and layers.0,
    the second containing layers.1 and layers.2, and the third containing layers.3 and layers.4.
    This provides more explicit control over which modules belong to each chunk compared to split points.
    """

    pipeline_parallel_first_stage_less_layers: int = 1
    """
    The number of layers to reduce in the first stage of pipeline parallelism. This is because
    the first stage has the extra overhead of the embedding layer, which is not present in the other stages.
    """

    pipeline_parallel_last_stage_less_layers: int = 1
    """
    The number of layers to reduce in the last stage of pipeline parallelism. This is because
    the last stage has the extra overhead of the output layer, which is not present in the other stages.
    """

    pipeline_parallel_layers_per_stage: int | None = None
    """
    The number of layers per (virtual) pipeline stage. If specified, the module_fqns_per_model_part will be
    calculated from the number of layers and pipeline_parallel_degree. If not specified, the
    layers per stage will be inferred from the model, schedule, and pipeline_parallel_degree.
    """

    pipeline_parallel_schedule: str = "1F1B"
    """
    Specify the Pipeline Parallel schedule to use. The supported schedules are:
    https://github.com/pytorch/pytorch/blob/de4c2a3b4e89d96334dc678d1c3f2ae51a6630a0/torch/distributed/pipelining/schedules.py#L2161.
    The schedule must be compatible with the split points and stages_per_rank.
    Looped schedules (e.g. Interleaved1F1B) require specifying pipeline_parallel_degree = number of ranks,
    and split_points = number of stages - 1
    """

    pipeline_parallel_schedule_csv: str | None = ""
    """
    Specify the path to the pipeline parallel schedule csv file to use.
    The pipeline_parallel_schedule argument must be either
    PipelineScheduleSingle, PipelineScheduleMulti, or _PipelineScheduleRuntime.
    """

    pipeline_parallel_microbatch_size: int = 1
    """
    The size of each pipeline parallel microbatch (default 1).
    This value is used to compute the total number of microbatches by dividing local_batch_size with
    pipeline_parallel_microbatch_size.
    The global training batch size must be evenly divisible by pipeline_parallel_microbatch_size.
    """

    pipeline_parallel_expert_parallel_overlap: bool = True
    """Whether to turn on the optimization to overlap expert parallel and pipeline parallel
    communication. This is only effective when the pipeline parallel schedule is DualPipeV and
    pipeline_parallel_degree > 1 and expert_parallel_degree > 1.

    TODO: Does not support activation_checkpoint, set mode="none"
    """

    context_parallel_degree: int = 1
    """Context parallelism degree. 1 means disabled."""

    context_parallel_load_balancer: str | None = "headtail"
    """
    Load balancer type for context parallelism. Options:
    - "headtail": Use HeadTailLoadBalancer for SDPA
    - "ptrr": Use PTRRLoadBalancer for FlexAttention
    - None: Disable load balancing
    """

    def __post_init__(self):
        if self.context_parallel_load_balancer == "":
            raise ValueError(
                "context_parallel_load_balancer cannot be an empty string. "
                "Use None to disable load balancing."
            )

    context_parallel_rotate_method: Literal["allgather", "alltoall"] = "allgather"
    """
    The collective to use in context parallel SDPA for kv shards exchange.
    - 'allgather' means to all-gather all kv shards on ranks after the first sub-SDPA computation,
    - 'alltoall' means to all-to-all shuffle the kv shards.
    The default value is 'allgather'.
    """

    expert_parallel_degree: int = 1
    """
    Expert parallelism degree. 1 means disabled. No effect for non-MoE models.

    Currently, etp is either 1 or is the same as tp.

    Note that this is still an experimental feature. Some constraints will be
    relaxed soon when we have more flexible DeviceMesh support.
    """

    expert_tensor_parallel_degree: int = 1
    """
    Expert tensor parallelism degree. 1 means disabled. No effect for non-MoE models, or when ep = 1.
    With this option, the tensor parallel degree on routed experts can be different from that on other params.
    Currently, we only support either
    - [partial dp -> ep] etp = tp
    - [partial dp + all tp -> ep] etp = 1
    Note that this is still an experimental feature.
    """

    expert_parallel_comm_backend: Literal["standard", "deepep"] = "standard"
    """
    Expert-parallel communication backend. No effect for non-MoE models or when ep = 1.

    - "standard": Uses PyTorch all-to-all collectives (default)
    - "deepep": Uses DeepEP custom kernels for more efficient communication

    DeepEP requires installation:
    https://github.com/deepseek-ai/DeepEP.
    """


@dataclass
class Checkpoint:
    enable: bool = False
    """Whether to enable checkpoint"""

    enable_ft_dataloader_checkpoints: bool = True
    """
    Warning: Disabling this can have fault tolerant replicas training
    over the same data multiple times. Use it with caution if training
    over the same data is acceptable.

    Used to enable checkpointing the dataloader index for fault tolerant training with torchft.

    Fault tolerant training stores data loader index in the checkpoints, so that training can resume
    without going over the same batch twice.

    If enabled, data loader state is checkpointed. Otherwise, replicas
    will train over the same data multiple times, which can result in
    overfitting.

    The failed replcia will still recover other state e.g. model
    parameters from other replcias.

    Note, if regular checkpointing is enabled, we also checkpoint the
    data loader state. But when not using fault tolerance, the entire training starts from scratch.
    """

    folder: str = "checkpoint"
    """
    The folder to store the checkpoints.
    When enable is set to true, checkpoints will be in {--job.dump_folder}/{--checkpoint.folder}.
    """

    interval: int = 500
    """Checkpointing interval in steps."""

    initial_load_path: str | None = None
    """
    This option specifies the path to the initial checkpoint to load, which is
    particularly useful for resuming training from a previous run with a
    different output path or when loading a checkpoint from a pre-trained model.
    If the checkpoint folder for the current run is not empty,
    located at {--job.dump_folder}/{--checkpoint.folder}, this option will be ignored.
    This feature allows users to load an initial checkpoint from a different folder and
    continue training, saving new checkpoints to the specified folder without affecting
    the existing ones.

    Note that the path should contain the full path to the checkpoint folder,
    including the step number, if any; for example,
    "//pre_train/checkpoints/llama3/llama3_8b/step_10000".
    """

    initial_load_model_only: bool = True
    """
    This option specifies if only the model should be loaded during the initial
    checkpoint load. The option is only used when `initial_load_path` is specified.
    If False, the checkpoint at `initial_load_path` is treated as a standard training
    checkpoint, including optimizer, lr scheduler, training states, etc.
    The default setting for this option is True. Note that you will have to use
    `--checkpoint.no_initial_load_model_only` to override the default setting.
    """

    initial_load_in_hf: bool = False
    """
    Enable the use of HuggingFace's safetensors format for checkpointing. The option
    is only used when `initial_load_path` is specified. This will load checkpoints
    in HF's model definition and safetensors format instead of the default torchtitan
    model definition and DCP format, after necessary model state dict transformation.
    `initial_load_model_only` must be true because safetensors doesn't support saving
    non-tensors. The default value is False.
    """

    initial_load_in_hf_quantized: bool = False
    """
    Enable loading of HuggingFace's safetensors format with quantized state dict keys. The option
    is only used when `initial_load_path` and `initial_load_path_in_hf` is specified. This will load
    checkpoints in HF's model definition and dequantize on model weights if necessary. To support
    this parameter, the model need to define proper HuggingFaceStorageReader to perform dequantize.
    """

    skip_last_save: bool = False
    """
    When skip_last_save=True, the final checkpoint at the end of training will be skipped.
    This is useful when you only want intermediate checkpoints and don't need the final model.
    The default value is False.
    """

    last_save_model_only: bool = True
    """
    When last_save_model_only=True, only the model will be saved at the end of training,
    the last save.  With this, checkpoints can be loaded using `torch.load(..., weights_only=True)`
    after conversion.  When last_save_model_only=False, the full checkpoint will be saved.
    A full checkpoint includes model, optimizer and train_state, which can be used to resume training.
    The default value is True.
    """

    last_save_in_hf: bool = False
    """
    Enable the use of Hugging Face's safetensors format for checkpointing. This will save the
    final checkpoints in safetensors format instead of the default DCP format, after necessary
    model state dict transformation. There will be a performance cost in using this as we need
    to consolidate the sharded tensors to full tensors as a separate step.
    last_save_model_only must be true because safetensors doesn't support saving
    non-tensors. On load, this argument isn't needed as we will detect whether the loaded
    checkpoint is in safetensors format or not. The default value is False.
    """

    export_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    """
    Converts to the specified precision when training completes and last_save_model_only=true.
    """

    async_mode: Literal["disabled", "async", "async_with_pinned_mem"] = "disabled"
    """
    Which async checkpoint mode to use. Currently there are 3 different modes.

    - "disabled": synchronized checkpointing will be used.
    - "async": torch.distributed.checkpoint.async_save will be used.
    - "async_with_pinned_mem": this option utilizes a dedicated pinned memory space and creates a
      separate process for faster GPU->CPU transfer performance and eliminating GIL contention.
      The cost is increased CPU memory usage. If insufficient CPU memory is available, performance
      may degrade due to memory paging. For most users, "async" should suffice as the performance
      overhead is typically small (on the order of tens of seconds) compared to checkpointing
      frequency. This mode can be employed to pursue near-zero checkpointing times
      (e.g., < 1 second) given appropriate hardware support such as ample CPU memory and fast PCIe.

    "disabled" is the default mode.
    """

    keep_latest_k: int = 10
    """
    Keeps only the latest k checkpoints, and purging older ones. If 0, keep all checkpoints.
    K cannot be 1 as the last one may be in the process of being saved. As a result,
    the metadata of the last one may not be ready yet. The default value is 10 to avoid
    filling up the disk.
    """

    load_step: int = -1
    """Load the checkpoint at the specified step. If -1, load the latest checkpoint."""

    exclude_from_loading: list[str] = field(default_factory=list)
    """
    Exclude specific keys from being loaded from the checkpoint.
    Provide a comma-separated list of keys to exclude, e.g. 'optimizer,lr_scheduler,dataloader'.
    This will load the model only, excluding the specified keys.
    """

    enable_first_step_checkpoint: bool = False
    """
    Enable the checkpoint save at first step. This will save a checkpoint immediately
    after the first step to ensure checkpointing functions correctly. This is useful
    when running on a new cluster or storage to verify checkpointing without waiting
    for many steps or checkpointing too frequently. The default value is False.
    """

    create_seed_checkpoint: bool = False
    """
    Initializes the full model without applying parallelisms, and then saves it as a seed checkpoint.
    Note: requires user to call train.py without specifying any parallelisms, e.g. NGPU=1.
    Could be implemented as a separate script, but this way shares more code.
    """

    load_only: bool = False
    """
    In certain scenarios, you may only need to load checkpoints for verification or debugging
    purposes, without saving any new checkpoints. For example, you might use seed checkpoints
    to validate model correctness. Enabling this option allows checkpoints to be loaded
    without saving any during the training.
    """

    method: Literal["dcp", "gemini", "moevement"] = "dcp"
    """
    Checkpointing implementation: "dcp" (torch distributed checkpoint, the
    default), "gemini" (in-memory snapshots interleaved into FSDP comm gaps),
    or "moevement" (sparse in-memory checkpointing ported from MoEvement).
    The legacy use_gemini flag is honored as an alias for method="gemini".
    """

    use_gemini: bool = False

    def resolved_method(self) -> str:
        """method, with the legacy use_gemini flag folded in."""
        if self.use_gemini and self.method == "dcp":
            return "gemini"
        return self.method

    moevement_mem_fs_folder: str = ""
    """tmpfs folder for moevement's kill-survivable snapshot pools
    (counterpart of gemini_mem_fs_folder)."""

    moevement_pcie_bandwidth_gbs: float = 0.0
    """Effective device->host bandwidth (GiB/s) for moevement's window sizing
    — Algorithm 1's B_PCIe input: each iteration's snapshot bytes are budgeted
    against iter_time * bandwidth * moevement_snapshot_overlap_target.
    0 (the default) means AUTO: profile the device once at init, on the
    snapshot capture stream, with a few 256 MiB pinned D2H transfers (median).
    Any positive value overrides the measurement verbatim."""

    moevement_snapshot_overlap_target: float = 1.0
    """Fraction of one iteration's PCIe budget the per-iteration sparse
    snapshot may consume. 1.0 (default) is recovery-optimal (smallest
    window); lower values trade a longer window for less D2H pressure."""

    moevement_w_sparse_override: int = 0
    """Pin the sparse window length to this many iterations instead of
    deriving it from the PCIe budget. 0 (default) derives it. Also pins the
    schedule cadence world-wide regardless of per-rank iteration timing."""

    moevement_profile_policy: bool = False
    """Force Algorithm 1 to run live and (re)write the policy profile at
    moevement_policy_profile_path, ignoring any artifact already there.

    This is the moevement analogue of --leto.profile_init: a dedicated,
    short profiling run (see awsexps/common/run_init.sh) produces
    <dump_dir>/moevement_profile/policy.json, and every later run of that
    workload reuses the recorded w_sparse instead of re-deriving it. Without
    this flag a run reuses an existing matching profile and only falls back
    to the live policy when there is none (or its provenance does not match).
    Ignored under checkpoint.moevement_w_sparse_override, which always wins."""

    moevement_policy_profile_path: str = ""
    """Where the recorded policy profile lives. Empty (default) means
    <job.dump_folder>/moevement_profile/policy.json."""

    moevement_reorder_threshold: float = 0.10
    """Relative shift in an expert's rolling popularity that counts it as
    "changed" when deciding whether to regenerate the snapshot schedule."""

    moevement_reorder_fraction: float = 0.25
    """Fraction of experts that must have shifted by more than
    moevement_reorder_threshold to trigger a schedule regeneration."""

    moevement_activation_count_window_iters: int = 100
    """Length (iterations) of the rolling window of per-expert token counts
    driving popularity ordering and reorder detection."""

    moevement_iter_time_window_iters: int = 50
    """Effective length (iterations) of the wall-clock iteration-time EMA
    that feeds window sizing."""

    moevement_initial_iter_time_sec: float = 1.0
    """Iteration-time seed for window sizing before any steps have been
    measured."""

    moevement_upstream_logging: bool = True
    """Whether to log pipeline stage-boundary RECEIVES (forward activations
    arriving from the previous stage + gradients arriving from the next,
    keyed by iteration/microbatch/PRODUCING-virtual-stage/direction) into a
    kill-survivable shm ring for log-fed replay. Receive-side (not the
    paper's send-side) because leto kills every rank: logging what you
    receive makes each rank self-sufficient after relaunch, with identical
    storage and copies. Effective only when pipeline parallelism is enabled;
    at PP=1 the logger is not constructed."""

    moevement_frozen_skip: bool = False
    """Paper §3.3: during sparse->dense REPLAY, run frozen operators
    forward + input-gradient ONLY — skip their weight-gradient computation
    and their optimizer update instead of computing the full backward and
    dropping the grads afterwards (the default, numerically equivalent but
    zero-FLOP-saving behaviour).

    Mechanism: whole-tensor operators (non_expert, gate) get
    ``requires_grad_(False)``; grouped expert tensors are per-SLICE, so a
    layer's experts are skipped only when EVERY EP-local expert of that
    layer is frozen (the weights are then handed to the compiled grouped-mm
    detached, which drops the wgrad matmul from the AOTAutograd backward
    graph). Partially-frozen expert tensors keep the full backward and fall
    back to grad masking, as do model parts that would otherwise be left
    with no trainable parameter at all.

    Requires the replayed window to carry per-iteration clip-norm anchors
    (captured only when this flag is on): the total grad norm feeding
    ``training.max_norm`` is otherwise computed over a smaller grad set and
    the surviving operators' updates would drift. A window captured without
    the flag disables the skip on restore, loudly."""

    moevement_frozen_skip_warmup: int = 0
    """Warmup-only: exercise the frozen-skip path for this many steps at the
    start of the run (steps 1..N) so torch.compile/inductor caches every
    graph variant a real replay will need. Two activation patterns alternate
    (odd steps freeze the dense backbone + routers, even steps freeze every
    expert module), so **use N >= 2** to cover both. Numerics for those steps
    are intentionally perturbed (frozen operators do not update), so this is
    for THROWAWAY warmup/profiling jobs only — never for a measured run. 0
    (default) disables it."""

    moevement_replication: bool = True
    """Whether to replicate finalized windows to the pair rank
    (rank + world/2) over a dedicated gloo group, enabling the uniform
    faulty-rank peer-fetch recovery (plan §3.7). Auto-disabled (with a log
    line) when the world is smaller than 2 or odd."""

    gemini_profile_comm_gaps: bool = False
    """Whether to profile communication gaps for Gemini checkpointing"""

    gemini_skip_first_k: int = 5
    """Number of initial iterations to skip when profiling communication gaps (warmup)"""

    gemini_comm_gaps_folder: str = "gemini_comm_gaps"
    """Folder to save communication gap profiles (relative to dump_folder)"""

    gemini_mem_fs_folder: str = ""


@dataclass
class ActivationCheckpoint:
    mode: Literal["selective", "full", "memory_budget", "none"] = "selective"
    """Type of activation checkpointing to use"""

    selective_ac_option: str = "2"
    """
    Selective activation checkpointing options ['int', 'op'].
    'int' (e.g., 2) for every nth layer, or 'op' for op level ac.
    """

    per_op_sac_force_recompute_mm_shapes_by_fqns: list[str] = field(
        default_factory=lambda: ["moe.router.gate"]
    )
    """
    When per-op selective ac is used, this list of fully qualified names is used
    to determine which mm shapes to force recompute, rather than being considered
    by rest of the sac policy, e.g save every other mm. Only nn.Linear modules are
    supported today.

    Note: this config applies to mms not limited to those matching the specified
    fqns, e.g. if "moe.router.gate", corresponding to Linear(in, out), is specified,
    ANY mm with shape matching (*, in) x (in, out) will be force recomputed.
    """

    early_stop: bool = False
    """
    Whether to stop recomputing early when all activations have already been
    rematerialized.
    """

    memory_budget: float = 0.5
    """
    When mode is set to "memory_budget", this value determines how much
    partitioner in the compiler should trade off compute for memory.
    0.0 corresponds to the activation memory from applying
    activation checkpointing to the full compiled region, and 1.0 corresponds to
    the activation memory from the default runtime-optimized strategy. Read here:
    https://pytorch.org/blog/activation-checkpointing-techniques/
    """

    visualize_memory_budget_pareto: bool = False
    """
    This dumps out a SVG visualization of the expected runtime vs. activation
    memory tradeoffs for all memory budget values from 0 to 1 in increments of
    0.05 in {--job.dump_folder}/memory_budget_pareto folder. See an example here:
    https://github.com/pytorch/pytorch/pull/126320#discussion_r1625104015
    """

    preserve_rng_state: bool = False
    """
    If deterministic output compared to non-checkpointed passes is required, set
    to true. Results in stashing and restoring the RNG state during each checkpoint,
    may be slower. See https://docs.pytorch.org/docs/stable/checkpoint.html
    for details.
    """

    determinism_check: str = "default"
    """
    A string specifying the determinism function. See
    https://docs.pytorch.org/docs/stable/checkpoint.html for details.
    """

    debug: bool = False
    """
    Capture ac debug information. Will be slower. See
    https://docs.pytorch.org/docs/stable/checkpoint.html for details.
    """


@dataclass
class Compile:
    enable: bool = False
    """Whether to apply torch.compile"""

    components: list[str] = field(default_factory=lambda: ["model", "loss"])
    """Which components to compile"""
    backend: str = "inductor"


@dataclass
class Float8Linear:
    enable_fsdp_float8_all_gather: bool = False
    """Whether enable float8 all-gather in FSDP, recommended for tensorwise scaling"""

    precompute_float8_dynamic_scale_for_fsdp: bool = False
    """Whether precompute float8 scales dynamically for FSDP, recommended for tensorwise scaling"""

    recipe_name: Literal["tensorwise", "rowwise", "rowwise_with_gw_hp"] | None = None
    """If specified, creates float8 config from recipe name"""

    filter_fqns: list[str] = field(default_factory=list)
    """
    Comma-separated list of fully qualified names of modules to skip applying float8 training to.
    nn.Linear modules with any dim size not divisible by 16 are always skipped due to hardware requirements.
    Example: --quantize.linear.float8.filter_fqns "attention.wq,attention.wk,attention.wv,output"
    """
    emulate: bool = False
    """
    If True, emulation is used instead of hardware accelerated gemm. This is for test purpose only,
    as the current CI does not have sm_89 capability, required by Float8.
    Not compatible with torch.compile.
    """


@dataclass
class Float8GroupedMM:
    fqns: list[str] | str = field(default_factory=list)
    """
    *Prototype feature, performance optimization still in progress*
    Comma-separated list of fully qualified names of MoE Layers to apply FP8 dynamic quantization on grouped GEMM operations.
    This is a prototype feature that requires the torchao nightly build.
    Example: --quantize.grouped_mm.float8.fqns="experts"
    """


@dataclass
class MXLinear:
    mxfp8_dim1_cast_kernel_choice: Literal["triton", "cuda", "torch"] = "triton"
    """
    Temp work around for inductor performance gap.

    CUDA is recommended for best performance.

    Example: --quantize.linear.mx.mxfp8_dim1_cast_kernel_choice="cuda"
    """

    recipe_name: str = "mxfp8_cublas"
    """
    If specified, creates MX config from recipe name. See
    https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats for more information.
    Example: --quantize.linear.mx.recipe_name="mxfp8_cublas"
    """

    filter_fqns: list[str] = field(default_factory=lambda: ["output"])
    """
    Comma-separated list of fully qualified names of modules to skip applying mxfp8 training to.
    nn.Linear modules with any dim size not divisible by 16 are also always skipped due to hardware requirements.
    By default we always skip the output layer.
    Example: --quantize.linear.mx.filter_fqns="attention.wq,attention.wk,attention.wv,output"
    """


@dataclass
class MXGroupedMM:
    recipe_name: Literal["mxfp8"] = "mxfp8"
    """
    Quantization recipe name for grouped GEMMs. Options: ["mxfp8"]

    Example: --quantize.grouped_mm.mx.recipe_name="mxfp8"
    """

    fqns: list[str] | str = field(default_factory=list)
    """
    *Prototype feature, performance optimization still in progress*
    Comma-separated list of fully qualified names of MoE modules to apply MXFP8 dynamic quantization on grouped GEMM operations.
    This is a prototype feature that requires the torchao nightly build.
    Example: --quantize.grouped_mm.mx.fqns="experts"
    """


@dataclass
class QuantizedLinear:
    float8: Float8Linear = field(default_factory=Float8Linear)
    """FP8 training config for nn.Linear layers"""

    mx: MXLinear = field(default_factory=MXLinear)
    """MX training config for nn.Linear layers"""


@dataclass
class QuantizedGroupedMM:
    float8: Float8GroupedMM = field(default_factory=Float8GroupedMM)
    """FP8 training config for grouped GEMMs"""

    mx: MXGroupedMM = field(default_factory=MXGroupedMM)
    """MX training config for grouped GEMMs"""


@dataclass
class Quantize:
    linear: QuantizedLinear = field(default_factory=QuantizedLinear)
    """Quantized training config for nn.Linear layers"""

    grouped_mm: QuantizedGroupedMM = field(default_factory=QuantizedGroupedMM)
    """Quantized training config for grouped GEMMs"""


@dataclass
class Comm:
    init_timeout_seconds: int = 300
    """Timeout for communication operations, during initialization and first train step."""

    train_timeout_seconds: int = 100
    """
    Timeout for communication operations after the first train step --
    usually a tighter bound than during initialization.
    """

    trace_buf_size: int = 20000
    """Flight recorder ring buffer size, >0 means recording by default, 0 means disabled"""

    save_traces_folder: str = "comm_traces"
    """Flight recorder trace files location"""

    save_traces_file_prefix: str = "rank_"
    """Flight recorder trace files prefix"""

    mode: Literal["default", "fake_backend", "local_tensor"] = "default"
    """
    Communication mode for distributed training.

    Options:
    - "default": Normal distributed training with real communication
    - "fake_backend": Fake comm backend for dry run mode only (configuration validation without GPU)
    - "local_tensor": Local tensor mode for debugging purposes. There will be only one process
      regardless of the number of GPUs. LocalTensor will simulate the computation by running one
      rank after another. While the performance will be slow, the numerics should be the same.
      This enables us to verify numerics with fewer GPUs. For example, we can directly run 5D
      parallelisms within a single node to reduce the combinations we need to use in integration tests.

    NOTE: local_tensor is an experimental feature and automatically uses fake_backend internally.
    """


@dataclass
class MemoryEstimation:
    enable: bool = False
    """Whether to estimate memory usage for FSDP"""

    disable_fake_mode: bool = False
    """Whether to estimate memory under FakeTensorMode"""


@dataclass
class FaultTolerance:
    enable: bool = False
    """
    Enable TorchFT integration. When TorchFT is enabled, HSDP will be used.
    And --fault_tolerance.data_parallel_replicate_degree should be 1 and
    --fault_tolerance.group_size will be used to control the maximum
    replicate group size as the replicate group size is dynamic.
    Note that this is still an experimental feature.
    """

    process_group: str = "gloo"
    """
    The process group to use for fault tolerance. Currently, only "gloo" and "nccl" are supported.
    """

    process_group_timeout_ms: int = 10000
    """
    The process group will abort if operations don't succeed within this duration.
    Note: This currently only works with gloo process group.
    """

    replica_id: int = 0
    """The TorchFT replica ID of this run."""

    group_size: int = 0
    """
    The number of TorchFT replicate groups. This number will be used for
    dataloader to split the dataset across the replicate groups and FSDP
    dimension
    """

    min_replica_size: int = 1
    """The minimum number of FT replica for each step."""

    semi_sync_method: str | None = None
    """
    The algorithm to use for semi-sync training. Currently, only "local_sgd" and "diloco" from
    torchft are supported
    (https://github.com/pytorch/torchft/blob/360c5c534bdeac959507e9d238ba9f3902d3fda9/torchft/local_sgd.py#L41)
    """


@dataclass
class Experimental:
    custom_import: str = ""
    """
    This option enables the importation of external modules.
    Currently, it only supports dotted import modules (e.g., some_package.model_x).
    It is the user's responsibility to ensure that the specified path can be
    successfully imported. One method to achieve this, you can place your module
    inside the ``torchtitan/torchtitan`` folder and execute ``pip install -e .`` to
    make it available for import.
    """

    custom_args_module: str = ""
    """
    DEPRECATED (moved to Job.custom_config_module). Will be removed soon.

    This option allows users to extend TorchTitan's existing JobConfig by extending
    a user defined JobConfig dataclass. Similar to ``--experimental.custom_import``, the user
    needs to ensure that the path can be imported.
    """


@dataclass
class Validation:
    enable: bool = False
    """Enable validation to default run validation after each training loop"""

    dataset: str = "c4_validation"
    """Dataset to use for validation"""

    dataset_path: str | None = None
    """Path to dataset to use for validation"""

    local_batch_size: int = 8
    """Batch size for validation"""

    seq_len: int = 2048
    """Sequence length for validation"""

    freq: int = 10
    """Frequency of validation"""

    freq_offset: int = 0
    """Phase offset for the validation cadence: validate at steps where
    ``(step - freq_offset) % freq == 0``. Step 1 is always validated.
    Useful when training has a periodic event at ``step % freq == 0``
    (e.g. fault injection at every freq-th step) that would otherwise
    eat the validation. With freq=10 and freq_offset=1, validation runs
    at steps 1, 11, 21, 31, …"""

    steps: int = -1
    """
    Number of steps to take in the validation set, -1 means consuming all the data in the validation dataset
    WARNING: When setting to -1 there could be hangs due to mismatch among ranks
    """

    dataloader: DataLoader = field(default_factory=DataLoader)
    """DataLoader configuration"""

    def __post_init__(self):
        assert (
            self.steps > 0 or self.steps == -1
        ), "validation steps must be positive or -1"


@dataclass
class Debug:
    seed: int | None = None
    """Choose the base RNG seed used for training"""

    deterministic: bool = False
    """Use deterministic algorithms wherever possible, may be slower"""

    deterministic_warn_only: bool = False
    """Only warns about ops without deterministic implementations rather than erroring out  """

    moe_force_load_balance: bool = False
    """If True, we force each experts to get the same amount of tokens via round-robin. This option is for debugging usage only."""

@dataclass
class Leto:
    init_mode: str = "baseline"
    """Initialization mode: 'baseline' (original order, CUDA immediately),
    'reordered' (defers CUDA until after standby poll)."""

    profile_init: bool = False
    """If True, profile each init task's wall time and GPU memory usage (via pynvml).
    Writes a JSON file to dump_folder/init_profile_rank{rank}.json.

    When set, the active group skips the training loop entirely (see
    torchtitan.train.main): a profile run only needs to capture init cost, and
    the training loop's activation/optimizer allocations would OOM on top of the
    standby group's still-resident init memory. With enable_standby the active
    group additionally waits for the standby to finish writing its profile
    before exiting (see profile_init_wait_timeout_s)."""

    profile_init_wait_timeout_s: float = 1800.0
    """On a profile_init + enable_standby run, how long (seconds) the active
    group waits, after finishing init, for the standby group to finish writing
    its init profile (init_profile/<mode>/{rank_*.json,solution.json}) before
    exiting. The wait prevents the active's exit from triggering a shutdown that
    SIGKILLs the standby mid-profile. On timeout the active logs an error and
    exits anyway."""

    eager_init_list: list[str] = field(default_factory=list)
    """
        all, none, nccl
    """

    enable_stage_input_record: bool = False
    """Record stage inputs and outputs during the first training iteration.
    Saves per-rank JSON files under stage_inputs_folder. Required before
    enable_skip_shape_inference can be used."""

    enable_skip_shape_inference: bool = False
    """If True and a recorded shape file exists, pass input_args/output_args as
    meta tensors to PipelineStage, bypassing PyTorch's runtime _shape_inference
    (which runs a full dummy forward pass across all PP ranks on the first step).
    Requires a prior run with enable_stage_input_record=True."""

    enable_stage_warmup: bool = False
    """Warmup stages with recorded inputs before training starts"""

    stage_warmup_mode: str = "fake"
    """Mode for stage warmup. "fake" (default) traces the model under
    FakeTensorMode with monkey-patched MoE stubs to avoid real kernel
    launches — fast but FX-graph divergence vs real-mode AOT compile
    causes small bf16 drift on MoE-heavy workloads. "real" runs an actual
    CUDA forward+backward with zero-init recorded inputs; the FX graph
    matches real training exactly so the warmup-compiled cache produces
    bit-identical losses, at the cost of one extra real iteration of
    activation memory and compute."""

    stage_inputs_folder: str = "stage_inputs"
    """Folder to save/load recorded stage shape metadata (relative to dump_folder)."""

    enable_rmp_gpu: bool = False
    """Enable RMP (Remote Memory Provider) for shared GPU memory across processes."""

    enable_skip_commit: bool = False

    rmp_server_port: int = 29051
    """Base port for RMP servers. Actual port = rmp_server_port + local_rank.

    Chosen below the kernel's default ephemeral range (32768-60999) so random
    outgoing TCP connections never collide with our listeners. The worker
    controller rotates this base by 0/100/.../900 per launch (see
    RmpProcessGroup), staying clear of the torchrun rendezvous ports
    (29610-29625 active, 29710-29725 standby)."""

    enable_standby: bool = False
    """If True, launch standby torchrun group for faster fault recovery."""

    standby_poll_interval: float = 0.5
    """Polling interval in seconds for standby processes checking activation status."""

    progressive_init: bool = False
    """If True, standby advances through init tasks gated on the active's free-GPU
    signal (see init_profile/{mode}/solution.json). Requires enable_standby."""

    progressive_reservation_margin_mb: int = 512
    """Global safety headroom (MiB) the active's reservation broker keeps free,
    and the master switch for the whole reservation/reclaim mechanism under
    progressive_init + enable_standby:

      * > 0 — the active installs the reservation broker: it grants a standby
        reservation only while `others_used + granted <= capacity - margin`,
        and reclaims (kills) the standby before growing into reserved memory
        when an allocation would breach the margin. The standby drives the shm
        reservation handshake, advancing each init task only once granted.
      * == 0 — no broker, no reclaim. The standby skips the reservation
        handshake entirely and just runs its init tasks in the reordered
        (solver-scheduled) order, ungated. This is the grant-only control that
        OOMs under memory pressure.

    The margin absorbs profiling error, NVML granularity, and allocations
    outside the caching allocator (CUDA context growth, NCCL, cuBLAS
    workspaces). Bump it if the active OOMs from un-brokered allocations the
    broker can't see."""

    progressive_poll_interval_ms: int = 100
    """Standby poll interval (ms) while waiting for a reservation verdict."""

    progressive_protocol: str = "two_phase"
    """Reservation protocol the standby's progressive init uses:
      "two_phase"  — RESERVE (soft, cancellable) on all ranks, then GRANT
                     (commit) on all ranks; any phase-2 denial rolls every
                     rank back. Default.
      "grant_only" — single-phase ablation: GRANT commits directly against
                     the estimate, no reservation phase and no rollback (a
                     rank that committed while peers denied keeps the
                     phantom entitlement). For measuring two_phase's impact."""

    progressive_solution: str = "solver"
    """Which init-task schedule the progressive standby follows:
      "solver"        — the solver-optimized order (init_profile/<mode>/solution.json)
      "no_reordering" — progressive gating with the baseline execution order
                        (solution_no_reordering.json). A/B control that
                        isolates the benefit of the solver's reordering; both
                        files are written by the same profile_init run."""

    progressive_zero_delta_threshold_mb: float = 1.0
    """Tasks with delta_mb below this threshold are treated as CPU-only and
    bypass the gated wait under progressive_init."""

    fmcb_est_ttl_ms: int = 0
    """Estimate-cache TTL (ms) for the OOM-safeguard FreeMemoryCallback. An
    allocator cache miss re-reads NVML (and walks the allocator snapshot)
    only when the cached component snapshot is older than this, or when a
    conservative screen of the cached values signals possible
    cancellation/pressure — CANCEL/KILL always re-read fresh. 0 = read NVML
    on every miss (the pre-cache behavior; costs ~1% steady-state at MoE
    cache-miss rates).

    DEFAULT DISABLED (0) pending the steady-state overhead breakdown: the
    cache empirically removes ~86% of the parked-standby tax at ttl=60000,
    but the breakdown showed the dominant cost is the allocator-snapshot
    WALK inside the callback's full path (not the NVML read the cache was
    named for), and that attribution is not yet fully confirmed. Keeping
    the default at 0 means every run uses the original, understood per-miss
    behavior; re-enable (e.g. 100) only once the breakdown is settled. The
    fast-path code in leto_free_mem_callback.cpp is dormant while ttl=0."""

    fmcb_timing: bool = False
    """Enable per-component wall-time instrumentation inside the OOM-safeguard
    FreeMemoryCallback (diagnostic; default off, zero overhead when off). When
    on, the callback accumulates ns-sums and call-counts for each sub-op — the
    two NVML calls (memoryinfo, computeprocs), each allocator-snapshot walk
    (tail_reuse_bytes, releasable_lower_bound) and the snapshot() inside it —
    plus a request-size histogram and the whole-Execute() total. train.py logs
    the cumulative counters once per step (`fmcb_timing step=N {json}`);
    awsexps/measure_mem/steady/analyze_fmcb_timing.py diffs consecutive steps
    and buckets by step%8 to attribute the drain-step (step%8==2) overhead.
    Pair with fmcb_est_ttl_ms=0 so every miss takes the full path (clean
    sampling of the walk/NVML costs)."""

    fault_injection_step_enabled: bool = False
    """Enable step-based fault injection (worker self-injects faults at specific steps)."""

    fault_injection_step_interval: int = 0
    """Inject a fault every N training steps. 0 = disabled."""

    fault_injection_step_rank_mode: str = "single"
    """Which ranks fault: 'single' (one rank per step), 'tp' (all ranks in TP
    group), 'fsdp' (all ranks in FSDP group), 'prob' (each rank independently
    decides, with probability fault_injection_step_prob), 'random' (per-step
    deterministic 50/50: 'single' kernel_trap, or target_rank's FSDP group
    os._exit(1))."""

    fault_injection_step_prob: float = 0.5
    """For fault_injection_step_rank_mode == 'prob': per-fault-step independent
    probability in [0, 1] that each rank faults. Deterministic per
    (fault_injection_step_seed, step, global_rank), so the schedule is
    reproducible across restarts. Unused for the other rank modes."""

    fault_injection_step_barrier: bool = False
    """If True, call dist.barrier() before raising the fault exception."""

    fault_injection_step_seed: int = 42
    """Seed for deterministic hash used in rank selection for step-based fault injection."""

    fault_injection_start_step: int = 0
    """First step at which faults can be injected (0 = from the beginning)."""

    fault_injection_end_step: int = 0
    """Last step at which faults can be injected (0 = until training ends)."""

    fault_injection_raise_error: bool = False
    """If True, raise RuntimeError on fault. If False (default), skip optimizer.step() instead."""

    fault_injection_target_rank: int = -1
    """If >= 0, always fault this specific rank instead of selecting via hash. Compatible with rank_mode."""

    fault_injection_nocommit: bool = False
    """If True, reset dataloader and lr scheduler at the start of the next step after a faulted step."""

    fault_injection_kernel_trap: bool = False
    """If True, instead of raising/skipping when a step fault fires, arm the
    NVBit ``adam_trap`` tool so the next fused-AdamW kernel launch traps in
    GPU code. Requires ``adam_trap.so`` loaded via ``CUDA_INJECTION64_PATH``
    (leto's launcher sets this automatically when
    ``fault_injection.mode == 'kernel_trap'``)."""

    dump_optimizer_info: bool = False
    """If True, dump optimizer setup (shapes, dtypes, hyperparams) to optimizer_info.json on step 1."""

    resilient_opt_fault_injection: bool = False
    """Enable fault injection in resilient optimizer marker."""

    resilient_opt_fault_injection_prob: float = 0.005
    """Per-fill_() probability of injecting a fault (only when fault injection enabled)."""

    enable_cpu_snapshot_opt: bool = False
    """Use AsyncCpuSnapshotOptimizer baseline (mutually exclusive with enable_rmp_gpu)."""

    disable_resilient_opt: bool = False
    """If True with enable_rmp_gpu=True, allocate the RMP-GPU shared tensor pool
    (so model/optim params are RMP-backed) but do NOT wrap the optimizer with
    ResilientOptimizer. The training step path falls through to
    ``self.optimizers.step()`` and recovery happens only via
    ``checkpointer.load`` at the start of train(). Independent of
    ``enable_skip_commit``: the per-step ``rmp_manager.maybe_commit`` still
    runs unless ``enable_skip_commit`` is also set. Combine with
    ``enable_skip_commit`` to isolate the GPU mirror cost alone, or leave
    ``enable_skip_commit`` off to isolate the CPU-metadata-commit cost."""

    oom_test_alloc_step: int = 0
    """Test hook for the OOM safeguard. If > 0, allocate a fresh CUDA tensor
    of size `oom_test_alloc_mb` MiB at the start of this training step on
    every active rank. The allocation is sized to be a cache-miss expansion
    (so the FreeMemoryCallback fires) and is freed immediately afterwards.
    0 disables. Use together with progressive_reservation_margin_mb > 0 to
    verify the end-to-end reclaim+relaunch path."""

    oom_test_alloc_mb: int = 0
    """MiB of GPU memory to allocate at oom_test_alloc_step. Should be sized
    so that the allocation would push effective free GPU memory below the
    reservation margin (progressive_reservation_margin_mb) — so the broker's
    reclaim callback actually fires — and so the allocation can succeed after
    the standby is reclaimed."""

    oom_test_standby_alloc_mb: int = 0
    """OOM-safeguard E2E test hook (standby side). If > 0, each standby rank
    reserves (through the broker) and holds this many MiB of GPU memory after
    its progressive init, before parking. This gives the standby a controlled,
    substantial footprint so that the active's reclaim of it is the difference
    between OOM and success when the active's usage spikes
    (oom_test_alloc_step / oom_test_alloc_mb). Requires progressive_init and
    progressive_reservation_margin_mb > 0 (the reclaim path); with margin == 0
    the standby holds the memory ungated (the OOM control)."""

    standby_test_occupy_mbs: str = ""
    """OOM-safeguard test hook: comma-separated MiB sizes (e.g. "100,500,2048").
    For each entry i a synthetic init task `standby_test_occupy_{i}` is
    appended to the init sequence; on STANDBY ranks it allocates and HOLDS that
    much GPU memory (no-op on the active). This gives the standby a controlled,
    tunable footprint so the OOM-safeguard reclaim path can be exercised even
    when the model's natural standby footprint is small. The tasks participate
    in the init profile and the solver schedule, so changing this value
    requires re-running profile_init. On activation (promotion) the held
    ballast is released before training starts."""

@dataclass
class JobConfig:
    """
    Default container for training configuration.
    """

    job: Job = field(default_factory=Job)
    profiling: Profiling = field(default_factory=Profiling)
    metrics: Metrics = field(default_factory=Metrics)
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)
    training: Training = field(default_factory=Training)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    activation_checkpoint: ActivationCheckpoint = field(
        default_factory=ActivationCheckpoint
    )
    compile: Compile = field(default_factory=Compile)
    quantize: Quantize = field(default_factory=Quantize)
    comm: Comm = field(default_factory=Comm)
    memory_estimation: MemoryEstimation = field(default_factory=MemoryEstimation)
    fault_tolerance: FaultTolerance = field(default_factory=FaultTolerance)
    experimental: Experimental = field(default_factory=Experimental)
    validation: Validation = field(default_factory=Validation)
    debug: Debug = field(default_factory=Debug)
    leto: Leto = field(default_factory=Leto)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def maybe_log(self) -> None:
        if self.job.print_config:
            logger.info(
                f"Running with configs: {json.dumps(self.to_dict(), indent=2, ensure_ascii=False)}"
            )

        if self.job.save_config_file is not None:
            config_file = os.path.join(self.job.dump_folder, self.job.save_config_file)
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    os.makedirs(os.path.dirname(config_file), exist_ok=True)
                    with open(config_file, "w") as f:
                        json.dump(self.to_dict(), f, indent=2)
                logger.info(f"Saved job configs to {config_file}")
            else:
                logger.warning(
                    "Job configs logging is disabled due to torch.distributed not initialized."
                )
