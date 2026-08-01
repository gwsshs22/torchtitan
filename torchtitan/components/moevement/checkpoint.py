import glob
import os
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import _init_optim_state
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from torchtitan.components.checkpoint import DATALOADER, LR_SCHEDULER, ModelWrapper
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.moevement.conversion import (
    _build_bundle,
    apply_iteration,
    find_committed_windows,
    load_usable_windows,
    restore_rng,
    window_used_nbytes,
)
from torchtitan.components.moevement.replication import (
    agreed_window,
    build_replicator,
    vote_constraint,
)
from torchtitan.components.moevement.freeze import FrozenSkipController
from torchtitan.components.moevement.operators import (
    OperatorKind,
    _layer_label,
    discover_operators,
)
from torchtitan.components.moevement.policy_profile import (
    build_payload,
    build_provenance,
    default_profile_path,
    describe,
    read_profile,
    usable_w_sparse,
    write_profile,
)
from torchtitan.components.moevement.scheduler import SparseCheckpointScheduler
from torchtitan.components.moevement.snapshot_engine import (
    MoevementSnapshotEngine,
    measure_d2h_bandwidth_gbs,
)
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Checkpoint as CheckpointConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.moe.moe import MoE
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import _to_local_tensor

try:
    from leto.launch.worker_controller_client import get_faulty_ranks
    _LETO_FAULTY_AVAILABLE = True
except ImportError:
    _LETO_FAULTY_AVAILABLE = False


def _agreement_device(model_parts: list) -> torch.device:
    """Device for the frozen-skip agreement collective.

    Taken from the model itself so the payload lands on the backend's device
    (NCCL needs CUDA); falls back to CPU for gloo/single-process.
    """
    for part in model_parts:
        for param in part.parameters():
            if param.device.type == "cuda":
                return param.device
            break
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


class MoevementCheckpointManager:
    """Trainer-facing manager for checkpoint.method="moevement".

    Mirrors GeminiCheckpointManager's duck-typed surface (two-phase ctor +
    lazy_init for standby compatibility; begin_step/save/maybe_wait_for_staging
    /load/wait_for_tracking/close from the train loop). Unlike gemini it never
    registers FSDP collective callbacks: MoEvement schedules its snapshot
    D2H traffic by Algorithm-1 byte budgeting on a side stream, not by
    interleaving into comm gaps.

    M5 status: sparse capture + windowed commits (M2), load() -> replay
    (sparse->dense conversion, plan §9-M3), upstream-log-fed replay under
    PP (M4), and uniform faulty-rank recovery (M5, plan §3.7): finalized
    windows replicate to the pair rank (rank + world/2) over dedicated gloo
    groups; load() runs a cross-rank window-agreement vote (all_reduce MIN
    of each rank's newest restorable window), then non-faulty ranks restore
    their local dumps while faulty ranks (LETO_FAULTY_RANKS — transient or
    fatal alike) peer-fetch the agreed window from their pair. The trainer's
    normal step loop then re-runs the window's remaining iterations through
    the begin_step / maybe_wait_for_staging / save hooks with zero train.py
    changes, activating each iteration's captured operators until every
    operator is dense again.
    """

    def __init__(
        self,
        dataloader: BaseDataLoader | None,
        states: dict[str, Any],
        checkpoint_config: CheckpointConfig,
        base_folder: str = "",
        **kwargs: Any,
    ) -> None:
        self.enable = checkpoint_config.enable
        self.states = states
        self.states[DATALOADER] = dataloader
        self._checkpoint_config = checkpoint_config
        self._base_folder = base_folder
        # Only used for the policy profile's provenance (model name/flavor).
        # Passed explicitly by init.py; the train_state fallback keeps the
        # older two-arg construction (and the unit tests) working.
        self._job_config = kwargs.get("job_config") or getattr(
            states.get("train_state"), "job_config", None
        )
        self._engine: MoevementSnapshotEngine | None = None
        self._replicator = None  # WindowReplicator (plan §3.7), or None
        self._upstream_logger = None  # constructed only when PP + logging on
        self._pop_group = None  # non-PP "stage column" group under PP
        self._step_t0: float | None = None
        # Replay plan armed by load(): {"S": window_start-1, "last": S+w,
        # "bundle": WindowBundle, "activated": set[str]}. None outside
        # replay; save(S+w) ends it.
        self._replay: dict[str, Any] | None = None
        # Paper §3.3 frozen-skip (see freeze.py). The controller is built in
        # lazy_init; `_frozen_skip` records the configured intent, and
        # `_frozen_skip_live` whether the ARMED replay can actually use it
        # (a window captured without clip-norm anchors cannot).
        self._frozen_skip = bool(
            getattr(checkpoint_config, "moevement_frozen_skip", False)
        )
        self._frozen_skip_live = False
        self._freezer: FrozenSkipController | None = None
        # Last iteration's pre-clip total grad norm, handed over by train.py
        # (record_clip_total_norm) and captured with the iteration's bytes.
        self._pending_clip_norm: torch.Tensor | None = None
        # Warmup-only frozen-skip exercise (compile-cache priming).
        self._frozen_skip_warmup = max(
            0, int(getattr(checkpoint_config, "moevement_frozen_skip_warmup", 0))
        )
        self._warmup_freeze_active = False

        if checkpoint_config.interval != 1:
            # Sparse checkpointing is rolling: a scheduled operator subset is
            # captured every iteration (the paper's Ckpt_interval=1).
            logger.warning(
                "checkpoint.interval=%d is ignored under method=moevement "
                "(sparse checkpointing captures every iteration)",
                checkpoint_config.interval,
            )

    def lazy_init(
        self,
        model_parts: list[nn.Module],
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        parallel_dims: ParallelDims | None = None,
        collective_manager=None,  # accepted for interface parity; unused
    ) -> None:
        self.model_wrapper = ModelWrapper(model_parts)
        self.model_parts = model_parts
        self.optimizers = optimizers
        self.states[LR_SCHEDULER] = lr_schedulers
        self.parallel_dims = parallel_dims

        if not self.enable:
            return
        cfg = self._checkpoint_config
        if not cfg.moevement_mem_fs_folder:
            logger.warning(
                "checkpoint.moevement_mem_fs_folder is not set; moevement "
                "checkpointing stays inert (no snapshots will be captured)"
            )
            return

        # PP wiring (plan §9-M4, pinned P1): stage_ids are the TRUE global
        # virtual-stage ids of model_parts in local-chunk order — the 'loop'
        # mapping pp_rank + s*pp_degree (pipeline_parallel.py:461-481),
        # verified against the ids pipeline_module_split stamped on the parts.
        stage_ids = self._compute_stage_ids(model_parts, parallel_dims)
        # Popularity reduction group (pinned P2): under PP, per-rank count
        # vectors cover only THIS rank's stages' layers and differ in length
        # across pp ranks — reduce over the non-PP "stage column" (ranks
        # sharing this pp_rank) instead of WORLD. None => default group.
        self._pop_group = self._build_popularity_group(parallel_dims)
        self.operators = discover_operators(model_parts, stage_ids)
        self._operators_by_name = {op.name: op for op in self.operators}
        # id(param) -> names of every op claiming (a slice of) it, for
        # replay-time grad masking at the right granularity: grouped expert
        # weights are claimed by several expert ops and may be only
        # partially frozen.
        self._param_claims: dict[int, list[str]] = {}
        for op in self.operators:
            for _fqn, param, _expert_idx in op.param_entries:
                self._param_claims.setdefault(id(param), []).append(op.name)
        self._freezer = FrozenSkipController(
            self.operators,
            model_parts,
            stage_ids,
            enabled=self._frozen_skip,
            # Same stage column as the popularity reduction, and for the same
            # reason: it is the group over which every rank holds the same
            # operator vector. A fully-frozen expert module makes FSDP2 skip
            # that group's reduce-scatter, so the verdict must be world-agreed
            # or ranks desynchronise.
            agree_group=self._pop_group,
            agree_device=_agreement_device(model_parts),
        )
        if self._frozen_skip:
            logger.info(
                "[moevement] frozen-skip (paper §3.3) ENABLED: %s",
                self._freezer.describe(),
            )
        self._pcie_bandwidth_gbs = self._resolve_pcie_bandwidth_gbs(cfg)
        self._scheduler = SparseCheckpointScheduler(
            [op.schedulable() for op in self.operators],
            pcie_bandwidth_gbs=self._pcie_bandwidth_gbs,
            overlap_target=cfg.moevement_snapshot_overlap_target,
            w_sparse_override=cfg.moevement_w_sparse_override,
            reorder_threshold=cfg.moevement_reorder_threshold,
            reorder_fraction=cfg.moevement_reorder_fraction,
        )
        # Wall-clock per-step EMA; alpha = 1/window makes the effective
        # averaging length ~moevement_iter_time_window_iters.
        self._iter_time_ema = cfg.moevement_initial_iter_time_sec
        self._iter_time_alpha = 1.0 / max(1, cfg.moevement_iter_time_window_iters)
        # POPULARITY DECAY: the scheduler accumulates whatever it is fed, so
        # rolling-window semantics live here. The reference keeps an exact
        # W-iteration deque of per-iter counts (scheduler.py:340-444); we use
        # an EMA with per-step decay (W-1)/W — same steady-state rates, O(E)
        # state instead of O(W*E), and needs_reorder keeps firing because
        # totals stop growing without bound. Deliberate approximation.
        window = max(1, cfg.moevement_activation_count_window_iters)
        self._popularity_decay = (window - 1) / window
        self._sched_popularity_mirror: dict[str, float] = {}
        self._last_reduced_flat: torch.Tensor | None = None
        self._global_counts_at_last_order: torch.Tensor | None = None
        self._init_popularity_tracking(model_parts, stage_ids)

        # Cadence decision, once, before anything is sized against it.
        # Precedence (plan §9-M11): explicit override > recorded policy
        # profile whose provenance matches this run > live Algorithm 1
        # (whose verdict is then RECORDED for the next run).
        self._rank = int(os.environ.get("RANK", "0"))
        self._resolve_cadence(cfg)

        # Identical inputs (pinned cadence, zero popularity, name tie-breaks)
        # make the initial schedule world-identical without a collective.
        self._schedule = self._scheduler.regenerate(self._iter_time_ema)
        self._slot_idx = 0

        # Seed the deferred window-start entry with the pre-training
        # position (train step 0 / freshly-built dataloader). M3's load()
        # must reseed this from the restored state before the step loop.
        self._window_start_entry = self._capture_boundary_entry(
            getattr(self.states.get("train_state"), "step", 0)
        )

        mem_fs_folder = cfg.moevement_mem_fs_folder
        self._engine = self._build_engine(mem_fs_folder, self._rank)

        # Window replication + uniform faulty-rank recovery (plan §3.7).
        # Builds three dedicated gloo groups (collective — every rank passes
        # through here) and the init-allocated unpinned remote slots; None
        # when disabled (config off / world unsuitable).
        # The partner is drawn from the FSDP process group by the shared
        # rule (snapshot.partner) — same rule gemini uses: same pipeline
        # stage (identically shaped state) and cross-host wherever the pool
        # allows it.
        self._replicator = build_replicator(
            self._engine,
            mem_fs_folder,
            cfg.moevement_replication,
            pool_ranks=self._partner_pool_ranks(parallel_dims),
        )
        if self._replicator is not None:
            self._engine.attach_replicator(self._replicator)

        # Upstream logger (plan §3.5, pinned P3): constructed ONLY when PP is
        # enabled AND moevement_upstream_logging is on. At PP=1 the module is
        # simply absent — no hooks, no pool, no asserts.
        if (
            parallel_dims is not None
            and parallel_dims.pp_enabled
            and cfg.moevement_upstream_logging
        ):
            from torchtitan.components.moevement.upstream_logger import (
                attach_logger_to_stages,
            )

            self._upstream_logger = self._build_upstream_logger(
                mem_fs_folder, self._rank
            )
            attached = attach_logger_to_stages(
                self._upstream_logger, model_parts, stage_ids
            )
            assert attached == len(model_parts), (
                f"[moevement] upstream logging enabled but only {attached}/"
                f"{len(model_parts)} pipeline stages are tee-instrumented — "
                f"pipeline_module_split must build UpstreamTeePipelineStage "
                f"under checkpoint.method=moevement"
            )
            logger.info(
                "[moevement] upstream logging armed on %d stage(s) %s "
                "(retention %d iterations)",
                attached, stage_ids, self._upstream_logger.capacity,
            )

    def _probe_pcie_bandwidth_gbs(self) -> float:
        """Bandwidth-probe seam (tests substitute a stub)."""
        return measure_d2h_bandwidth_gbs()

    def _resolve_pcie_bandwidth_gbs(self, cfg: CheckpointConfig) -> float:
        """Algorithm 1's B_PCIe input: measured by default, config wins.

        The paper profiles this; the reference hardcodes it. A positive
        ``checkpoint.moevement_pcie_bandwidth_gbs`` is honored verbatim
        (escape hatch / reproducibility); <= 0 means AUTO — profile the
        device once, before the training loop, on the capture stream.
        """
        configured = cfg.moevement_pcie_bandwidth_gbs
        if configured > 0:
            logger.info(
                "[moevement] B_PCIe = %.2f GiB/s (CONFIGURED via "
                "checkpoint.moevement_pcie_bandwidth_gbs; set it <= 0 to "
                "profile the device instead)",
                configured,
            )
            return configured
        t0 = time.perf_counter()
        measured = self._probe_pcie_bandwidth_gbs()
        logger.info(
            "[moevement] B_PCIe = %.2f GiB/s AUTO-PROFILED in %.0f ms "
            "(effective device->host on the snapshot capture stream, into a "
            "pinned host buffer of the engine's kind); this is Algorithm 1's "
            "bandwidth input",
            measured,
            1e3 * (time.perf_counter() - t0),
        )
        return measured

    def _pin_policy_window_size(self) -> tuple[dict[str, Any], int]:
        """Run Algorithm 1 once and pin its (world-aligned) verdict.

        Returns ``(this rank's pre-pin policy report, the pinned w_sparse)``
        so the caller can record both in the policy profile.

        Two reasons the cadence cannot be left free per rank:

        (a) Correctness. Every cross-rank protocol here is iteration-keyed
            (window boundaries, the boundary barrier's collectives, peer
            replication, the window-agreement vote), so a per-rank cadence
            desyncs the world — which is exactly what
            _boundary_should_regenerate's ``w_sparse diverged across ranks``
            assert exists to catch. Rank-local inputs *do* differ: operator
            sets differ across PP stages, the iteration-time EMA is wall
            clock, and B_PCIe is now measured per device. Taking the WORLD
            MAX keeps every rank's per-slot budget at most as tight as its
            own choice would have been (ranks with fewer operators just get
            trailing "rest" slots) — the reference's
            _generate_schedule_world_aligned, same argument.

        (b) Sizing. The engine's window pools are allocated once, against
            scheduler.max_window_bytes(); with a free cadence that bound must
            assume w_sparse == total_ops (~20x the realistic window at 266
            operators, tens of GB of tmpfs per rank).
        """
        proposal = self._scheduler.policy_report(self._iter_time_ema)
        w_local = proposal["proposed_w_sparse"]
        w_world = w_local
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor(
                [w_local], dtype=torch.int64, device=self._collective_device()
            )
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            w_world = int(t[0].item())
        if w_world != w_local:
            logger.info(
                "[moevement] Algorithm 1 proposed w_sparse=%d locally; "
                "aligning to the world MAX w_sparse=%d (operator-count / "
                "iteration-time asymmetry — trailing slots stay empty on "
                "this rank)",
                w_local,
                w_world,
            )
        self._scheduler.pin_window_size(
            w_world,
            reason="Algorithm 1 at init, world-aligned",
            from_policy=True,
        )
        return proposal, w_world

    # -- cadence precedence: override > recorded profile > live policy ------

    def _collective_device(self) -> torch.device:
        return (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    def _world_min(self, value: int) -> int:
        """all_reduce(MIN) of an int; identity when there is no world."""
        if not (dist.is_available() and dist.is_initialized()):
            return int(value)
        t = torch.tensor(
            [int(value)], dtype=torch.int64, device=self._collective_device()
        )
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return int(t[0].item())

    def _policy_provenance(self) -> dict[str, Any]:
        job_config = self._job_config
        model = getattr(job_config, "model", None) if job_config else None
        world_size = (
            self.parallel_dims.world_size
            if self.parallel_dims is not None
            else int(os.environ.get("WORLD_SIZE", "1"))
        )
        return build_provenance(
            getattr(model, "name", ""),
            getattr(model, "flavor", ""),
            self.parallel_dims,
            world_size,
        )

    def _resolve_cadence(self, cfg: CheckpointConfig) -> None:
        """Decide who picks w_sparse, and say so.

        Precedence, highest first:

        1. ``checkpoint.moevement_w_sparse_override`` — explicit, always wins
           (the scheduler is already pinned by it at construction).
        2. A recorded policy profile (``moevement_policy_profile_path``,
           default ``<dump_dir>/moevement_profile/policy.json``) whose
           provenance matches this run's model, parallelism dims and world
           size — the reuse path this repo already uses for progressive
           init's ``solution.json``. ``--checkpoint.moevement_profile_policy``
           forces the profile to be re-derived and rewritten instead.
        3. Live Algorithm 1 (``_pin_policy_window_size``), whose verdict is
           then recorded for the next run.

        The accept/reject decision is made world-uniform by an
        ``all_reduce(MIN)`` before it is acted on: the artifact lives on the
        shared dump_dir, but a partially-visible or per-rank-stale file must
        never leave some ranks on the recorded cadence and others on a live
        one (every cross-rank protocol here is iteration-keyed).
        """
        self._policy_profile_path = (
            cfg.moevement_policy_profile_path
            or default_profile_path(self._base_folder)
        )
        self._cadence_provenance = self._policy_provenance()

        if cfg.moevement_w_sparse_override > 0:
            self._cadence_source = "override"
            logger.info(
                "[moevement] cadence source: OVERRIDE — w_sparse=%d from "
                "checkpoint.moevement_w_sparse_override (no policy profile "
                "read or written); Algorithm 1 is still evaluated and "
                "reported at every schedule regeneration",
                cfg.moevement_w_sparse_override,
            )
            return

        forced = bool(getattr(cfg, "moevement_profile_policy", False))
        if not self._policy_profile_path:
            # No dump_dir and no explicit path: nowhere canonical to record
            # the verdict, so the policy simply runs live every time.
            logger.info(
                "[moevement] policy-profile persistence is DISABLED (no "
                "job.dump_folder and no checkpoint.moevement_policy_profile"
                "_path); Algorithm 1 runs live on every start"
            )
            self._run_live_policy(cfg, record=False)
            return
        profile = None if forced else read_profile(self._policy_profile_path)
        w_recorded, reasons = usable_w_sparse(
            profile,
            self._cadence_provenance,
            self._rank,
            local_total_ops=len(self.operators),
        )
        # World-uniform verdict: reuse only if EVERY rank can reuse it.
        world_ok = bool(self._world_min(1 if w_recorded is not None else 0))

        if world_ok and w_recorded is not None:
            self._cadence_source = "profile"
            self._scheduler.pin_window_size(
                w_recorded,
                reason=f"recorded policy profile {self._policy_profile_path}",
                from_policy=True,
            )
            logger.info(
                "[moevement] cadence source: RECORDED PROFILE — w_sparse=%d "
                "from %s (%s). Algorithm 1 is NOT re-derived; it is still "
                "evaluated every regeneration, so a DRIFTED line means the "
                "recorded cadence no longer matches live conditions "
                "(re-profile with --checkpoint.moevement_profile_policy)",
                w_recorded,
                self._policy_profile_path,
                describe(profile),
            )
            return

        if forced:
            logger.info(
                "[moevement] re-profiling the sparse-checkpointing policy "
                "(--checkpoint.moevement_profile_policy): any artifact at %s "
                "will be overwritten with this run's verdict",
                self._policy_profile_path,
            )
        elif reasons and reasons != ["no recorded profile"]:
            logger.warning(
                "[moevement] IGNORING the recorded policy profile at %s — %s; "
                "falling back to live Algorithm 1 (and rewriting the profile)",
                self._policy_profile_path,
                "; ".join(reasons),
            )
        elif w_recorded is not None:
            logger.warning(
                "[moevement] the recorded policy profile at %s is usable on "
                "this rank but was REJECTED by another rank; falling back to "
                "live Algorithm 1 world-wide",
                self._policy_profile_path,
            )

        self._run_live_policy(cfg, record=True)

    def _run_live_policy(self, cfg: CheckpointConfig, record: bool) -> None:
        """Precedence level 3: derive the cadence now, and say where from."""
        proposal, w_world = self._pin_policy_window_size()
        self._cadence_source = "policy"
        logger.info(
            "[moevement] cadence source: LIVE POLICY (paper Algorithm 1) — "
            "w_sparse=%d (world MAX of the per-rank proposals; this rank "
            "proposed %d from B_PCIe=%.2f GiB/s, T_iter=%.4f s, %d operators)",
            w_world,
            proposal["proposed_w_sparse"],
            proposal["pcie_bandwidth_gbs"],
            proposal["iter_time_sec"],
            proposal["total_ops"],
        )
        if record:
            self._record_policy_profile(cfg, proposal, w_world)

    # Fields of the per-rank policy row, in the order they ride the
    # all_gather tensor. float64 holds every one of these exactly (byte
    # counts stay far below 2**53).
    _POLICY_ROW_FIELDS = (
        "rank",
        "total_ops",
        "pcie_bandwidth_gbs",
        "iter_time_sec",
        "budget_bytes",
        "proposed_w_sparse",
        "proposed_num_active",
        "proposed_worst_slot_bytes",
        "num_active",
        "worst_slot_bytes",
        "max_window_bytes",
    )
    _POLICY_ROW_INTS = frozenset(
        {
            "rank",
            "total_ops",
            "budget_bytes",
            "proposed_w_sparse",
            "proposed_num_active",
            "proposed_worst_slot_bytes",
            "num_active",
            "worst_slot_bytes",
            "max_window_bytes",
        }
    )

    def _gather_policy_rows(self, applied: dict[str, Any]) -> list[dict[str, Any]]:
        """All-gather every rank's policy report.

        Operator counts differ per pipeline stage, so the artifact records a
        rank-keyed dict rather than pretending the numbers are uniform. Only
        w_sparse is world-uniform (that is what the MAX all_reduce buys).
        """
        local = {"rank": self._rank, **applied}
        row = [float(local[name]) for name in self._POLICY_ROW_FIELDS]
        if not (dist.is_available() and dist.is_initialized()):
            return [self._policy_row_to_dict(row)]

        t = torch.tensor(row, dtype=torch.float64, device=self._collective_device())
        gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, t)
        return [self._policy_row_to_dict(g.tolist()) for g in gathered]

    def _policy_row_to_dict(self, row) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, value in zip(self._POLICY_ROW_FIELDS, row):
            out[name] = int(round(value)) if name in self._POLICY_ROW_INTS else value
        return out

    def _record_policy_profile(
        self, cfg: CheckpointConfig, proposal: dict[str, Any], w_world: int
    ) -> None:
        """Persist the live policy verdict for reuse by later runs."""
        applied = self._scheduler.policy_report(self._iter_time_ema)
        # Keep the pre-pin proposal (what THIS rank wanted) alongside the
        # post-pin applied numbers (what the world cadence costs it).
        applied["proposed_w_sparse"] = proposal["proposed_w_sparse"]
        applied["proposed_num_active"] = proposal["proposed_num_active"]
        applied["proposed_worst_slot_bytes"] = proposal["proposed_worst_slot_bytes"]
        rows = self._gather_policy_rows(applied)
        for row in rows:
            row["won_world_max"] = row["proposed_w_sparse"] == int(w_world)

        configured_bw = cfg.moevement_pcie_bandwidth_gbs
        payload = build_payload(
            w_sparse=int(w_world),
            provenance=self._cadence_provenance,
            policy_inputs={
                "overlap_target": cfg.moevement_snapshot_overlap_target,
                "iter_time_sec": self._iter_time_ema,
                # The pin happens in lazy_init, before any step has been
                # timed, so this is the configured SEED, not a measurement —
                # recorded honestly (plan §9-M9 (d)3).
                "iter_time_source": (
                    "seeded (checkpoint.moevement_initial_iter_time_sec; the "
                    "cadence is pinned at init, before any step is timed)"
                ),
                "iter_time_measured_steps": 0,
                "pcie_bandwidth_source": (
                    "configured" if configured_bw > 0 else "auto-profiled"
                ),
                "pcie_bandwidth_gbs_configured": configured_bw,
            },
            rank_rows=rows,
        )
        if self._rank != 0:
            return
        try:
            write_profile(self._policy_profile_path, payload)
        except OSError as exc:
            logger.warning(
                "[moevement] could not record the policy profile at %s: %s "
                "(this run is unaffected; later runs will re-derive it)",
                self._policy_profile_path,
                exc,
            )
            return
        logger.info(
            "[moevement] policy profile written to %s (w_sparse=%d, "
            "%d rank entries, winning_ranks=%s) — later runs of this "
            "workload reuse it instead of re-deriving Algorithm 1",
            self._policy_profile_path,
            payload["w_sparse"],
            len(payload["ranks"]),
            payload["winning_ranks"],
        )

    def _ensure_storage(self) -> None:
        """Deferred pool allocation (M7 structural fix).

        The engine's CURR/PREV window pools and the replicator's remote
        slots allocate HERE — not at lazy_init — so load() can read the
        previous attempt's dump generation into RAM and sweep it from tmpfs
        first: peak tmpfs becomes max(dumps, pools) + bundle-in-RAM instead
        of dumps + pools (which overflowed the 126 GB-/dev/shm host under
        deepseek: SIGBUS at pool prefault mid-vote, collapsing the other
        ranks' vote all_reduce). Idempotent; invoked (a) in load() after
        the read+sweep on EVERY path — restore and fresh-start alike, (b)
        at the first save() as a backstop for runs that never pass through
        load(), (c) at standby-promotion init via notify_rmp_restored — so
        allocation always happens at init-time or behind the engine's
        full-sync fence (M4 pinning caveat). Gloo groups and the container
        still form at lazy_init; only the slot POOLS defer.
        """
        if self._engine is not None:
            self._engine.ensure_storage()
        if self._replicator is not None:
            self._replicator.ensure_slots()

    def _partner_pool_ranks(
        self, parallel_dims: ParallelDims | None
    ) -> list[int] | None:
        """Candidate pool for the replica partner: this rank's FSDP process
        group (global ranks in group-rank order).

        Its members own the same model chunk — same pipeline stage, same TP
        shard column — so their window pools are identically shaped, which
        is what the recovery fetch/serve path needs (plan requirement:
        size-compatible partners). Returns None when there is no FSDP mesh
        (dp_shard*cp == 1): the whole world then becomes the pool, i.e. the
        pre-unification behavior, which is the only way to have ANY replica
        at all in that layout. Under PP that fallback pairs across stages,
        so the pair's pool size may differ — build_replicator's size
        exchange still sizes the remote slots correctly.
        """
        if parallel_dims is None or not (
            dist.is_available() and dist.is_initialized()
        ):
            return None
        mesh = parallel_dims.get_optional_mesh("fsdp")
        if mesh is None:
            logger.warning(
                "[moevement] no FSDP mesh (dp_shard*cp == 1): the replica "
                "partner is drawn from the WHOLE WORLD, which under PP "
                "pairs ranks in different pipeline stages. Enable FSDP "
                "sharding for a same-stage, cross-host partner."
            )
            return None
        return list(dist.get_process_group_ranks(mesh.get_group()))

    def _build_engine(
        self, mem_fs_folder: str, rank: int
    ) -> MoevementSnapshotEngine:
        """Engine construction seam (tests substitute backend/container)."""
        return MoevementSnapshotEngine(
            self._scheduler,
            mem_fs_folder,
            os.path.join(mem_fs_folder, "logs"),
            rank,
        )

    def _build_upstream_logger(self, mem_fs_folder: str, rank: int):
        """Upstream-logger construction seam (tests substitute the backend).
        Retention 2*w_sparse iterations (plan §3.5), fixed at the initial
        schedule; a regen that grows w_sparse past capacity/2 only shortens
        effective retention (warned at the boundary)."""
        from torchtitan.components.moevement.upstream_logger import (
            UpstreamLogger,
        )

        return UpstreamLogger(
            self._engine,
            mem_fs_folder,
            rank,
            retention_iters=2 * self._scheduler.w_sparse,
        )

    def _compute_stage_ids(
        self, model_parts: list[nn.Module], parallel_dims: ParallelDims | None
    ) -> list[int]:
        if parallel_dims is None or not parallel_dims.pp_enabled:
            return [0] * len(model_parts)
        pp_degree = parallel_dims.pp
        pp_rank = parallel_dims.get_mesh("pp").get_local_rank()
        stage_ids = [
            pp_rank + s * pp_degree for s in range(len(model_parts))
        ]
        for idx, part in enumerate(model_parts):
            true_idx = getattr(part, "_pp_virtual_stage_index", None)
            if true_idx is not None and true_idx != stage_ids[idx]:
                raise NotImplementedError(
                    f"model_parts[{idx}] is pipeline stage {true_idx} but "
                    f"the 'loop' mapping gives {stage_ids[idx]} — non-loop "
                    f"(V-style) stage placements are not supported by "
                    f"checkpoint.method=moevement"
                )
        return stage_ids

    def _build_popularity_group(self, parallel_dims: ParallelDims | None):
        """Build (once) the non-PP mesh group: all world ranks sharing this
        pp_rank. Membership is derived from an all_gather of pp local ranks
        rather than mesh-order assumptions; dist.new_group is called for
        every column (collectively required), ours is kept."""
        if (
            parallel_dims is None
            or not parallel_dims.pp_enabled
            or not (dist.is_available() and dist.is_initialized())
        ):
            return None
        my_pp = parallel_dims.get_mesh("pp").get_local_rank()
        world = dist.get_world_size()
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        gathered = torch.zeros(world, dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(
            gathered,
            torch.tensor([my_pp], dtype=torch.int64, device=device),
        )
        pp_of = gathered.cpu().tolist()
        group = None
        for pp_rank in range(parallel_dims.pp):
            members = [r for r in range(world) if pp_of[r] == pp_rank]
            assert members, f"no ranks found for pp_rank {pp_rank}"
            g = dist.new_group(ranks=members)
            if pp_rank == my_pp:
                group = g
        assert group is not None
        return group

    def _init_popularity_tracking(
        self, model_parts: list[nn.Module], stage_ids: list[int]
    ) -> None:
        """Map each MoE layer's (global) tokens_per_expert counts to this
        rank's EP-local expert op names.

        Expert weights are DTensors sharded on dim 0 over the ep (and, with
        2D expert sharding, efsdp) mesh dims; the local shard's global dim-0
        offset — via compute_local_shape_and_global_offset, which handles
        FSDP2's strided sharding of an already-EP-sharded dim — is exactly
        the global expert index of local expert 0.
        """
        self._moe_layers: list[dict[str, Any]] = []
        for model_part, stage in zip(model_parts, stage_ids):
            for moe_fqn, moe in model_part.named_modules():
                if not isinstance(moe, MoE):
                    continue
                label = _layer_label(moe_fqn)
                weight = next(
                    param
                    for _name, param in moe.experts.named_parameters(recurse=False)
                    if param.requires_grad
                )
                if isinstance(weight, DTensor):
                    _local_shape, global_offset = (
                        compute_local_shape_and_global_offset(
                            weight.shape, weight.device_mesh, weight.placements
                        )
                    )
                    dim0_offset = global_offset[0]
                    num_local = weight.to_local().shape[0]
                else:
                    dim0_offset = 0
                    num_local = weight.shape[0]
                names = []
                for idx in range(num_local):
                    name = f"stage{stage}_layer{label}_expert{idx}"
                    assert name in self._operators_by_name, (
                        f"popularity mapping found no operator {name}"
                    )
                    names.append(name)
                counts = _to_local_tensor(moe.tokens_per_expert).detach()
                assert dim0_offset + num_local <= counts.numel(), (
                    f"expert offset {dim0_offset}+{num_local} exceeds "
                    f"tokens_per_expert length {counts.numel()} at {moe_fqn}"
                )
                self._moe_layers.append(
                    {
                        "moe": moe,
                        "names": names,
                        "dim0_offset": dim0_offset,
                        "acc": torch.zeros_like(counts, dtype=torch.float32),
                        # With load balancing on, the optimizer step pre-hook
                        # zeroes tokens_per_expert every step, so the
                        # pre-optimizer read sees exactly this iteration's
                        # counts; without it the buffer only accumulates, so
                        # track deltas against the previous read.
                        "prev": (
                            None
                            if moe.load_balance_coeff is not None
                            else torch.zeros_like(counts, dtype=torch.float32)
                        ),
                    }
                )

    @torch.no_grad()
    def begin_step(self, curr_step: int, last_step: bool = False) -> None:
        self._step_t0 = time.perf_counter()
        self._pending_clip_norm = None
        if self._replay is not None and self._frozen_skip_live:
            # Paper §3.3: freeze not-yet-activated operators BEFORE the
            # forward, so the backward computes their input-gradients only.
            stats = self._freezer.apply(self._replay["activated"])
            logger.info(
                "[moevement] replay step %d frozen-skip: %s",
                curr_step, stats.summary() if stats.applied else "deferred",
            )
        elif self._frozen_skip_warmup:
            self._apply_warmup_freeze(curr_step)
        if self._replay is not None and curr_step > self._replay["S"] + 2:
            # Replay step I consumes the RNG stream the baseline's step I
            # saw = the post-step-(I-1) capture. S+2's stream was already
            # set by load() (RNG(S+1)); later steps restore here.
            restore_rng(self._replay["bundle"].rng[curr_step - 1])
        if self._upstream_logger is not None:
            # Opens the log iteration (and evicts the reused ring slot)
            # before the schedule's tees fire; replay steps never re-log.
            self._upstream_logger.begin_iteration(
                curr_step, replaying=self._replay is not None
            )

    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> None:
        if self._engine is None:
            return
        if self._step_t0 is not None:
            measured = time.perf_counter() - self._step_t0
            self._last_step_wall = measured
            self._iter_time_ema += self._iter_time_alpha * (
                measured - self._iter_time_ema
            )
            self._step_t0 = None
        if self._replay is not None:
            self._replay_save(curr_step)
            return
        if self._warmup_freeze_active:
            # Frozen-skip compile warmup freezes EVERY operator, so no
            # parameter receives a gradient and the optimizer never
            # materializes Adam state — capture would raise "no Adam state
            # found for a scheduled parameter". These steps are throwaway by
            # construction (their numerics are perturbed on purpose); their
            # only job is to land the frozen-skip compiled variants in the
            # caches, so skip capture entirely rather than snapshotting state
            # that no run will ever restore.
            logger.info(
                "[moevement] step %d: frozen-skip warmup active — skipping "
                "capture (warmup steps are not snapshotted)", curr_step,
            )
            return
        # Deferred-allocation backstop (see _ensure_storage): the normal
        # paths allocated inside load()/notify_rmp_restored; a run that
        # reached its first capture without either allocates here, behind
        # the engine's full-sync fence.
        self._ensure_storage()
        if self._upstream_logger is not None:
            # Enqueued BEFORE this step's capture/final records so every log
            # commit precedes its window's final commit in the committer's
            # FIFO — the window-boundary barrier then bounds how long a
            # ring-slot eviction can wait on its commit.
            self._upstream_logger.finish_iteration(curr_step)
        slot = self._schedule[self._slot_idx]
        self._engine.capture_iteration(
            curr_step,
            slot,
            self._operators_by_name,
            self.optimizers,
            # Frozen-skip replay anchor (plan §3.3): captured only when the
            # feature is configured, so every pinned baseline's capture
            # payload and metadata are byte-for-byte unchanged without it.
            clip_total_norm=(
                self._pending_clip_norm if self._frozen_skip else None
            ),
        )
        self._pending_clip_norm = None
        self._slot_idx += 1
        if self._slot_idx >= len(self._schedule):
            self._finish_window(curr_step)

    def _replay_save(self, curr_step: int) -> None:
        """Replay-mode save(I): apply iteration I's captures instead of
        capturing — activate I's ops with their exact fp32+moments (the
        masked step wrote nothing real for them), bf16-refresh the
        still-frozen tail — and end the replay after the window's last
        iteration."""
        rp = self._replay
        assert rp["S"] + 2 <= curr_step <= rp["last"], (
            f"replay save at step {curr_step}, expected "
            f"{rp['S'] + 2}..{rp['last']}"
        )
        # Per-replayed-iteration wall clock, for the §3.3 measurement: the
        # EMA update in save() already measured this step.
        logger.info(
            "[moevement] REPLAY_STEP step=%d wall=%.4fs frozen_skip=%s "
            "frozen_ops=%d frozen_numel=%d",
            curr_step, getattr(self, "_last_step_wall", -1.0),
            self._frozen_skip_live,
            self._freezer.stats.frozen_ops if self._freezer is not None else 0,
            self._freezer.stats.frozen_param_numel
            if self._freezer is not None else 0,
        )
        apply_iteration(
            rp["bundle"],
            curr_step,
            self._operators_by_name,
            self.optimizers,
            rp["activated"],
        )
        if curr_step == rp["last"]:
            # End of replay: state == post-S+w. Pin the RNG to the capture
            # (post-step-S+w == pre-step-S+w+1) and resume normal sparse
            # capture with a fresh window whose START state is exactly now.
            restore_rng(rp["bundle"].rng[curr_step])
            self._verify_all_activated(rp["activated"])
            if self._freezer is not None:
                # Full trainability restored before the first normal step.
                self._freezer.clear()
            self._frozen_skip_live = False
            self._replay = None
            self._slot_idx = 0
            self._window_start_entry = self._capture_boundary_entry(curr_step)
            self._clear_replay_pending()
            if self._upstream_logger is not None:
                # The whole point of receive-side logging: this rank fed its
                # own boundary recvs instead of waiting on a neighbour.
                logger.info(
                    "[moevement] LOG_FED_REPLAY log_fed=%s fill_hits=%d "
                    "fill_misses=%d",
                    self._upstream_logger.replay_active,
                    self._upstream_logger.fill_hits,
                    self._upstream_logger.fill_misses,
                )
                # Drop the reloaded log store; normal logging resumes next
                # step into a fresh ring (retention rebuilds over 2*w_sparse
                # iterations, like the engine's fresh window).
                self._upstream_logger.end_replay()
            logger.info(
                "[moevement] replay complete at step %d; all %d operators "
                "active, sparse capture resumes at step %d",
                curr_step, len(self._operators_by_name), curr_step + 1,
            )

    def _verify_all_activated(self, activated: set[str]) -> None:
        missing = sorted(set(self._operators_by_name) - activated)
        if missing:
            raise RuntimeError(
                f"[moevement] replay finished with {len(missing)} operators "
                f"never activated (bundle does not cover the model): "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

    def _finish_window(self, curr_step: int) -> None:
        # Snapshot the schedule the finalized window actually used, before
        # any regeneration below.
        finished_schedule = [
            {
                "slot": slot.slot_index,
                "active": list(slot.active),
                "frozen": list(slot.frozen),
            }
            for slot in self._schedule
        ]

        totals = self._reduce_popularity()
        if totals:
            # Feed the (world-consistent) rolling totals as deltas so the
            # scheduler's accumulated popularity EQUALS the rolling totals —
            # its ordering and needs_reorder then see rolling-window rates,
            # not an unbounded lifetime sum.
            delta = {
                name: value - self._sched_popularity_mirror.get(name, 0.0)
                for name, value in totals.items()
            }
            self._scheduler.update_popularity(delta)
            self._sched_popularity_mirror = totals
        if self._boundary_should_regenerate():
            self._schedule = self._scheduler.regenerate(self._iter_time_ema)
            if self._last_reduced_flat is not None:
                self._global_counts_at_last_order = (
                    self._last_reduced_flat.clone()
                )
            if (
                self._upstream_logger is not None
                and 2 * self._scheduler.w_sparse
                > self._upstream_logger.capacity
            ):
                logger.warning(
                    "[moevement] schedule regen grew w_sparse to %d but the "
                    "upstream log ring holds %d iterations (< 2*w_sparse); "
                    "log retention is shortened accordingly",
                    self._scheduler.w_sparse, self._upstream_logger.capacity,
                )

        # Window W's metadata must carry W's START state (see
        # finalize_window's contract): pass the entry captured at the
        # previous boundary and defer the entry captured now — the state
        # after curr_step, i.e. the start of window W+1 — to the next
        # finalize.
        start_entry = self._window_start_entry
        self._window_start_entry = self._capture_boundary_entry(curr_step)
        self._engine.finalize_window(
            start_entry["dataloader"],
            start_entry["scalars"],
            lr_scheduler_state=start_entry["lr_scheduler"],
            schedule_snapshot=finished_schedule,
        )
        self._slot_idx = 0

    def _capture_boundary_entry(self, step: int) -> dict[str, Any]:
        dataloader = self.states.get(DATALOADER)
        train_state = self.states.get("train_state")
        lr_schedulers = self.states.get(LR_SCHEDULER)
        scalars: dict[str, Any] = {"step": step}
        ntokens_seen = getattr(train_state, "ntokens_seen", None)
        if ntokens_seen is not None:
            scalars["ntokens_seen"] = ntokens_seen
        return {
            "dataloader": (
                dataloader.state_dict() if dataloader is not None else None
            ),
            "scalars": scalars,
            # Post-step scheduler state (train.py steps the scheduler inside
            # train_step, before save): replay restores it and compensates
            # one step for the pumped iteration S+1.
            "lr_scheduler": (
                lr_schedulers.state_dict() if lr_schedulers is not None else None
            ),
        }

    def _accumulate_popularity(self) -> None:
        decay = self._popularity_decay
        for entry in self._moe_layers:
            counts = _to_local_tensor(entry["moe"].tokens_per_expert).detach()
            if entry["prev"] is None:
                step_counts = counts
            else:
                step_counts = counts - entry["prev"]
                entry["prev"].copy_(counts)
            entry["acc"].mul_(decay).add_(step_counts)

    def _reduce_popularity(self) -> dict[str, float]:
        """All-reduce the rolling per-layer counts (once per window) and map
        them to this rank's EP-local expert op names. The group is the
        non-PP stage column under PP (pinned P2 — flat vectors differ in
        length across pp ranks, so a WORLD reduce would be malformed);
        WORLD/default at PP=1."""
        if not self._moe_layers:
            return {}
        flat = torch.cat(
            [entry["acc"].reshape(-1) for entry in self._moe_layers]
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flat, group=self._pop_group)
        flat = flat.cpu()
        self._last_reduced_flat = flat
        totals: dict[str, float] = {}
        offset = 0
        for entry in self._moe_layers:
            n = entry["acc"].numel()
            layer_counts = flat[offset : offset + n]
            offset += n
            base = entry["dim0_offset"]
            for idx, name in enumerate(entry["names"]):
                totals[name] = float(layer_counts[base + idx])
        return totals

    def _needs_reorder_global(self) -> bool:
        """Column-consistent reorder verdict.

        The scheduler's own needs_reorder sees only this rank's EP-LOCAL
        experts, so its verdict can legitimately differ across ranks — which
        desyncs the collective order at a window boundary (observed as an
        NCCL timeout at boundary 10 of the M2 gate). Applying the same
        eps/threshold/fraction semantics to the all-reduced count vector —
        bit-identical within the reduction group — makes the verdict
        group-identical with no extra collective. Under PP the group is the
        stage column, so verdicts can still differ ACROSS columns;
        _boundary_should_regenerate's unconditional world MAX resolves that
        (pinned P2, replacing the reference's OR-reduce of rank-local
        verdicts, coordinator.py:1113-1120).
        """
        flat = self._last_reduced_flat
        if flat is None or flat.numel() == 0:
            return False
        if self._global_counts_at_last_order is None:
            logger.info(
                "[moevement] reorder trigger FIRED: first window boundary "
                "since the ordering was last built (no baseline counts yet)"
            )
            return True
        old = self._global_counts_at_last_order
        rel = (flat - old).abs() / old.clamp_min(1e-9)
        cfg = self._checkpoint_config
        changed_fraction = (
            (rel > cfg.moevement_reorder_threshold).float().mean().item()
        )
        fired = changed_fraction >= cfg.moevement_reorder_fraction
        # Both outcomes are logged: a schedule that regenerates at EVERY
        # boundary and one that never regenerates look identical from the
        # outside otherwise (semantics unchanged — this is reporting only).
        logger.info(
            "[moevement] reorder trigger %s: %.1f%% of %d expert counters "
            "moved by more than %.0f%% since the last ordering (fires at "
            ">= %.0f%%)",
            "FIRED" if fired else "SUPPRESSED",
            100.0 * changed_fraction,
            flat.numel(),
            100.0 * cfg.moevement_reorder_threshold,
            100.0 * cfg.moevement_reorder_fraction,
        )
        return fired

    def _boundary_should_regenerate(self) -> bool:
        """World-aligned regen decision (pinned P2 boundary alignment).

        At EVERY window boundary, one tiny UNCONDITIONAL world collective:
        all_reduce MAX of the int pair (reorder verdict as 0/1 — OR via MAX
        — and the current w_sparse). Every rank regenerates iff the reduced
        verdict says so, so the collective order stays identical world-wide
        even when columns' verdicts differ; the previous CONDITIONAL
        alignment collective ran only on regenerating ranks and would desync
        under PP. The w_sparse component keeps the old assert's content:
        popularity (reduced) and the tie-broken ordering are group-consistent
        by construction, but the rank-local iter-time EMA could in principle
        diverge w_sparse at a budget boundary without an override — the MAX
        compare turns that into a loud assert instead of silent slot skew
        (e2e pins the override, plan §8-R5). A post-regen divergence is
        caught by the NEXT boundary's pair.
        """
        verdict = 1 if self._needs_reorder_global() else 0
        if not (dist.is_available() and dist.is_initialized()):
            return bool(verdict)
        if self._moe_layers:
            device = self._moe_layers[0]["acc"].device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pair = torch.tensor(
            [verdict, self._scheduler.w_sparse],
            dtype=torch.int64,
            device=device,
        )
        dist.all_reduce(pair, op=dist.ReduceOp.MAX)
        reduced_verdict, reduced_w = int(pair[0].item()), int(pair[1].item())
        assert reduced_w == self._scheduler.w_sparse, (
            f"w_sparse diverged across ranks: local {self._scheduler.w_sparse}"
            f" vs world max {reduced_w}"
        )
        if reduced_verdict != verdict:
            logger.info(
                "[moevement] reorder trigger raised to FIRED by the world "
                "(this rank's local verdict was SUPPRESSED)"
            )
        return bool(reduced_verdict)

    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        """Presence-driven sparse->dense restore (plan §9-M3).

        Picks the newest fully-committed window (iters S+1..S+w, ring =
        state post-S), then: restore dataloader/scalars/lr-scheduler from
        the ring; pump exactly one step's batches through the dataloader
        and discard (positions data before S+2 and advances ntokens_seen
        to post-S+1 — iteration S+1 is applied from the bundle, never
        re-executed); set train_state.step = S+1; materialize optimizer
        state; apply iteration S+1's captures (slot-0 ops exact, the whole
        frozen tail bf16-upcast); restore RNG(S+1); arm the replay plan for
        S+2..S+w. The trainer's step loop (starting at step+1 = S+2) then
        replays through the normal hooks. ``step`` is ignored — like
        gemini, presence of committed state decides.

        Known subtlety: the trainer builds its data iterator after load(),
        and DataLoader iterator creation draws a base_seed from the global
        CPU RNG — so the CPU stream during replayed step S+2 is shifted
        relative to the baseline (whose draw happened at step 1). This is
        invisible under the supported configs: the forward consumes only
        device RNG (untouched by that draw), num_workers=0 means base_seed
        never reaches batch content, and begin_step(I>S+2) / replay-end
        restores re-pin both streams to the captures.
        """
        del step
        if self._engine is None:
            return False
        # Whatever this load() decides supersedes any predecessor's armed
        # replay: the marker is re-armed below iff we arm a replay ourselves.
        # (The RMP-resume vote in train.py has already read it by now — see
        # has_pending_replay.)
        self._clear_replay_pending()
        folder = self._checkpoint_config.moevement_mem_fs_folder
        faulty = get_faulty_ranks() if _LETO_FAULTY_AVAILABLE else []

        if self._replicator is None:
            # Legacy / replication-off path (M3 behavior): every rank —
            # including a faulty one — loads its own local dump.
            if faulty:
                logger.warning(
                    "[moevement] faulty_ranks=%s but replication is disabled; "
                    "falling back to LOCAL restore on every rank", faulty,
                )
            bundle = find_committed_windows(folder, self._rank)
            if bundle is None:
                logger.info(
                    "[moevement] no committed window in %s; starting fresh",
                    folder,
                )
                self._sweep_stale_dumps()
                self._ensure_storage()
                return False
            return self._restore_from_bundle(bundle, fetched=False)

        # --- Uniform faulty-rank recovery (plan §3.7, decision D3) ---
        self_faulty = self._rank in faulty
        pair = self._replicator.pair
        pair_faulty = pair in faulty

        # A faulty rank's own dump is IGNORED BY DESIGN (transient included);
        # its state comes from the pair's remote copy.
        local: dict[int, Any] = {}
        if not self_faulty:
            local = {
                b.window_start: b
                for b in load_usable_windows(folder, self._rank, remote=False)
            }
        remote_starts: list[int] = []
        if pair_faulty and not self_faulty:
            remote_starts = [
                b.window_start
                for b in load_usable_windows(folder, self._rank, remote=True)
            ]

        constraint = vote_constraint(
            own=set(local),
            remote=set(remote_starts),
            self_faulty=self_faulty,
            pair_faulty=pair_faulty,
        )
        agreed = agreed_window(self._replicator.vote(constraint))
        logger.info(
            "[moevement] window-agreement vote: rank %d can supply %s -> "
            "agreed %s (faulty_ranks=%s, self_faulty=%s, pair=%d, "
            "pair_faulty=%s, local=%s, remote=%s)",
            self._rank,
            "ANY" if constraint is None else sorted(constraint),
            f"w{agreed}" if agreed is not None else "FRESH-START",
            faulty, self_faulty, pair, pair_faulty,
            sorted(local), sorted(remote_starts),
        )
        if agreed is None:
            logger.info("[moevement] no world-agreed window; starting fresh")
            # World-consistent decision: nobody restores, nobody fetches —
            # the unusable generation would only leak tmpfs. Sweep BEFORE the
            # deferred pools allocate (M7 tmpfs-peak ordering).
            self._sweep_stale_dumps()
            self._ensure_storage()
            return False

        if self_faulty:
            logger.info(
                "[moevement] rank %d is FAULTY: peer-fetching window w%d "
                "from pair rank %d (local dump ignored by design)",
                self._rank, agreed, pair,
            )
            meta, pool_bytes = self._replicator.fetch_from_pair()
            if int(meta["window_start"]) != agreed:
                raise RuntimeError(
                    f"[moevement] pair served window "
                    f"w{meta['window_start']} but the vote agreed w{agreed}"
                )
            bundle = _build_bundle(f"w{agreed}", meta, pool_bytes)
            return self._restore_from_bundle(bundle, fetched=True)

        if pair_faulty:
            # Serve BEFORE the (potentially slow) local restore so the
            # faulty pair is never left waiting on it.
            self._replicator.serve_remote_window(agreed, self._rank)
        bundle = local.get(agreed)
        if bundle is None:
            raise RuntimeError(
                f"[moevement] rank {self._rank} is not faulty but has no "
                f"usable dump of the agreed window w{agreed} "
                f"(local windows: {sorted(local)}) — window skew across the "
                f"kill; cannot restore consistently"
            )
        # Free the non-agreed bundles' RAM copies before the deferred pools
        # allocate (each is a full window, ~GBs; `bundle` stays referenced).
        local.clear()
        return self._restore_from_bundle(bundle, fetched=False)

    def _sweep_stale_dumps(self) -> None:
        """Delete this rank's consumed dump generation from mem_fs.

        Dump files are window-keyed (``rank_{r}_moevement_w{S}.pt``), so
        successive PERSIST drains across restart attempts accumulate a full
        generation (~GBs/rank) each — until tmpfs starves NCCL's own shm
        segments (observed in the M7 matrix at the 4th restart). By the
        time this runs — vote resolved, any faulty pair already served over
        the wire, restored state re-committed into the fresh container —
        the on-disk generation is fully superseded: a later kill re-dumps
        the live ledger, and a crash before that still drains the
        re-committed window from the container.
        """
        folder = self._checkpoint_config.moevement_mem_fs_folder
        stale = glob.glob(
            os.path.join(folder, f"rank_{self._rank}_moevement_*")
        )
        for path in stale:
            try:
                os.unlink(path)
            except OSError:
                pass
        if stale:
            logger.info(
                "[moevement] swept %d consumed dump file(s) for rank %d",
                len(stale), self._rank,
            )

    @torch.no_grad()
    def _restore_from_bundle(self, bundle, fetched: bool) -> bool:
        """Restore + arm replay from one WindowBundle (local dump or
        peer-fetched). ``fetched`` marks the faulty-rank path: upstream-log
        rings are NOT replicated, so a fetched restore skips the log reload
        and replays boundary recvs over live p2p (plan §3.7)."""
        start, last = bundle.window_start, bundle.last_step  # S+1, S+w
        train_state = self.states.get("train_state")
        assert train_state is not None, "load() requires a train_state"

        ring = bundle.ring
        dataloader = self.states.get(DATALOADER)
        if dataloader is not None and ring.get("dataloader") is not None:
            dataloader.load_state_dict(ring["dataloader"])
        scalars = ring.get("train_state") or {}
        if "ntokens_seen" in scalars and hasattr(train_state, "ntokens_seen"):
            train_state.ntokens_seen = scalars["ntokens_seen"]
        lr_schedulers = self.states.get(LR_SCHEDULER)
        if lr_schedulers is not None and ring.get("lr_scheduler") is not None:
            lr_schedulers.load_state_dict(ring["lr_scheduler"])
            # The ring holds the post-S scheduler state; the loop resumes at
            # S+2, so advance once for the pumped iteration S+1 (the same
            # compensation the RMP resume paths apply, train.py:1071-1079).
            lr_schedulers.step()

        self._pump_one_step(dataloader, train_state)
        train_state.step = start

        # The optimizer has not stepped in this process; Adam state must
        # exist before captures can be copied into it in place (the same
        # zero-grad materialization OptimizersContainer.state_dict relies
        # on). No-op when state already exists.
        for opt in self.optimizers:
            _init_optim_state(opt)

        activated: set[str] = set()
        apply_iteration(
            bundle, start, self._operators_by_name, self.optimizers, activated
        )
        restore_rng(bundle.rng[start])

        # Fresh-capture bookkeeping (M2 hand-off notes): moment-state cache
        # is repopulated against the restored optimizer, the slot cursor
        # restarts, and the window-start ring reseeds from the restored
        # boundary (overwritten again at replay end for multi-iter windows).
        self._engine._moment_cache.clear()
        self._slot_idx = 0
        self._window_start_entry = self._capture_boundary_entry(start)

        # The upstream-log ring is dumped as rank_{r}_moevement_logs.pt —
        # which _sweep_stale_dumps' rank_{r}_moevement_* glob matches — so it
        # MUST be read into RAM before the sweep below, exactly like the
        # bundle. (Regression note: the sweep was moved ahead of the pool
        # allocation in M7 for the tmpfs-peak fix, which silently started
        # deleting the log dump before the reload; invisible until
        # receive-side keying made the reload actually matter. Observed live
        # as "upstream-log replay store: 0 keys".)
        log_store: dict = {}
        if self._upstream_logger is not None and start < last and not fetched:
            from torchtitan.components.moevement.upstream_logger import (
                load_persisted_logs,
            )

            log_store = load_persisted_logs(
                self._checkpoint_config.moevement_mem_fs_folder, self._rank
            )

        # M7 structural ordering: the consumed dump generation is fully in
        # process RAM by now (torch.load materializes into RAM — no
        # mmap/lazy references back to the files; a peer-fetched bundle
        # arrived over the wire), and any faulty pair has already been
        # served (serve reads the dump FILE and runs before this method).
        # So sweep the generation from tmpfs BEFORE the deferred pools
        # allocate: peak tmpfs = max(dumps, pools) + bundle-in-RAM, never
        # dumps + pools (the 126 GB-host overflow). Trade-off, accepted by
        # design: a kill between this sweep and recommit_window's commit
        # fresh-starts (the old post-recommit sweep had the same window
        # shifted later; both are narrow).
        self._sweep_stale_dumps()
        self._ensure_storage()

        # M6 hardening (plan §9-M5 carry-forward (a)): re-commit the
        # restored window into the fresh pools/container — and re-replicate
        # it to the pair — so a SECOND fault before the first fresh
        # finalize recovers from this window again instead of
        # fresh-starting. The double-buffer protocol drops the copy only
        # once a fresh finalized window is durable (see
        # engine.recommit_window).
        self._engine.recommit_window(
            bundle.raw_meta, bundle.pool_bytes,
            window_used_nbytes(bundle.raw_meta),
        )

        if start < last:
            self._replay = {
                "S": start - 1,
                "last": last,
                "bundle": bundle,
                "activated": activated,
            }
            self._arm_frozen_skip(bundle, start, last)
            # Cross-process durability marker (see has_pending_replay): from
            # here until save(last) this rank's live params/optimizer state
            # are a sparse-replay INTERMEDIATE, not a dense training state,
            # so no RMP-sourced resume may adopt them.
            self._set_replay_pending(last)
            if self._upstream_logger is not None:
                self._arm_log_fed_replay(start, last, fetched, log_store)
        else:  # w == 1: the whole window is applied here; no replay steps
            self._verify_all_activated(activated)
        logger.info(
            "[moevement] restored window %s: train resumes at step %d, "
            "replaying %d iteration(s) through step %d",
            bundle.key, start + 1, max(0, last - start), last,
        )
        return True

    # ------------------------------------------------------------------
    # Log-fed replay arming (plan §3.5; receive-side logging)
    # ------------------------------------------------------------------

    def _arm_log_fed_replay(
        self, start: int, last: int, fetched: bool, store: dict
    ) -> None:
        """Settle the world's log-fed vector over this rank's own
        RECEIVE-side upstream logs (already read into RAM by the caller,
        before the dump sweep).

        Receive-side logging makes a rank self-sufficient: the boundary
        tensors it needs during replay are the ones IT received, so they are
        in its own ring — no survivor and no cross-rank exchange required.
        Two ranks can still disagree about whether they have them (a faulty
        rank peer-fetched its window and its logs died with its mem_fs —
        logs are deliberately NOT replicated, plan §3.7), and a skipped recv
        must never leave an unmatched send, so the per-rank bit is
        all-gathered and BOTH ends of every boundary read the same vector:
        a transfer happens iff its receiver is not log-fed.

        Collective safety: every rank reaches this point together — the
        window vote is world-agreed, `start < last` is a function of the
        world-uniform w_sparse, and the logger exists on all ranks or none
        (PP + config gated).
        """
        if fetched:
            # Faulty rank: its mem_fs (fatal) or its dump (ignored by design)
            # holds nothing usable. It replays over live p2p.
            logger.info(
                "[moevement] fetched restore: no upstream-log reload (logs "
                "are not replicated); this rank replays over live p2p"
            )
        # The trainer replays steps start+1 .. last (start itself is applied
        # from the bundle, never re-executed).
        required = range(start + 1, last + 1)
        self_sufficient = self._upstream_logger.arm_replay(
            store, required_iterations=required
        )
        flags = self._all_gather_flag(self_sufficient)
        log_fed = [r for r, on in enumerate(flags) if on]
        self._upstream_logger.set_log_fed_ranks(log_fed)
        logger.info(
            "[moevement] upstream-log replay store: %d keys over iterations "
            "%s; needs %s -> self_sufficient=%s; world log-fed ranks %s "
            "(%d/%d)",
            len(store),
            sorted({k[0] for k in store}) if store else [],
            list(required),
            self_sufficient,
            log_fed,
            len(log_fed),
            len(flags),
        )
        if not self_sufficient and store:
            logger.warning(
                "[moevement] rank %d holds upstream logs but not for every "
                "replayed iteration (missing %s) — falling back to live p2p "
                "on this rank",
                self._rank, self._upstream_logger.missing_iterations,
            )

    def _all_gather_flag(self, flag: bool) -> list[bool]:
        """All-gather one boolean per GLOBAL rank (identity without a world)."""
        if not (dist.is_available() and dist.is_initialized()):
            return [bool(flag)]
        world = dist.get_world_size()
        out = torch.zeros(
            world, dtype=torch.int64, device=self._collective_device()
        )
        src = torch.tensor(
            [1 if flag else 0],
            dtype=torch.int64,
            device=self._collective_device(),
        )
        dist.all_gather_into_tensor(out, src)
        return [bool(v) for v in out.tolist()]

    # ------------------------------------------------------------------
    # Replay-pending marker (plan §3.9; 2026-07 gptoss_tp2fsdp4ep4 M8 bug)
    # ------------------------------------------------------------------
    #
    # A restore arms a replay that spans w_sparse-1 REAL training steps.
    # During those steps the live (and therefore RMP-persisted) state is a
    # sparse-replay intermediate: not-yet-activated operators hold their
    # bf16-upcast compute copy and *unrestored* Adam moments (zeroed by
    # _init_optim_state, then written by the unmasked replay step under the
    # resilient optimizer). That is bit-exact for the FORWARD (the bf16 cast
    # is idempotent) but WRONG for the optimizer — which is why a promoted
    # standby that adopts it trains the backbone from near-zero moments and
    # diverges one step later, with every replayed loss before it still
    # bit-identical (the M8 signature: replay steps match, the first step
    # after the promotion does not).
    #
    # The marker lives in mem_fs (tmpfs): it survives the training process's
    # death and is visible to the co-located standby at the same rank, and a
    # fatal restart clears mem_fs anyway (that path goes through load()).
    # Deliberately NOT named rank_{r}_moevement_* so _sweep_stale_dumps
    # cannot take it out from under the vote.

    def _replay_pending_path(self) -> str | None:
        folder = self._checkpoint_config.moevement_mem_fs_folder
        rank = getattr(self, "_rank", None)
        if not folder or rank is None:
            return None
        return os.path.join(folder, f"rank_{rank}_replay_pending")

    def _set_replay_pending(self, last_step: int) -> None:
        path = self._replay_pending_path()
        if path is None:
            return
        try:
            with open(path, "w") as f:
                f.write(str(last_step))
        except OSError as e:
            logger.warning(
                "[moevement] could not write the replay-pending marker %s: "
                "%s — a transient fault during this replay would resume from "
                "sparse-intermediate RMP state", path, e,
            )

    def _clear_replay_pending(self) -> None:
        path = self._replay_pending_path()
        if path is None:
            return
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning(
                "[moevement] could not clear the replay-pending marker %s: %s",
                path, e,
            )

    def has_pending_replay(self) -> bool:
        """True when THIS rank's mem_fs says a replay was armed and never
        finished — i.e. the RMP-backed params/optimizer state a promoted
        standby would inherit are a sparse-replay intermediate.

        train.py votes this down into the rmp resume decision (duck-typed;
        gemini defines no such hook), so the world falls back to
        checkpointer.load() — which either restores the window and replays
        it from the top, or fresh-starts — instead of silently adopting
        half-restored state.
        """
        path = self._replay_pending_path()
        return path is not None and os.path.exists(path)

    @torch.no_grad()
    def notify_rmp_restored(self, step: int) -> None:
        """Full-leto transient path (plan §3.9): the trainer calls this
        AFTER a successful RMP resilient-optimizer recovery, which bypassed
        load() entirely — moevement state is NOT consulted (the promoted
        standby's containers were killed without dump, by design). The
        RMP metadata restore has just repositioned the dataloader /
        lr-schedulers / train-state scalars to post-``step``, so reseed the
        deferred window-start ring from the RESTORED state (the entry
        captured at lazy_init holds the pre-training step-0 state and
        would corrupt the first finalized window's ring) and start a fresh
        sparse window.

        R6 (plan §8): fatal-fault coverage resumes at this group's FIRST
        finalized (+replicated) window — a fatal arriving before that
        finds no committed moevement window and falls back to RMP
        metadata / fresh start (documented property, logged loudly here).
        A dense re-commit of the live state at promotion time (the
        analogue of the post-restore re-commit) would require a full
        synchronous capture of every operator on the recovery critical
        path — deliberately not done (documented trade-off).
        """
        if self._engine is None:
            return
        assert self._replay is None, (
            "notify_rmp_restored must not race an armed replay — RMP "
            "recovery and load() are mutually exclusive per attempt"
        )
        # Defense in depth behind train.py's vote gate: reaching here with a
        # predecessor's marker still set means the RMP state being adopted is
        # a sparse-replay intermediate. Fail loudly rather than train on it.
        if self.has_pending_replay():
            raise RuntimeError(
                "[moevement] RMP resume at step "
                f"{step} would adopt a predecessor's UNFINISHED replay "
                f"(marker {self._replay_pending_path()}): its not-yet-"
                "activated operators hold bf16 compute copies and unrestored "
                "Adam moments. The resume vote must route this attempt "
                "through checkpointer.load() instead (train.py has_pending_"
                "replay gate)."
            )
        # Deferred pools (M7): the promoted standby allocates HERE — its
        # promotion-init time — preserving the ACTIVATE-time allocation
        # guarantee the M2c/M5 status blocks pin (allocs at init, never
        # unfenced mid-training). Any superseded dump residue is swept
        # first, mirroring load()'s sweep-before-allocate ordering (this
        # path never consults dumps by design, so they are pure tmpfs
        # ballast here).
        self._sweep_stale_dumps()
        self._ensure_storage()
        self._engine._moment_cache.clear()
        self._slot_idx = 0
        self._window_start_entry = self._capture_boundary_entry(step)
        logger.info(
            "[moevement] RMP recovery restored step %d; moevement state "
            "not consulted (transient full-leto path). Fresh sparse window "
            "begins at step %d; fatal-fault coverage resumes at this "
            "group's first finalized+replicated window (plan §8-R6).",
            step, step + 1,
        )

    def _pump_one_step(self, dataloader, train_state) -> None:
        """Consume exactly one training step's global batches and discard
        them, with the trainer batch_generator's state side effects
        (train.py:551-580): the dataloader advances past iteration S+1 and
        ntokens_seen advances to post-S+1."""
        if dataloader is None:
            return
        gas = int(getattr(train_state, "gradient_accumulation_steps", 1))
        iterator = iter(dataloader)
        for _ in range(gas):
            _input_dict, labels = next(iterator)
            if hasattr(train_state, "ntokens_seen"):
                train_state.ntokens_seen += labels.numel()
        # A StatefulDataLoader consumes its loaded resume state on the
        # iter() above; re-arm with the post-pump position so the trainer's
        # own batch_generator (built after load()) resumes at S+2's batches.
        dataloader.load_state_dict(dataloader.state_dict())

    # ------------------------------------------------------------------
    # Paper §3.3: frozen operators do forward + input-gradient only
    # ------------------------------------------------------------------

    def _arm_frozen_skip(self, bundle, start: int, last: int) -> None:
        """Decide whether the armed replay may skip frozen weight-gradients.

        Gate: every replayed step (start+1 .. last) must carry a captured
        pre-clip total gradient norm. Skipping the frozen wgrads shrinks the
        gradient set ``training.max_norm`` is computed over, which would
        change the clip coefficient — and therefore the ALREADY-ACTIVATED
        operators' updates — so the coefficient is pinned to the original
        run's norm instead. A window captured by an older build (or with the
        feature off) has no anchors; the replay then falls back to the exact
        full-backward + grad-masking path, loudly.
        """
        self._frozen_skip_live = False
        if not self._frozen_skip or self._freezer is None:
            return
        needed = list(range(start + 1, last + 1))
        # A total gradient norm of exactly 0 (or a non-finite one) never
        # occurs in real training: it means the anchor never made it into
        # the pool. Treat it as missing rather than pinning a coefficient of
        # 1.0 and silently un-clipping the replayed steps — the exact
        # failure a capture-stream lifetime bug produced once.
        def _bad(step: int) -> bool:
            anchor = bundle.clip_norms.get(step)
            if anchor is None:
                return True
            value = anchor[0]
            return not (value > 0.0) or value != value or value == float("inf")

        missing = [step for step in needed if _bad(step)]
        if missing:
            logger.warning(
                "[moevement] frozen-skip requested but window %s has no "
                "usable clip-norm anchor for step(s) %s (captured without "
                "checkpoint.moevement_frozen_skip, or the anchor was lost); "
                "replaying with the exact full-backward + grad-masking path "
                "instead",
                bundle.key, missing[:4],
            )
            return
        self._frozen_skip_live = True
        self._freezer.arm(self.model_parts)
        logger.info(
            "[moevement] frozen-skip ARMED for replay steps %d..%d "
            "(clip norms pinned from the captured window)",
            start + 1, last,
        )

    def replay_clip_total_norm(self, step: int) -> torch.Tensor | None:
        """train.py hook (duck-typed): the pre-clip total gradient norm this
        step must use, or None to compute it normally.

        Non-None only while a frozen-skip replay is live, where the live
        gradient set is deliberately smaller than the original run's."""
        if not self._frozen_skip_live or self._replay is None:
            return None
        anchor = self._replay["bundle"].clip_norms.get(step)
        if anchor is None:
            return None
        value, dtype_name = anchor
        dtype = getattr(torch, dtype_name, torch.float32)
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        return torch.tensor(value, dtype=dtype, device=device)

    def record_clip_total_norm(self, grad_norm) -> None:
        """train.py hook (duck-typed): stash this step's pre-clip total
        gradient norm so save() can capture it alongside the weight bytes.

        Kept as the live device tensor — the D2H rides the capture stream
        and is read by the committer after its event, so the trainer never
        synchronizes."""
        if not self._frozen_skip or self._replay is not None:
            return
        if isinstance(grad_norm, torch.Tensor):
            # Normally already a plain tensor (the reducers full_tensor() it);
            # _to_local_tensor keeps a DTensor-shaped corner case D2H-able.
            self._pending_clip_norm = _to_local_tensor(grad_norm.detach())

    def _resilient_opt_active(self) -> bool:
        # train.py assigns the Trainer itself as states["train_state"], so
        # this reaches Trainer._resilient_opt (see train.py:154).
        return (
            getattr(self.states.get("train_state"), "_resilient_opt", None)
            is not None
        )

    def _fill_frozen_grads_for_resilient(self) -> None:
        """Resilient-optimizer compatibility for the frozen skip.

        ``ResilientOptimizer`` precomputes a chunk schedule over EVERY
        parameter that has optimizer state and re-resolves ``param.grad`` by
        reference inside each chunk (resilient_opt.py ``_step_chunk``), so a
        ``None`` grad is fatal there — which is exactly why the old masking
        path was skipped under RMP (plan §9-M6 claim (3)). The wgrad skip
        removes the gradient at its source, so instead of masking we
        materialise a ZERO gradient for the skipped parameters just before
        the step. The FLOP saving (the wgrad matmuls and the FSDP2
        reduce-scatter) is fully preserved — only the optimizer's own
        elementwise work is paid — and the resulting update is the same
        harmless, later-overwritten one the unmasked RMP path already
        performed: every still-frozen operator is bf16-refreshed by
        ``_replay_save`` on the same iteration and gets exact moments at its
        activation."""
        if self._freezer is None:
            return
        for param in self._freezer.frozen_params_without_grad():
            param.grad = torch.zeros_like(param)

    def _apply_warmup_freeze(self, curr_step: int) -> None:
        """Throwaway-warmup only: cycle the frozen-skip graph variants during
        the first ``moevement_frozen_skip_warmup`` steps so torch.compile /
        inductor caches every variant a real replay will need, and a replay's
        first step pays a cache lookup instead of a fresh compile.

        Perturbs those steps' numerics by construction (frozen operators do
        not update), so it is gated behind its own config knob and belongs
        only in warmup / profiling jobs."""
        if self._freezer is None or not self._freezer.enabled:
            return
        if curr_step > self._frozen_skip_warmup:
            if self._warmup_freeze_active:
                self._freezer.clear()
                self._warmup_freeze_active = False
                logger.info(
                    "[moevement] frozen-skip compile warmup finished at step "
                    "%d; full trainability restored", curr_step,
                )
            return
        if not self._warmup_freeze_active:
            self._freezer.arm(self.model_parts)
            self._warmup_freeze_active = True
        # ALTERNATE two activation patterns, because one does not reach both
        # graph variants:
        #   odd steps  — activate one expert operator per part. Safety rule 1
        #                then does NOT fire, so `non_expert` and every `gate`
        #                really do go requires_grad=False: this is what warms
        #                the compiled dense submodules' frozen variant, which
        #                is the one a real replay uses (a replay's experts are
        #                activated first by the popularity ordering).
        #   even steps — activate nothing. Rule 1 keeps each part's backbone
        #                trainable, and every expert module is fully frozen:
        #                this warms the detached grouped-mm variant.
        # `moevement_frozen_skip_warmup >= 2` therefore covers both.
        if curr_step % 2 == 1:
            activated = {
                op.name
                for stage in {o.stage for o in self.operators}
                for op in [
                    next(
                        (
                            o
                            for o in self.operators
                            if o.stage == stage
                            and o.kind is OperatorKind.EXPERT
                        ),
                        None,
                    )
                ]
                if op is not None
            }
        else:
            activated = set()
        stats = self._freezer.apply(activated)
        logger.info(
            "[moevement] frozen-skip compile warmup step %d/%d "
            "(pattern=%s): %s",
            curr_step, self._frozen_skip_warmup,
            "expert-active" if curr_step % 2 == 1 else "all-frozen",
            stats.summary() if stats.applied else "deferred",
        )

    def _mask_frozen_grads(self) -> None:
        """Replay mode: stop optimizer updates for not-yet-activated
        operators by dropping their grads to None (plan §3.6 frozen-op
        mechanics). Runs after grad clipping by call position, so the clip
        norm was computed over the full, bit-matching grad set.

        Granularity: a param's grad is dropped only when EVERY op claiming
        it is still frozen. A grouped expert weight with any activated
        expert slice keeps its grad so the active slices update live; the
        frozen slices' (wasted) updates are overwritten by save()'s
        bf16-refresh / activation applies, and Adam is elementwise, so the
        active slices are unaffected.

        Under the RMP resilient optimizer (full-leto fatal recovery, plan
        §3.9) masking is SKIPPED entirely: the chunked step resolves
        param.grad by reference every step (a None would crash
        _step_chunk) and reads the RMP-persisted RS output buffers, so a
        None'd param.grad cannot stop the update anyway. Unmasked frozen
        updates are invisible by the same argument that makes the
        per-op-claims partial coverage safe: every still-frozen op's
        params are refreshed by _replay_save each replayed iteration, and
        its moments + Adam step are exactly restored at activation."""
        if self._resilient_opt_active():
            return
        activated = self._replay["activated"]
        for op in self.operators:
            if op.name in activated:
                continue
            for _fqn, param, _expert_idx in op.param_entries:
                claims = self._param_claims[id(param)]
                if all(name not in activated for name in claims):
                    param.grad = None

    def maybe_wait_for_staging(self) -> None:
        if self._engine is None:
            return
        if self._replay is not None:
            # No capture is in flight during replay; this slot instead masks
            # frozen grads (post-clip by call position). Popularity still
            # accumulates: the replayed counts bit-match the baseline's.
            # Under frozen-skip most of these grads were never computed
            # (paper §3.3); masking still covers what the skip could NOT —
            # partially-frozen grouped expert tensors and the parts kept
            # trainable by safety rule 1.
            self._mask_frozen_grads()
            if self._frozen_skip_live and self._resilient_opt_active():
                # ...and then the chunked step's per-parameter grad
                # requirement is satisfied with zeros (see the method).
                self._fill_frozen_grads_for_resilient()
            self._accumulate_popularity()
            return
        if self._warmup_freeze_active and self._resilient_opt_active():
            # Compile-warmup steps freeze operators outside a replay; the
            # resilient optimizer still needs every parameter's grad.
            self._fill_frozen_grads_for_resilient()
        # GPU-side wait on the last capture's D2H event: only optimizer.step
        # (the next writer of the captured tensors) orders after the drain —
        # no CPU sync (reference sparse_snapshot.py:373-434).
        self._engine.wait_pending_capture()
        # Read tokens_per_expert pre-optimizer, before the load-balancing
        # step pre-hook zeroes the buffers (plan §3.3).
        self._accumulate_popularity()

    def wait_for_tracking(self) -> None:
        return

    def close(self):
        if self._freezer is not None:
            self._freezer.clear()
        if self._replicator is not None:
            self._replicator.close()
            self._replicator = None
        if self._engine is not None:
            self._engine.close()
            self._engine = None
