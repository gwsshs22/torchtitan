"""Sparse-to-dense conversion for MoEvement (plan §3.6, §9-M3).

Reconstructs a replayable window from the container's PERSIST dumps
(`rank_{r}_moevement_{key}.pt` + `rank_{r}_moevement_metadata.json`, written
by MoevementDumpPolicy) and applies per-iteration captures IN PLACE into the
live model/optimizer:

- ``find_committed_windows`` picks the newest window whose full iteration
  set is committed (final flag + ring entry + contiguous iters + in-bounds
  headers), falling back to the other pool's window; ``None`` when nothing
  is usable (recovery then falls back to a fresh start — the documented
  pre-first-finalize property).
- ``WindowBundle`` exposes zero-copy typed tensor views over the dumped
  pool bytes via the committed headers (gemini's
  ``_reconstruct_tensors_from_pool`` pattern, in_mem_state.py), plus the
  per-iter RNG blobs, the window-START ring entry, and the schedule
  snapshot.
- ``apply_iteration`` restores one iteration: ops captured ACTIVE get their
  exact fp32 shard slice + Adam exp_avg/exp_avg_sq/step copied in place
  (dim-0 slices for expert entries; moments resolved with the engine's
  shared ``find_adam_state``); ops captured FROZEN get their bf16 capture
  upcast in place into the fp32 shard (exact round-trip under the bf16
  compute contract, plan §1.2); fp32-consumed buffers (expert_bias) restore
  at full fp32 in both cases. All copies preserve tensor identity
  (``copy_`` into the live storage), like gemini's load_state_dict.

The replay driver that sequences these applications lives in
MoevementCheckpointManager (checkpoint.py).
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from torchtitan.components.moevement.operators import Operator
from torchtitan.components.moevement.snapshot_engine import (
    _local_slice,
    find_adam_state,
    window_used_nbytes,  # noqa: F401  (re-export: lives in snapshot_engine
    # so the container-subprocess dump policy can compute used prefixes;
    # checkpoint.py / replication.py / tests keep importing it from here)
)

logger = logging.getLogger(__name__)


@dataclass
class WindowBundle:
    """One committed window, reconstructed from a container dump."""

    key: str
    window_start: int  # S+1: the window's first captured step
    steps: list[int]  # S+1 .. S+w, contiguous
    # step -> {op_name: {"is_active": bool, "tensors": {key: layout}}}
    headers: dict[int, dict[str, Any]]
    # step -> op_name -> tensor key -> CPU view into the dumped pool bytes
    tensors: dict[int, dict[str, dict[str, torch.Tensor]]]
    # step -> {"torch_cpu": ByteTensor, ["torch_cuda": ByteTensor]}
    rng: dict[int, dict[str, torch.Tensor]]
    # Window-START state: {"dataloader": ..., "train_state": {...},
    # "lr_scheduler": ...} — post-S, i.e. before the window's first step.
    ring: dict[str, Any]
    schedule: Any = None
    # Keeps the dumped pool bytes (the storage every view aliases) alive.
    pool_bytes: torch.Tensor = field(default=None, repr=False)
    # The dump's full metadata dict, retained verbatim so a restored window
    # can be RE-COMMITTED into the fresh pools/container (M6 hardening —
    # a second fault before the first fresh finalize then recovers from
    # this window instead of fresh-starting).
    raw_meta: dict = field(default=None, repr=False)

    @property
    def last_step(self) -> int:
        return self.steps[-1]


def _view(pool_storage, entry: dict[str, Any]) -> torch.Tensor:
    """Typed zero-copy view over the pool bytes for one header entry
    (mirrors gemini in_mem_state._reconstruct_tensors_from_pool; offsets
    here are bytes, converted to element offsets for set_)."""
    dtype = getattr(torch, entry["dtype"])
    elem = torch.empty(0, dtype=dtype).element_size()
    offset = entry["offset"]
    if offset % elem != 0:
        raise ValueError(
            f"pool offset {offset} is not aligned for dtype {entry['dtype']}"
        )
    tensor = torch.empty(0, dtype=dtype)
    tensor.set_(
        source=pool_storage,
        storage_offset=offset // elem,
        size=tuple(entry["shape"]),
    )
    return tensor


def _build_bundle(key: str, meta: dict[str, Any], pool_bytes) -> WindowBundle:
    """Validate one dumped window's completeness and reconstruct its views.

    Raises ValueError on any incompleteness — the caller treats that window
    as unusable and falls back to the other one.
    """
    if not meta.get("final"):
        raise ValueError(f"window {key} was never finalized (no final commit)")
    ring = meta.get("ring")
    if not isinstance(ring, dict):
        raise ValueError(f"window {key} has no ring entry")
    iters = meta.get("iters") or []
    if not iters:
        raise ValueError(f"window {key} has no committed iterations")
    window_start = meta["window_start"]
    steps = [entry["step"] for entry in iters]
    if steps != list(range(window_start, window_start + len(steps))):
        raise ValueError(
            f"window {key} iterations are not contiguous from "
            f"{window_start}: {steps}"
        )
    pool_nbytes = pool_bytes.numel()
    pool_storage = pool_bytes.untyped_storage()

    headers: dict[int, dict[str, Any]] = {}
    tensors: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    rng: dict[int, dict[str, torch.Tensor]] = {}
    for entry in iters:
        step = entry["step"]
        header = entry.get("header")
        step_rng = entry.get("rng")
        if header is None or not step_rng:
            raise ValueError(f"window {key} iter {step} lacks header/rng")
        step_tensors: dict[str, dict[str, torch.Tensor]] = {}
        for op_name, op_meta in header.items():
            op_views: dict[str, torch.Tensor] = {}
            for tkey, tmeta in op_meta["tensors"].items():
                if tmeta["offset"] + tmeta["nbytes"] > pool_nbytes:
                    raise ValueError(
                        f"window {key} iter {step} op {op_name} entry {tkey} "
                        f"exceeds the dumped pool "
                        f"({tmeta['offset']}+{tmeta['nbytes']} > {pool_nbytes})"
                    )
                op_views[tkey] = _view(pool_storage, tmeta)
            step_tensors[op_name] = op_views
        headers[step] = header
        tensors[step] = step_tensors
        rng[step] = step_rng
    return WindowBundle(
        key=key,
        window_start=window_start,
        steps=steps,
        headers=headers,
        tensors=tensors,
        rng=rng,
        ring=ring,
        schedule=meta.get("schedule"),
        pool_bytes=pool_bytes,
        raw_meta=meta,
    )


def load_usable_windows(
    mem_fs_folder: str, rank: int, remote: bool = False
) -> list[WindowBundle]:
    """Load this rank's dump and return every usable window, sorted by
    window_start (oldest first).

    The metadata index (written LAST by the dump policy) names the candidate
    keys; each window's .pt is torch.load'ed and validated for completeness
    (finalized + ring + contiguous iters + in-bounds headers). With
    ``remote=False`` only the rank's OWN windows are considered; with
    ``remote=True`` only the replicated pair-rank copies (index entries with
    kind == "remote", keys "r_w{S}", plan §3.7).
    """
    index_path = os.path.join(
        mem_fs_folder, f"rank_{rank}_moevement_metadata.json"
    )
    if not os.path.exists(index_path):
        return []
    try:
        with open(index_path) as f:
            index = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("[moevement] unreadable metadata index %s: %s",
                       index_path, e)
        return []

    usable: list[WindowBundle] = []
    for key, entry in index.items():
        is_remote = (
            isinstance(entry, dict) and entry.get("kind") == "remote"
        )
        if is_remote != remote:
            continue
        path = os.path.join(mem_fs_folder, f"rank_{rank}_moevement_{key}.pt")
        try:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            bundle = _build_bundle(key, saved["_meta"], saved["_pool_bytes"])
        except Exception as e:  # unusable window: fall back to the other one
            logger.warning(
                "[moevement] skipping window %s (rank %d): %s", key, rank, e
            )
            continue
        usable.append(bundle)
    usable.sort(key=lambda b: b.window_start)
    return usable


def find_committed_windows(mem_fs_folder: str, rank: int) -> WindowBundle | None:
    """Newest fully-committed OWN window of this rank, or None (fresh
    start). See load_usable_windows for validation semantics."""
    usable = load_usable_windows(mem_fs_folder, rank, remote=False)
    if not usable:
        return None
    chosen = usable[-1]
    logger.info(
        "[moevement] rank %d: recovering from window %s (steps %d..%d; "
        "%d candidate window(s))",
        rank, chosen.key, chosen.window_start, chosen.last_step, len(usable),
    )
    return chosen


@torch.no_grad()
def apply_iteration(
    bundle: WindowBundle,
    iter_step: int,
    operators_by_name: dict[str, Operator],
    optimizers,
    activated: set[str],
) -> None:
    """Apply iteration ``iter_step``'s captures in place into the live state.

    Ops captured ACTIVE at this iteration: exact fp32 param shard slice +
    Adam exp_avg/exp_avg_sq slices + per-param 'step' scalar, then added to
    ``activated``. Ops still FROZEN at this iteration: bf16 capture upcast
    into the fp32 shard (their exact restore comes at their own activation
    iteration; the upcast makes the next forward's bf16 cast bit-equal to
    the original's). fp32-consumed buffers restore at full fp32 either way.

    ``optimizers`` must already hold materialized Adam state for every
    scheduled param (the manager's load() ensures this).
    """
    header = bundle.headers.get(iter_step)
    if header is None:
        raise KeyError(
            f"iteration {iter_step} is not in window {bundle.key} "
            f"({bundle.window_start}..{bundle.last_step})"
        )
    views_by_op = bundle.tensors[iter_step]
    for op_name, op_meta in header.items():
        op = operators_by_name.get(op_name)
        if op is None:
            raise RuntimeError(
                f"[moevement] bundle operator {op_name} does not exist in "
                f"the live model — capture/restore topology mismatch"
            )
        views = views_by_op[op_name]
        if op_meta["is_active"]:
            for fqn, param, expert_idx in op.param_entries:
                _local_slice(param, expert_idx).copy_(views[f"params.{fqn}"])
                state = find_adam_state(optimizers, param)
                _local_slice(state["exp_avg"], expert_idx).copy_(
                    views[f"optimizer.{fqn}.exp_avg"]
                )
                _local_slice(state["exp_avg_sq"], expert_idx).copy_(
                    views[f"optimizer.{fqn}.exp_avg_sq"]
                )
                step_view = views.get(f"optimizer.{fqn}.step")
                if step_view is not None:
                    live_step = state.get("step")
                    if torch.is_tensor(live_step):
                        _local_slice(live_step, None).copy_(step_view)
                    else:  # int-step optimizers (not the torchtitan default)
                        state["step"] = step_view.item()
            activated.add(op_name)
        else:
            if op_name in activated:
                raise RuntimeError(
                    f"[moevement] window {bundle.key} iter {iter_step} "
                    f"carries a FROZEN capture for already-activated "
                    f"operator {op_name} — bundle/schedule inconsistency"
                )
            for fqn, param, expert_idx in op.param_entries:
                # copy_ performs the bf16 -> fp32 upcast (exact).
                _local_slice(param, expert_idx).copy_(
                    views[f"compute_weights.{fqn}"]
                )
        for fqn, buf in op.buffer_entries:
            _local_slice(buf, None).copy_(views[f"buffers.{fqn}"])


def restore_rng(rng: dict[str, torch.Tensor]) -> None:
    """Exact inverse of the transfer backend's capture_rng: torch CPU state,
    plus the capturing device's CUDA state when it was captured."""
    torch.set_rng_state(rng["torch_cpu"])
    if "torch_cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state(rng["torch_cuda"], torch.cuda.current_device())
