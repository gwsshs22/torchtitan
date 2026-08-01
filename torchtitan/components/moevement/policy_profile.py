"""Persistence of MoEvement's profiled sparse-checkpointing policy.

Algorithm 1 (scheduler.find_window_size) is a *profiling* decision: its two
inputs — the measured device->host bandwidth B_PCIe and the per-iteration
byte budget derived from it — cost a real measurement, and its verdict (the
world-aligned ``w_sparse``) is what the engine's shm window pools are sized
against for the whole run. This module records that verdict so later runs of
the same workload can reuse it instead of re-deriving it.

The convention deliberately mirrors the two that already exist in this repo:

  * progressive init writes ``<dump_dir>/init_profile/<mode>/solution.json``
    from a dedicated ``--leto.profile_init`` run and later
    ``--leto.progressive_init`` runs consume it (init.py);
  * the gemini checkpoint interval is derived once from ``run_iter_avg.sh``
    output and pinned per workload (``CKPT_INTERVAL_*`` in
    ``awsexps/common/run.sh``).

Here the artifact is ``<dump_dir>/moevement_profile/policy.json``, produced by
``awsexps/common/run_init.sh``'s moevement-policy phase (or by any run that
executes the policy) and consumed by every later run of that workload.

Layout (schema_version 1)::

    {
      "schema_version": 1,
      "created_at": "2026-07-31T09:12:44Z",     # UTC, ISO-8601
      "created_at_epoch": 1785...,
      "w_sparse": 4,                            # THE reusable number
      "w_sparse_source": "algorithm1_world_max",
      "winning_ranks": [3],                     # whose proposal won the MAX
      "provenance": {"model_name", "model_flavor",
                     "parallelism": {dp_replicate, dp_shard, cp, tp, pp, ep, etp},
                     "world_size"},
      "policy_inputs": {"overlap_target", "iter_time_sec", "iter_time_source",
                        "iter_time_measured_steps",
                        "pcie_bandwidth_source", "pcie_bandwidth_gbs_configured"},
      "ranks": {"<rank>": {rank, total_ops, pcie_bandwidth_gbs, iter_time_sec,
                           budget_bytes, proposed_w_sparse, proposed_num_active,
                           proposed_worst_slot_bytes, num_active,
                           worst_slot_bytes, max_window_bytes, won_world_max}}
    }

``ranks`` is a rank-keyed dict on purpose: operator sets differ per pipeline
stage (a PP2 run has a `non_expert` per virtual stage and only the local
experts), so ``total_ops`` / ``num_active`` / worst-slot bytes are *not*
uniform across the world and recording a single value would be a lie. Only
``w_sparse`` is world-uniform — that is exactly what the ``all_reduce(MAX)``
in ``MoevementCheckpointManager._pin_policy_window_size`` buys.

Reuse is gated on provenance: a profile recorded under a different
parallelism geometry, world size or model describes a different operator
population and a different byte budget, so it is rejected (loudly) rather
than silently mis-sizing the pools.
"""

import json
import os
import time
from typing import Any

SCHEMA_VERSION = 1
PROFILE_DIRNAME = "moevement_profile"
PROFILE_FILENAME = "policy.json"

# Parallelism dims recorded and compared. `world_size` is compared separately.
PARALLELISM_FIELDS = ("dp_replicate", "dp_shard", "cp", "tp", "pp", "ep", "etp")


def default_profile_path(dump_folder: str) -> str:
    """``<dump_dir>/moevement_profile/policy.json`` (repo convention).

    Empty when there is no dump_dir: the artifact is a per-workload record
    and has no meaningful home relative to whatever cwd a process happens to
    have. Callers treat "" as "policy persistence disabled".
    """
    if not dump_folder:
        return ""
    return os.path.join(dump_folder, PROFILE_DIRNAME, PROFILE_FILENAME)


def build_provenance(
    model_name: str,
    model_flavor: str,
    parallel_dims: Any,
    world_size: int,
) -> dict[str, Any]:
    """The identity a recorded profile must match to be reusable."""
    dims: dict[str, int] = {}
    if parallel_dims is not None:
        for field in PARALLELISM_FIELDS:
            value = getattr(parallel_dims, field, None)
            if value is not None:
                dims[field] = int(value)
    return {
        "model_name": str(model_name or ""),
        "model_flavor": str(model_flavor or ""),
        "parallelism": dims,
        "world_size": int(world_size),
    }


def provenance_mismatches(
    recorded: dict[str, Any] | None, current: dict[str, Any]
) -> list[str]:
    """Human-readable reasons ``recorded`` cannot be reused for ``current``.

    Empty list == reusable. The gate is the parallelism geometry and the
    world size (which together fix the operator population and therefore the
    per-slot byte budget), plus the model name/flavor — a different model in
    the same dump_dir would otherwise silently reuse an unrelated cadence.
    """
    reasons: list[str] = []
    rec = recorded if isinstance(recorded, dict) else {}
    if not rec:
        return ["profile has no provenance block"]

    for key in ("model_name", "model_flavor", "world_size"):
        rec_v = rec.get(key)
        cur_v = current.get(key)
        if key == "world_size":
            rec_v = None if rec_v is None else int(rec_v)
            cur_v = None if cur_v is None else int(cur_v)
        if rec_v != cur_v:
            reasons.append(f"{key}: recorded={rec_v!r} current={cur_v!r}")

    rec_dims = rec.get("parallelism")
    rec_dims = rec_dims if isinstance(rec_dims, dict) else {}
    cur_dims = current.get("parallelism") or {}
    for dim in sorted(set(rec_dims) | set(cur_dims)):
        rec_d = rec_dims.get(dim)
        cur_d = cur_dims.get(dim)
        rec_d = None if rec_d is None else int(rec_d)
        cur_d = None if cur_d is None else int(cur_d)
        if rec_d != cur_d:
            reasons.append(f"parallelism.{dim}: recorded={rec_d!r} current={cur_d!r}")
    return reasons


def read_profile(path: str) -> dict[str, Any] | None:
    """Load a recorded profile; None when absent or unreadable/corrupt.

    A corrupt artifact is never fatal — the caller falls back to running the
    policy live (and overwrites the file with a good one).
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_profile(path: str, payload: dict[str, Any]) -> str:
    """Atomically publish ``payload`` at ``path`` (tmp + fsync + os.replace).

    Same durability shape gemini's metadata claim uses: readers on other
    hosts (the artifact lives on the shared dump_dir) either see the previous
    complete file or the new complete file, never a half-written one.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def build_payload(
    w_sparse: int,
    provenance: dict[str, Any],
    policy_inputs: dict[str, Any],
    rank_rows: list[dict[str, Any]],
    now: float | None = None,
) -> dict[str, Any]:
    """Assemble the artifact from the world's per-rank policy reports."""
    now = time.time() if now is None else now
    ranks = {str(int(row["rank"])): row for row in rank_rows}
    winners = sorted(
        int(row["rank"])
        for row in rank_rows
        if int(row.get("proposed_w_sparse", -1)) == int(w_sparse)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "created_at_epoch": round(now, 3),
        "w_sparse": int(w_sparse),
        "w_sparse_source": "algorithm1_world_max",
        "winning_ranks": winners,
        "provenance": provenance,
        "policy_inputs": policy_inputs,
        "ranks": ranks,
    }


def usable_w_sparse(
    profile: dict[str, Any] | None,
    current_provenance: dict[str, Any],
    rank: int,
    local_total_ops: int | None = None,
) -> tuple[int | None, list[str]]:
    """Decide whether ``profile`` may pin this run's cadence.

    Returns ``(w_sparse, [])`` when reusable, else ``(None, reasons)``. The
    caller is expected to log the reasons at WARNING and fall back to running
    Algorithm 1 live.
    """
    if profile is None:
        return None, ["no recorded profile"]

    reasons: list[str] = []
    schema = profile.get("schema_version")
    if schema != SCHEMA_VERSION:
        reasons.append(f"schema_version: recorded={schema!r} expected={SCHEMA_VERSION}")

    reasons.extend(provenance_mismatches(profile.get("provenance"), current_provenance))

    w_sparse = profile.get("w_sparse")
    if not isinstance(w_sparse, int) or isinstance(w_sparse, bool) or w_sparse < 1:
        reasons.append(f"w_sparse missing or invalid: {w_sparse!r}")

    # Per-rank cross-check: operator counts are per-pipeline-stage, so a
    # recorded entry that disagrees with what discovery just found on THIS
    # rank means the artifact describes a different operator population even
    # if the coarse provenance happens to line up.
    ranks = profile.get("ranks")
    ranks = ranks if isinstance(ranks, dict) else {}
    entry = ranks.get(str(int(rank)))
    if entry is None:
        reasons.append(f"no recorded entry for rank {rank}")
    elif local_total_ops is not None:
        recorded_ops = entry.get("total_ops")
        if recorded_ops is None or int(recorded_ops) != int(local_total_ops):
            reasons.append(
                f"rank {rank} total_ops: recorded={recorded_ops!r} "
                f"current={int(local_total_ops)}"
            )

    if reasons:
        return None, reasons
    return int(w_sparse), []


def describe(profile: dict[str, Any]) -> str:
    """Short provenance blurb for the 'where did my schedule come from' log."""
    prov = profile.get("provenance") or {}
    dims = prov.get("parallelism") or {}
    dims_s = ",".join(f"{k}={dims[k]}" for k in PARALLELISM_FIELDS if k in dims)
    return (
        f"recorded {profile.get('created_at', '?')} for "
        f"{prov.get('model_name', '?')}/{prov.get('model_flavor', '?')} "
        f"world_size={prov.get('world_size', '?')} [{dims_s}]; "
        f"winning_ranks={profile.get('winning_ranks', [])}"
    )
