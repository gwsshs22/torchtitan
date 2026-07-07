"""Progressive standby init — explicit GPU-memory reservation.

A standby ("shadow") rank runs its init tasks against an explicit reservation
with the co-located active rank, instead of guessing from a stale free-memory
reading. Both share a per-rank shm ledger `/dev/shm/leto_progressive_rank_{R}`
(the file the worker controller pre-creates). The active process runs the C++
reservation broker (see `torchtitan/components/mem`), which owns the device's
NVML view and grants/denies; the standby here only posts requests and waits.

Per init task the standby posts its *cumulative* target footprint (sum of the
profiled deltas of all reserved tasks so far). The broker grants iff
`others_used + granted <= capacity - margin`, accounting the standby at its
reservation rather than its current usage. Cumulative targets make a request
idempotent under the MIN all-reduce retry across standby ranks: a task only
advances when *all* ranks are granted, and re-requesting the same target never
double-counts.
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import struct
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


_PREFIX_ENV = "LETO_PROGRESSIVE_SHM_PREFIX"

# Reservation-ledger byte layout. MUST match `struct ReservationLedger` in
# torchtitan/components/mem/leto_free_mem_callback.cpp. assert_ledger_layout()
# (and tests) verify this against the C++ module's exported offsets.
LEDGER_NBYTES = 64
LEDGER_MAGIC = 0x4C54524E  # "LTRN"
OFF_MAGIC = 0
OFF_STANDBY_PID = 4
OFF_STANDBY_EPOCH = 8
OFF_REQ_SEQ = 12
OFF_REQ_BYTES = 16
OFF_RESP_SEQ = 24
OFF_RESP_VERDICT = 28
OFF_GRANTED = 32
OFF_EFFECTIVE_FREE = 40
OFF_STANDBY_ACTUAL = 48
OFF_REQ_TYPE = 56
VERDICT_PENDING = -1
VERDICT_DENY = 0
VERDICT_GRANT = 1
REQ_RESERVE = 1
REQ_GRANT = 2
REQ_ROLLBACK = 3

_I32 = struct.Struct("<i")
_U32 = struct.Struct("<I")
_I64 = struct.Struct("<q")
_mmap_cache: dict[int, mmap.mmap] = {}
_resv_seq: dict[int, int] = {}  # rank -> last request seq we posted


def _shm_path(rank: int) -> str:
    prefix = os.environ.get(_PREFIX_ENV)
    if not prefix:
        raise RuntimeError(
            f"{_PREFIX_ENV} not set — worker controller did not initialize "
            f"progressive shm (is enable_standby on?)"
        )
    return f"{prefix}{rank}"


def _get_mmap(rank: int) -> mmap.mmap:
    mm = _mmap_cache.get(rank)
    if mm is not None:
        return mm
    path = _shm_path(rank)
    fd = os.open(path, os.O_RDWR)
    try:
        # The worker controller creates the file at LEDGER_NBYTES; grow it
        # defensively in case an older controller created the 4-byte version.
        if os.fstat(fd).st_size < LEDGER_NBYTES:
            os.ftruncate(fd, LEDGER_NBYTES)
        mm = mmap.mmap(fd, LEDGER_NBYTES, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)
    _mmap_cache[rank] = mm
    return mm


def get_free_mb() -> int:
    """Current GPU free memory in MiB via torch.cuda.mem_get_info. Kept for
    callers (e.g. the active OOM-safeguard kill callback); the reservation
    broker itself reads NVML, not this."""
    free_b, _ = torch.cuda.mem_get_info()
    return int(free_b // (1024 * 1024))


def assert_ledger_layout() -> None:
    """Verify this module's layout constants match the C++ struct. Raises on
    drift. Cheap; call from tests (or once at standby startup)."""
    from torchtitan.components.mem import ledger_constants

    c = ledger_constants()
    here = {
        "LEDGER_NBYTES": LEDGER_NBYTES,
        "LEDGER_MAGIC": LEDGER_MAGIC,
        "VERDICT_PENDING": VERDICT_PENDING,
        "VERDICT_DENY": VERDICT_DENY,
        "VERDICT_GRANT": VERDICT_GRANT,
        "OFF_MAGIC": OFF_MAGIC,
        "OFF_STANDBY_PID": OFF_STANDBY_PID,
        "OFF_STANDBY_EPOCH": OFF_STANDBY_EPOCH,
        "OFF_REQ_SEQ": OFF_REQ_SEQ,
        "OFF_REQ_BYTES": OFF_REQ_BYTES,
        "OFF_RESP_SEQ": OFF_RESP_SEQ,
        "OFF_RESP_VERDICT": OFF_RESP_VERDICT,
        "OFF_GRANTED": OFF_GRANTED,
        "OFF_EFFECTIVE_FREE": OFF_EFFECTIVE_FREE,
        "OFF_STANDBY_ACTUAL": OFF_STANDBY_ACTUAL,
        "OFF_REQ_TYPE": OFF_REQ_TYPE,
        "REQ_RESERVE": REQ_RESERVE,
        "REQ_GRANT": REQ_GRANT,
        "REQ_ROLLBACK": REQ_ROLLBACK,
    }
    for k, v in here.items():
        if c.get(k) != v:
            raise RuntimeError(
                f"reservation ledger layout drift: {k} python={v} cpp={c.get(k)}"
            )


# ---------------------------------------------------------------------------
# Standby side — reservation client
# ---------------------------------------------------------------------------

def standby_register(rank: int, epoch: Optional[int] = None) -> None:
    """Identify this standby instance in the ledger (call once before the
    first reservation). `epoch` defaults to LETO_PROCESS_GROUP_ID so a fresh
    standby instance makes the broker void the prior grant."""
    if epoch is None:
        epoch = int(os.environ.get("LETO_PROCESS_GROUP_ID", os.getpid()))
    mm = _get_mmap(rank)
    # Start our request-seq above the broker's last response so the first
    # reserve() doesn't mistake a stale response for its own.
    _resv_seq[rank] = _U32.unpack_from(mm, OFF_RESP_SEQ)[0]
    _I32.pack_into(mm, OFF_STANDBY_PID, int(os.getpid()))
    _U32.pack_into(mm, OFF_STANDBY_EPOCH, int(epoch) & 0xFFFFFFFF)


StatusCheck = Callable[[], Optional[int]]


def _request(
    rank: int,
    req_type: int,
    cumulative_bytes: int,
    poll_interval_s: float,
    status_check: Optional[StatusCheck],
    status_interval_s: float = 1.0,
) -> Tuple[str, int]:
    """Post one two-phase-protocol request and block for the broker's verdict.

    req_type: REQ_RESERVE (soft claim, cancellable by the active's callback
    under pressure), REQ_GRANT (hard commit; allowed only while the
    reservation is alive), or REQ_ROLLBACK (cumulative_bytes = the pre-task
    cumulative to roll back to; always acked).

    Returns ("grant"|"deny", 0) or ("status", code) if status_check trips.
    """
    mm = _get_mmap(rank)
    seq = (_resv_seq.get(rank, 0) + 1) & 0xFFFFFFFF
    if seq == 0:
        seq = 1  # 0 is the ledger's initial resp_seq; never use it as a req
    _resv_seq[rank] = seq
    _I64.pack_into(mm, OFF_REQ_BYTES, int(cumulative_bytes))
    _I32.pack_into(mm, OFF_REQ_TYPE, int(req_type))
    _U32.pack_into(mm, OFF_REQ_SEQ, seq)  # publish last → signals the request
    next_status = time.monotonic() + status_interval_s
    while True:
        if _U32.unpack_from(mm, OFF_RESP_SEQ)[0] == seq:
            verdict = _I32.unpack_from(mm, OFF_RESP_VERDICT)[0]
            return ("grant" if verdict == VERDICT_GRANT else "deny", 0)
        if status_check is not None and time.monotonic() >= next_status:
            code = status_check()
            if code:
                return ("status", int(code))
            next_status = time.monotonic() + status_interval_s
        time.sleep(poll_interval_s)


def _reserve(
    rank: int,
    cumulative_bytes: int,
    poll_interval_s: float,
    status_check: Optional[StatusCheck],
    status_interval_s: float = 1.0,
) -> Tuple[str, int]:
    """Single-rank RESERVE+GRANT convenience (no cross-rank unanimity) for
    callers that raise one rank's cumulative directly (test hooks). Returns
    the final verdict; a phase-2 deny (reservation canceled between the two
    requests) reports "deny" and leaves nothing committed."""
    kind, code = _request(rank, REQ_RESERVE, cumulative_bytes,
                          poll_interval_s, status_check, status_interval_s)
    if kind != "grant":
        return (kind, code)
    return _request(rank, REQ_GRANT, cumulative_bytes,
                    poll_interval_s, status_check, status_interval_s)


def _reduce_status_ok(gloo_pg, local_status: int, local_ok: int):
    """The status-carry reduce pair: every rank always enters both reduces
    (a rank that returned early would hang its peers in gloo). Returns
    (global_status, all_ok)."""
    s = torch.tensor([local_status], dtype=torch.int32)
    dist.all_reduce(s, op=dist.ReduceOp.MAX, group=gloo_pg)
    t = torch.tensor([local_ok], dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=gloo_pg)
    return int(s.item()), int(t.item()) == 1


def try_advance(
    gloo_pg,
    delta_mb: float,
    cumulative_mb: float,
    threshold_mb: float,
    poll_interval_s: float,
    status_check: Optional[StatusCheck] = None,
    protocol: str = "two_phase",
) -> Tuple[str, Optional[int]]:
    """Decide, across all standby ranks, whether the next task can advance —
    TWO-PHASE: (1) every rank RESERVEs (soft, cancellable by the active's
    callback under pressure); only if all ranks hold a reservation, (2) every
    rank GRANTs (hard commit). A phase-2 deny means some rank's reservation
    was canceled between the phases — every rank then ROLLBACKs to the
    pre-task cumulative and the whole procedure is redone, so a unilateral
    commit never lingers as phantom kill pressure on the ranks whose peers
    denied (the useless-kill problem).

    Tasks with `delta_mb < threshold_mb` (per-rank) allocate ~no GPU memory
    and skip both phases locally, but still join every reduce. Caller loops
    on "retry".

    Returns:
      ("advance", None) — all ranks committed; run the task
      ("retry",   None) — denied at some phase; poll again
      ("status",  code) — status_check tripped (ACTIVATE / TERMINATE)
    """
    rank = int(os.environ.get("RANK", "0"))
    cum_bytes = int(cumulative_mb * 1024 * 1024)
    prev_cum_bytes = max(0, int((cumulative_mb - delta_mb) * 1024 * 1024))
    tiny = delta_mb < threshold_mb

    # SUBTRACTIVE ABLATION (env-gated, default off) for the standby-overhead
    # root-cause (awsexps/measure_mem/fix_overhead). Remove ONE component from the
    # FULL parked loop to see which removal kills the active's per-step tax:
    #   LETO_ABLATE=nogloo    — skip the cross-rank gloo all_reduce (local vote only)
    #   LETO_ABLATE=noreserve — skip the broker RESERVE handshake (behave as denied,
    #                           so ranks reach the gloo with no RESERVE-latency spread)
    #   LETO_ABLATE=nostatus  — skip the status gRPC inside the reserve wait
    # (comma-separable). Default "" == unchanged behavior.
    _ablate = os.environ.get("LETO_ABLATE", "")
    _no_gloo = "nogloo" in _ablate
    _no_resv = "noreserve" in _ablate
    _no_stat = "nostatus" in _ablate

    def _reduce(ls: int, lo: int) -> Tuple[int, bool]:
        if _no_gloo:
            return int(ls), int(lo) == 1  # skip the collective; local vote only
        return _reduce_status_ok(gloo_pg, ls, lo)

    def _phase(req_type: int) -> Tuple[int, int]:
        """Run one protocol phase locally; returns (local_status, local_ok)."""
        if tiny:
            return 0, 1  # tiny / CPU-only task: no reservation needed
        if _no_resv:
            return 0, 0  # ablate broker handshake; behave as denied (no RESERVE lat)
        kind, code = _request(rank, req_type, cum_bytes, poll_interval_s,
                              None if _no_stat else status_check)
        if kind == "status":
            return int(code), 0
        logger.debug(
            f"[progressive] rank={rank} cumulative_mb={cumulative_mb:.1f} "
            f"type={req_type} → {kind}"
        )
        return 0, 1 if kind == "grant" else 0

    # Ablation (leto.progressive_protocol=grant_only): single-phase — GRANT
    # directly, one unanimity round, NO rollback, so a rank that committed
    # while its peers denied keeps the phantom entitlement. This is the
    # pre-two-phase behavior, kept to measure the protocol's impact.
    if protocol == "grant_only":
        local_status, local_ok = _phase(REQ_GRANT)
        global_status, all_ok = _reduce(local_status, local_ok)
        if global_status > 0:
            return ("status", global_status)
        return ("advance" if all_ok else "retry", None)

    # Phase 1 — RESERVE on every rank. A partial success leaves soft
    # reservations behind on the granted ranks: harmless (nothing
    # allocated) and self-cleaning (the callback cancels them under
    # pressure); the retry re-RESERVEs idempotently.
    local_status, local_ok = _phase(REQ_RESERVE)
    global_status, all_ok = _reduce(local_status, local_ok)
    if global_status > 0:
        return ("status", global_status)
    if not all_ok:
        return ("retry", None)

    # Phase 2 — GRANT on every rank (allowed only while the reservation is
    # alive; a cancellation in between shows up as a deny here).
    local_status, local_ok = _phase(REQ_GRANT)
    global_status, all_ok = _reduce(local_status, local_ok)
    if global_status > 0:
        return ("status", global_status)
    if all_ok:
        return ("advance", None)

    # Some rank's reservation was canceled between the phases: roll back
    # EVERY rank to the pre-task cumulative (ranks that committed phase 2
    # lower their grant; the rest just clear any leftover reservation), then
    # redo the whole two-phase procedure. One extra status reduce keeps the
    # collective count uniform if a promotion lands mid-rollback.
    local_status = 0
    if not tiny:
        kind, code = _request(rank, REQ_ROLLBACK, prev_cum_bytes,
                              poll_interval_s, status_check)
        if kind == "status":
            local_status = int(code)
    global_status, _ = _reduce(local_status, 1)
    if global_status > 0:
        return ("status", global_status)
    return ("retry", None)


# ---------------------------------------------------------------------------
# Schedule / profile loading
# ---------------------------------------------------------------------------

_SOLUTION_FILES = {
    "solver": "solution.json",
    "no_reordering": "solution_no_reordering.json",
}


def load_solution(dump_folder: str, mode: str,
                  variant: str = "solver") -> list[str]:
    """Return task names in the scheduled order.

    variant selects the schedule (leto.progressive_solution):
      "solver"        — the solver-optimized order (solution.json)
      "no_reordering" — progressive gating with the BASELINE execution order
                        (solution_no_reordering.json); the A/B control that
                        isolates the benefit of the solver's reordering.
    """
    filename = _SOLUTION_FILES.get(variant)
    if filename is None:
        raise ValueError(
            f"leto.progressive_solution={variant!r}: expected one of "
            f"{sorted(_SOLUTION_FILES)}"
        )
    path = Path(dump_folder) / "init_profile" / mode / filename
    if not path.exists():
        raise FileNotFoundError(
            f"progressive_init requires {path}. Run the profiling job first "
            f"(leto.profile_init=true) to generate it."
        )
    data = json.loads(path.read_text())
    return [t["name"] for t in data["tasks"]]


def load_rank_deltas(dump_folder: str, mode: str, rank: int) -> dict[str, float]:
    """Return {task_name: delta_mb} from this rank's profile."""
    path = Path(dump_folder) / "init_profile" / mode / f"rank_{rank}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"progressive_init requires {path}. Run the profiling job first "
            f"(leto.profile_init=true) to generate it."
        )
    data = json.loads(path.read_text())
    out: dict[str, float] = {}
    for entry in data["tasks"]:
        dm = entry.get("delta_mb")
        out[entry["task"]] = 0.0 if dm is None else float(dm)
    return out
