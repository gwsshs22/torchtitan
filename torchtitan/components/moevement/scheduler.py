"""Sparse checkpoint scheduling policy (MoEvement paper Algorithm 1).

Port of MoEvement's scheduler onto a torch-free bytes-and-names core:
operators are described by SchedulableOp (name + per-state snapshot byte
costs), the scheduler picks the sparse window size w_sparse and emits one
CheckpointSchedule per window slot. Within one window every operator is
captured ACTIVE (fp32 params + optimizer moments) exactly once; on each
earlier slot of the same window it is captured FROZEN (bf16 compute copy) —
slot i's frozen set is strictly the not-yet-captured tail, operators already
captured ACTIVE in earlier slots appear in neither list (reference
scheduler.py:281-296; the paper's prose disagrees, we follow the code —
docs/moevement_port_plan.md §6-C11).

Cadence ownership: find_window_size IS Algorithm 1, and it is evaluated on
every regenerate — but the *applied* w_sparse can be pinned, either by
config (checkpoint.moevement_w_sparse_override) or by the manager, which
pins the world-aligned verdict once at init (pin_window_size). Both cases
still log what the policy would pick right now, so a pinned run reports the
policy's live opinion instead of hiding it.
"""

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SchedulableOp:
    """Minimal per-operator view the scheduler consumes.

    Discovery (operators.py) adapts its richer operator records to this
    protocol via .schedulable().
    """

    name: str  # unique
    active_bytes: int  # bytes to snapshot when ACTIVE (fp32 params + optimizer moments, local shard)
    frozen_bytes: int  # bytes to snapshot when FROZEN (bf16 compute copy + fp32-consumed buffers)
    is_expert: bool  # experts are popularity-ordered; gates/non_expert go to the tail


@dataclass
class CheckpointSchedule:
    """One window slot's assignment (operator names)."""

    slot_index: int
    active: list[str]
    frozen: list[str]


class SparseCheckpointScheduler:
    """Implements Algorithm 1: window sizing + popularity-ordered slots.

    Popularity is fed externally via update_popularity (the manager reads
    rank-local tokens_per_expert and all-reduces the accumulated counts at
    window boundaries — plan §3.3); ordering always uses the accumulated
    totals at regenerate() time.
    """

    def __init__(
        self,
        ops: list[SchedulableOp],
        pcie_bandwidth_gbs: float,
        overlap_target: float,
        w_sparse_override: int = 0,
        reorder_threshold: float = 0.10,
        reorder_fraction: float = 0.25,
    ) -> None:
        names = [op.name for op in ops]
        if len(set(names)) != len(names):
            raise ValueError("SchedulableOp names must be unique")
        self.ops = list(ops)
        # Matches the reference config's pcie_bandwidth_bytes_per_sec (GiB/s).
        self.pcie_bandwidth_bytes = pcie_bandwidth_gbs * (1024**3)
        self.overlap_target = overlap_target
        self.w_sparse_override = w_sparse_override
        # Cadence pinned by the CALLER after Algorithm 1 has run once — the
        # manager pins the world-MAX of the per-rank proposal (reference
        # coordinator._generate_schedule_world_aligned). From then on it
        # behaves exactly like w_sparse_override: per-rank operator sets and
        # per-rank iteration-time EMAs must not be allowed to drift the
        # cadence apart (every cross-rank protocol here is iter-keyed), and
        # the engine's window pools are sized ONCE against this cadence.
        self.w_sparse_pin = 0
        self._pin_reason = ""
        self._pin_from_policy = False
        self.reorder_threshold = reorder_threshold
        self.reorder_fraction = reorder_fraction

        self.w_sparse = 1
        self.num_active = 0
        self.schedule: list[CheckpointSchedule] = []
        # Accumulated per-op popularity totals fed by update_popularity.
        self._popularity: dict[str, float] = {}
        # Per-expert totals snapshotted when the ordering was last used
        # (regenerate); needs_reorder measures shift against these.
        self._popularity_at_last_order: dict[str, float] | None = None

    def pinned_w_sparse(self) -> int:
        """Cadence this scheduler is committed to, 0 when Algorithm 1 is free
        to re-pick it every regenerate (config override wins over the pin)."""
        return self.w_sparse_override or self.w_sparse_pin

    def pin_window_size(
        self, w_sparse: int, reason: str = "", from_policy: bool = False
    ) -> None:
        """Freeze the applied cadence at ``w_sparse`` for every later
        regenerate. Algorithm 1 keeps being *evaluated* (and logged) at each
        regenerate — only its verdict stops being applied.

        ``from_policy`` marks a pin that IS Algorithm 1's own decision (the
        manager's world-aligned init-time verdict) rather than an external
        cadence, so the logs can say APPLIED instead of BYPASSED — and can
        call out the case where the policy's live verdict has since drifted
        away from what the pools were sized for.
        """
        self.w_sparse_pin = max(1, int(w_sparse))
        self._pin_reason = reason or "caller"
        self._pin_from_policy = from_policy

    def update_popularity(self, counts: dict[str, float]) -> None:
        """Accumulate per-expert token counts into the popularity totals."""
        for name, count in counts.items():
            self._popularity[name] = self._popularity.get(name, 0.0) + count

    def _ordered(self) -> list[SchedulableOp]:
        # Experts ascending by accumulated popularity (least popular first);
        # gates/non_expert at the tail — they fire on every token, so their
        # effective popularity is maximal, and tail placement defers their
        # (large) snapshot to the window's last slots (paper Fig. 6).
        #
        # Deviation from the reference (deliberate): the reference's sort is
        # stable on registration order, so equal counts could order ops
        # differently across ranks if discovery order ever differed — and
        # slot-level active sets must match world-wide. Tie-break by name
        # (and name-sort the tail) so equal counts give the identical order
        # on every rank regardless of input order.
        experts = sorted(
            (op for op in self.ops if op.is_expert),
            key=lambda op: (self._popularity.get(op.name, 0.0), op.name),
        )
        non_experts = sorted(
            (op for op in self.ops if not op.is_expert), key=lambda op: op.name
        )
        return experts + non_experts

    @staticmethod
    def _worst_slot_bytes(ordered: list[SchedulableOp], num_active: int) -> int:
        """Max over slots of active bytes + not-yet-captured-tail frozen bytes.

        Prices every slot rather than slot 0: tail slots carry non_expert
        (the dense backbone, often 10x+ any single expert) and would
        silently blow the budget otherwise.
        """
        n = len(ordered)
        active_prefix = [0] * (n + 1)
        for i, op in enumerate(ordered):
            active_prefix[i + 1] = active_prefix[i] + op.active_bytes
        frozen_suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            frozen_suffix[i] = frozen_suffix[i + 1] + ordered[i].frozen_bytes
        worst = 0
        for start in range(0, n, num_active):
            end = min(start + num_active, n)
            slot = active_prefix[end] - active_prefix[start] + frozen_suffix[end]
            worst = max(worst, slot)
        return worst

    def find_window_size(
        self, iter_time_sec: float, warn: bool = True
    ) -> tuple[int, int]:
        """Pick (w_sparse, num_active) for the per-iteration PCIe budget.

        Budget = iter_time_sec * pcie_bandwidth * overlap_target. Scans
        num_active from all-ops DOWN; the first (= largest) fit wins, giving
        the smallest w_sparse = ceil(total_ops / num_active).
        """
        return self._find_window_size(self._ordered(), iter_time_sec, warn=warn)

    def _find_window_size(
        self,
        ordered: list[SchedulableOp],
        iter_time_sec: float,
        warn: bool = True,
    ) -> tuple[int, int]:
        total_ops = len(ordered)
        if total_ops == 0:
            return 1, 0
        budget = iter_time_sec * self.pcie_bandwidth_bytes * self.overlap_target
        num_active = total_ops
        while num_active > 1:
            if self._worst_slot_bytes(ordered, num_active) <= budget:
                break
            num_active -= 1
        w_sparse = math.ceil(total_ops / num_active)
        # num_active floors at 1 even when that slot still overruns — a
        # slow-PCIe config yields a larger window rather than a hard failure.
        # Warn against the *unscaled* iter budget (like the reference) so a
        # deliberately tight overlap_target doesn't spuriously fire this.
        worst = self._worst_slot_bytes(ordered, num_active)
        unscaled_budget = iter_time_sec * self.pcie_bandwidth_bytes
        if warn and worst > unscaled_budget:
            logger.warning(
                "[moevement] schedule PCIe budget cannot be met: worst-slot "
                "snapshot = %.1f MB, iter budget = %.1f MB; w_sparse=%d, "
                "num_active=%d — the snapshot stream will fall behind training",
                worst / 1e6,
                unscaled_budget / 1e6,
                w_sparse,
                num_active,
            )
        return w_sparse, num_active

    def policy_report(self, iter_time_sec: float) -> dict:
        """Algorithm 1's inputs and verdict AS DATA (what _log_policy prints).

        Same numbers, machine-readable, so the manager can persist the
        decision (policy_profile.py) instead of leaving it in a log line.
        ``proposed_*`` is what the policy wants right now; ``applied_*`` is
        what the pinned cadence actually produces on this rank (they differ
        on every rank that lost the world MAX).
        """
        ordered = self._ordered()
        algo_w, algo_na = self._find_window_size(ordered, iter_time_sec, warn=False)
        budget = iter_time_sec * self.pcie_bandwidth_bytes * self.overlap_target
        pinned = self.pinned_w_sparse()
        if pinned > 0:
            applied_w = pinned
            applied_na = max(1, math.ceil(len(ordered) / pinned)) if ordered else 0
        else:
            applied_w, applied_na = algo_w, algo_na
        return {
            "total_ops": len(ordered),
            "pcie_bandwidth_gbs": self.pcie_bandwidth_bytes / (1024**3),
            "overlap_target": self.overlap_target,
            "iter_time_sec": iter_time_sec,
            "budget_bytes": int(budget),
            "proposed_w_sparse": int(algo_w),
            "proposed_num_active": int(algo_na),
            "proposed_worst_slot_bytes": int(
                self._worst_slot_bytes(ordered, algo_na) if algo_na > 0 else 0
            ),
            "applied_w_sparse": int(applied_w),
            "num_active": int(applied_na),
            "worst_slot_bytes": int(
                self._worst_slot_bytes(ordered, applied_na) if applied_na > 0 else 0
            ),
            "max_window_bytes": int(self.max_window_bytes()),
        }

    def _log_policy(
        self,
        ordered: list[SchedulableOp],
        iter_time_sec: float,
        algo_w: int,
        algo_na: int,
        pinned: int,
    ) -> None:
        """Report Algorithm 1's inputs and verdict at INFO — including when
        the verdict is NOT applied, so a cadence-pinned run still states the
        policy's opinion (and its B_PCIe input, which is otherwise invisible).
        """
        budget = iter_time_sec * self.pcie_bandwidth_bytes * self.overlap_target
        worst = self._worst_slot_bytes(ordered, algo_na) if algo_na > 0 else 0
        slack = budget - worst
        pct = (100.0 * slack / budget) if budget > 0 else float("nan")
        if pinned == 0:
            verdict = "APPLIED — chose"
        elif not self._pin_from_policy:
            verdict = (
                f"BYPASSED (w_sparse pinned to {pinned} by "
                f"{self._pin_reason or 'config'}) — it WOULD have chosen"
            )
        elif algo_w == pinned:
            verdict = (
                f"APPLIED (cadence pinned at w_sparse={pinned} by "
                f"{self._pin_reason}) — reconfirms"
            )
        else:
            # The live verdict has drifted from the one the engine's pools
            # were sized against; the pin stands, and the drift is the
            # signal that the pinning point (init, on the SEEDED iteration
            # time) mispredicted this workload.
            verdict = (
                f"DRIFTED — cadence stays pinned at w_sparse={pinned} "
                f"({self._pin_reason}; pools are sized for it), but it now"
                f" prefers"
            )
        logger.info(
            "[moevement] Algorithm 1 %s num_active=%d, w_sparse=%d | "
            "B_PCIe=%.2f GiB/s, T_iter=%.4f s, overlap=%.2f -> budget "
            "%.1f MB/iter | %d operators, worst-slot %.1f MB, slack "
            "%.1f MB (%.1f%% of budget)",
            verdict,
            algo_na,
            algo_w,
            self.pcie_bandwidth_bytes / (1024**3),
            iter_time_sec,
            self.overlap_target,
            budget / 1e6,
            len(ordered),
            worst / 1e6,
            slack / 1e6,
            pct,
        )

    def regenerate(self, iter_time_sec: float) -> list[CheckpointSchedule]:
        """Build the window schedule for the current popularity ordering."""
        ordered = self._ordered()
        pinned = self.pinned_w_sparse()
        # Algorithm 1 is EVALUATED on every regenerate even when a pin is in
        # force — that is what makes a pinned run report the policy's opinion
        # (plan §9-M9). Only an applied verdict warns about an unmeetable
        # budget; a bypassed one just gets logged.
        algo_w, algo_na = self._find_window_size(
            ordered, iter_time_sec, warn=(pinned == 0)
        )
        self._log_policy(ordered, iter_time_sec, algo_w, algo_na, pinned)
        if pinned > 0:
            # Pinned cadence (config override, or the caller's world-aligned
            # Algorithm-1 decision): num_active derives from w_sparse.
            # Trailing slots may carry empty active sets — a "rest" slot with
            # no D2H — keeping slot indices aligned with peers that have more
            # ops in the same window.
            w_sparse = pinned
            num_active = max(1, math.ceil(len(ordered) / w_sparse))
        else:
            w_sparse, num_active = algo_w, algo_na

        schedule = []
        for i in range(w_sparse):
            start = i * num_active
            end = min(start + num_active, len(ordered))
            schedule.append(
                CheckpointSchedule(
                    slot_index=i,
                    active=[op.name for op in ordered[start:end]],
                    frozen=[op.name for op in ordered[end:]],
                )
            )

        self.w_sparse = w_sparse
        self.num_active = num_active
        self.schedule = schedule
        self._popularity_at_last_order = {
            op.name: self._popularity.get(op.name, 0.0)
            for op in ordered
            if op.is_expert
        }
        logger.info(
            "[moevement] sparse checkpoint schedule: w_sparse=%d, "
            "num_active=%d, total_ops=%d",
            w_sparse,
            num_active,
            len(ordered),
        )
        return schedule

    def needs_reorder(self) -> bool:
        """Whether expert popularity shifted enough since the last ordering.

        Mirrors the reference's should_reorder comparison semantics: an
        expert counts as changed when its relative shift is strictly greater
        than reorder_threshold (eps-guarded denominator so dormant->active
        registers as changed, not divide-by-zero); the reorder fires when
        the changed fraction is >= reorder_fraction. Never-ordered -> True,
        no experts -> False.
        """
        if self._popularity_at_last_order is None:
            return True
        experts = [op for op in self.ops if op.is_expert]
        if not experts:
            return False
        changed = 0
        for op in experts:
            old = self._popularity_at_last_order.get(op.name, 0.0)
            new = self._popularity.get(op.name, 0.0)
            if abs(new - old) / max(old, 1e-9) > self.reorder_threshold:
                changed += 1
        return changed / len(experts) >= self.reorder_fraction

    def max_window_bytes(self) -> int:
        """Monotone upper bound on one window's total snapshot bytes.

        Used for shm pool sizing (plan §8-R4): no regenerate() call — any
        popularity ordering, any iter time — can produce a window exceeding
        this, so pools sized to it never need realloc + re-REGISTER on
        schedule regen.

        A window captures each op ACTIVE once plus FROZEN once per slot
        before its active slot: the op at ordered position p (slot p//na)
        is frozen exactly p//na times. The bound takes num_active at its
        floor (1 while the cadence is free — find_window_size never goes
        lower; ceil(total/w) once the cadence is pinned by the config
        override OR by the manager's world-aligned Algorithm-1 decision —
        then num_active cannot vary at all) and pairs the largest frozen
        costs with the largest slot indices (rearrangement bound over any
        ordering).

        Pinning matters here: a free cadence forces na_floor=1, i.e. pools
        sized for the w_sparse=total_ops worst case (~20x the realistic
        window at 266 operators). The manager therefore pins the cadence
        BEFORE the engine sizes its pools.
        """
        n = len(self.ops)
        if n == 0:
            return 0
        pinned = self.pinned_w_sparse()
        if pinned > 0:
            na_floor = max(1, math.ceil(n / pinned))
        else:
            na_floor = 1
        total_active = sum(op.active_bytes for op in self.ops)
        frozen_asc = sorted(op.frozen_bytes for op in self.ops)
        return total_active + sum(
            (p // na_floor) * fb for p, fb in enumerate(frozen_asc)
        )
