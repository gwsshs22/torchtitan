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

    def find_window_size(self, iter_time_sec: float) -> tuple[int, int]:
        """Pick (w_sparse, num_active) for the per-iteration PCIe budget.

        Budget = iter_time_sec * pcie_bandwidth * overlap_target. Scans
        num_active from all-ops DOWN; the first (= largest) fit wins, giving
        the smallest w_sparse = ceil(total_ops / num_active).
        """
        return self._find_window_size(self._ordered(), iter_time_sec)

    def _find_window_size(
        self, ordered: list[SchedulableOp], iter_time_sec: float
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
        if worst > unscaled_budget:
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

    def regenerate(self, iter_time_sec: float) -> list[CheckpointSchedule]:
        """Build the window schedule for the current popularity ordering."""
        ordered = self._ordered()
        if self.w_sparse_override > 0:
            # Pinned cadence (world-aligned by the caller): num_active derives
            # from w_sparse. Trailing slots may carry empty active sets — a
            # "rest" slot with no D2H — keeping slot indices aligned with
            # peers that have more ops in the same window.
            w_sparse = self.w_sparse_override
            num_active = max(1, math.ceil(len(ordered) / w_sparse))
        else:
            w_sparse, num_active = self._find_window_size(ordered, iter_time_sec)

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
        floor (1 without an override — find_window_size never goes lower;
        ceil(total/w_sparse_override) with one — then num_active cannot
        vary at all) and pairs the largest frozen costs with the largest
        slot indices (rearrangement bound over any ordering).
        """
        n = len(self.ops)
        if n == 0:
            return 0
        if self.w_sparse_override > 0:
            na_floor = max(1, math.ceil(n / self.w_sparse_override))
        else:
            na_floor = 1
        total_active = sum(op.active_bytes for op in self.ops)
        frozen_asc = sorted(op.frozen_bytes for op in self.ops)
        return total_active + sum(
            (p // na_floor) * fb for p, fb in enumerate(frozen_asc)
        )
