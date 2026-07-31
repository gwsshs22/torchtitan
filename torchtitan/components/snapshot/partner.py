"""Shared replica-partner ("buddy") selection for the in-memory checkpointing
methods (gemini and moevement).

Both methods keep ONE peer copy of every rank's checkpoint state, and both
address that copy from BOTH sides during recovery (gemini:
faulty -> RECV / pair -> SEND; moevement: faulty -> fetch_from_pair /
pair -> serve_remote_window). Until this module existed the two picked the
partner differently — gemini `group_rank + group_size//2` inside the FSDP
process group, moevement `(rank + world//2) % world` over global ranks —
and gemini's variant is intra-host on any layout whose FSDP mesh does not
span hosts (pp2/dp_shard2/tp2 on 2-GPU hosts: FSDP group {0,2} may sit on
one node), so a host-fatal destroyed BOTH replicas of a pair. This module
is the single rule.

The rule
--------
1. Candidate pool = the ranks of ONE process group, in group-rank order.
   Callers pass the FSDP process group: its members are the ranks that own
   the same model chunk (same pipeline stage, same TP shard column), i.e.
   the only ranks whose checkpoint state is guaranteed identically shaped.
   The pool must have an even size >= 2.

2. Within the pool the pairing is a HALF-ROTATION matching parameterized by
   an offset k in [0, n/2):

       position i in the first half  <->  n/2 + ((i + k) mod n/2)

   Every member of this family is a perfect matching and is symmetric by
   construction (partner(partner(i)) == i), so requirement "mutual" holds
   for every k. k = 0 is exactly the legacy `+ n/2` rule; larger k rotate
   the second half against the first. (The naive generalization
   `partner(i) = (i + k) mod n` is NOT mutual for k != n/2 — 2k must vanish
   mod n — which is why the offset rotates one half instead of the whole
   pool.)

3. k is chosen to MAXIMIZE the number of pairs whose two members sit on
   different hosts, ties broken by the smallest k. Inputs (the pool's rank
   order and the host of each member, all-gathered at init) are identical
   on every rank, and the search is a deterministic argmax, so every rank
   computes the same pairing without a second collective.

4. If no offset makes every pair cross-host (e.g. the whole stage lives on
   one host), ``log_pairing`` emits an ERROR naming the co-located ranks and
   states that host-fatal recovery is unavailable in that layout; the best
   available offset is used anyway (transient/process-local faults still
   recover, and a single-host job cannot do better).

This module is pure stdlib on purpose: no torch, no torch.distributed. The
host map is produced by the callers, which piggyback a hostname on an
all_gather they already perform once at init (gemini: the pair-PG
all_gather_object; moevement: the pool-size exchange in build_replicator) —
never on the hot path.
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "PartnerPlan",
    "hostname",
    "is_pairable",
    "rotation_partners",
    "cross_host_pair_count",
    "choose_offset",
    "plan_pairing",
    "plan_pools",
    "all_pairs",
    "log_pairing",
]


def hostname() -> str:
    """The host identity used for the cross-host decision (one value per
    node; every process on a node must report the same string)."""
    return socket.gethostname()


def is_pairable(pool_size: int) -> bool:
    """A pool can be paired iff it has an even number of members >= 2."""
    return pool_size >= 2 and pool_size % 2 == 0


def rotation_partners(size: int, offset: int) -> list[int]:
    """position -> partner position for the half-rotation matching (see the
    module docstring). Symmetric and fixed-point-free for every offset."""
    if not is_pairable(size):
        raise ValueError(
            f"partner pool size must be even and >= 2, got {size}"
        )
    half = size // 2
    offset %= half
    partners = [-1] * size
    for i in range(half):
        j = half + (i + offset) % half
        partners[i] = j
        partners[j] = i
    return partners


def _pair_positions(partners: Sequence[int]) -> list[tuple[int, int]]:
    """The matching's unique (low, high) position pairs, sorted."""
    return sorted({(min(i, p), max(i, p)) for i, p in enumerate(partners)})


def cross_host_pair_count(
    partners: Sequence[int], hosts: Sequence[str]
) -> int:
    """How many of the matching's pairs straddle two hosts."""
    return sum(
        1 for a, b in _pair_positions(partners) if hosts[a] != hosts[b]
    )


def choose_offset(hosts: Sequence[str]) -> tuple[int, list[int]]:
    """Deterministic argmax over the half-rotation family.

    Returns (offset, partners). Ties (including the all-same-host case, where
    every offset scores 0) resolve to the SMALLEST offset, so a layout that
    was already fully cross-host under the legacy `+ n/2` rule keeps exactly
    that pairing.
    """
    size = len(hosts)
    best_offset, best_partners, best_score = 0, rotation_partners(size, 0), -1
    for offset in range(size // 2):
        partners = rotation_partners(size, offset)
        score = cross_host_pair_count(partners, hosts)
        if score > best_score:
            best_offset, best_partners, best_score = offset, partners, score
        if best_score == size // 2:
            break  # cannot do better than "every pair cross-host"
    return best_offset, best_partners


def _host_of(host_by_rank: Mapping[int, str] | Sequence[str], rank: int) -> str:
    try:
        return host_by_rank[rank]
    except (KeyError, IndexError):
        raise KeyError(
            f"no host recorded for global rank {rank}; the host map must "
            f"cover every member of the candidate pool"
        ) from None


@dataclass(frozen=True)
class PartnerPlan:
    """The pairing of one candidate pool. Value object — identical on every
    rank of the pool (and on every rank that knows the pool's membership)."""

    pool_ranks: tuple[int, ...]          # global ranks, in group-rank order
    hosts: tuple[str, ...]               # host of each pool member, same order
    offset: int                          # chosen half-rotation offset
    partner_pos: tuple[int, ...]         # position -> partner position

    @property
    def size(self) -> int:
        return len(self.pool_ranks)

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        """Unique (low, high) GLOBAL-rank pairs, sorted."""
        return tuple(
            sorted(
                (
                    min(self.pool_ranks[a], self.pool_ranks[b]),
                    max(self.pool_ranks[a], self.pool_ranks[b]),
                )
                for a, b in _pair_positions(self.partner_pos)
            )
        )

    @property
    def colocated_pairs(self) -> tuple[tuple[int, int], ...]:
        """The pairs whose two members share a host (empty == fully
        redundant against a host-fatal)."""
        return tuple(
            sorted(
                (
                    min(self.pool_ranks[a], self.pool_ranks[b]),
                    max(self.pool_ranks[a], self.pool_ranks[b]),
                )
                for a, b in _pair_positions(self.partner_pos)
                if self.hosts[a] == self.hosts[b]
            )
        )

    @property
    def cross_host_pairs(self) -> int:
        return cross_host_pair_count(self.partner_pos, self.hosts)

    @property
    def total_pairs(self) -> int:
        return self.size // 2

    @property
    def fully_cross_host(self) -> bool:
        return not self.colocated_pairs

    def position_of(self, global_rank: int) -> int:
        return self.pool_ranks.index(global_rank)

    def partner_of(self, global_rank: int) -> int:
        """The GLOBAL rank this rank replicates to / recovers from."""
        return self.pool_ranks[self.partner_pos[self.position_of(global_rank)]]

    def partner_position_of(self, global_rank: int) -> int:
        """The partner's position in the pool (== its group rank)."""
        return self.partner_pos[self.position_of(global_rank)]

    def host_of(self, global_rank: int) -> str:
        return self.hosts[self.position_of(global_rank)]


def plan_pairing(
    pool_ranks: Sequence[int],
    host_by_rank: Mapping[int, str] | Sequence[str],
) -> PartnerPlan:
    """THE rule: pair the pool's members so that as many pairs as possible
    are cross-host. Pure and deterministic — same inputs, same plan, on
    every rank.

    Args:
        pool_ranks: global ranks of the candidate pool, in group-rank order
            (i.e. ``dist.get_process_group_ranks(fsdp_pg)``).
        host_by_rank: global rank -> hostname, covering at least the pool.
    """
    pool = tuple(int(r) for r in pool_ranks)
    if not is_pairable(len(pool)):
        raise ValueError(
            f"candidate pool {pool} is not pairable: it needs an even "
            f"number of members >= 2, got {len(pool)}"
        )
    if len(set(pool)) != len(pool):
        raise ValueError(f"candidate pool {pool} has duplicate ranks")
    hosts = tuple(_host_of(host_by_rank, r) for r in pool)
    offset, partners = choose_offset(hosts)
    return PartnerPlan(
        pool_ranks=pool,
        hosts=hosts,
        offset=offset,
        partner_pos=tuple(partners),
    )


def plan_pools(
    pools: Iterable[Sequence[int]],
    host_by_rank: Mapping[int, str] | Sequence[str],
) -> dict[tuple[int, ...], PartnerPlan]:
    """Plan every distinct pool (used by gemini, which must know EVERY
    pair to build the 2-rank P2P process groups — new_group is a world
    collective). Deterministic iteration order."""
    unique = sorted({tuple(int(r) for r in pool) for pool in pools})
    return {pool: plan_pairing(pool, host_by_rank) for pool in unique}


def all_pairs(
    plans: Mapping[tuple[int, ...], PartnerPlan] | Iterable[PartnerPlan],
) -> list[tuple[int, int]]:
    """Sorted unique global-rank pairs across a set of plans."""
    values = plans.values() if isinstance(plans, Mapping) else plans
    return sorted({pair for plan in values for pair in plan.pairs})


def log_pairing(
    plan: PartnerPlan,
    my_rank: int,
    method: str = "snapshot",
    world_hosts: Iterable[str] | None = None,
    log: logging.Logger | None = None,
) -> None:
    """Announce the pairing once per pool (from the pool's lowest rank) and
    make an unavoidable co-location LOUD.

    ``world_hosts`` (all hosts in the job, if known) only downgrades the
    message for a genuinely single-host job, where cross-host redundancy is
    not a thing that could have been achieved.
    """
    log = log or logger
    if my_rank != min(plan.pool_ranks):
        return
    detail = (
        f"pool={list(plan.pool_ranks)} hosts={list(plan.hosts)} "
        f"offset={plan.offset} pairs={[list(p) for p in plan.pairs]}"
    )
    if plan.fully_cross_host:
        log.info(
            "[%s/partner] all %d pair(s) are cross-host: %s",
            method, plan.total_pairs, detail,
        )
        return
    single_host_world = (
        world_hosts is not None and len(set(world_hosts)) <= 1
    )
    if single_host_world:
        log.info(
            "[%s/partner] single-host job: pair(s) %s are necessarily "
            "co-located (no cross-host redundancy is possible here): %s",
            method, [list(p) for p in plan.colocated_pairs], detail,
        )
        return
    log.error(
        "[%s/partner] NO pairing of this candidate pool makes every pair "
        "cross-host — pair(s) %s keep BOTH checkpoint replicas on one host "
        "(%s). A host-fatal on that host destroys both copies and recovery "
        "will fall back to a fresh start; only process-local (transient) "
        "faults are covered for those ranks. This is a property of the "
        "parallelism layout, not of the pairing: the candidate pool spans "
        "too few hosts (%d/%d pairs cross-host at best). Use a layout whose "
        "FSDP mesh spans hosts. Proceeding with the best available pairing: "
        "%s",
        method,
        [list(p) for p in plan.colocated_pairs],
        sorted({plan.hosts[plan.position_of(r)] for p in plan.colocated_pairs
                for r in p}),
        plan.cross_host_pairs,
        plan.total_pairs,
        detail,
    )
