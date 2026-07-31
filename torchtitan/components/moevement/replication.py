"""MoEvement window replication + uniform faulty-rank recovery (plan §3.7).

Replication (always on when enabled and the candidate pool is evenly
pairable): each window is shipped CPU->CPU to the pair rank chosen by the
SHARED partner rule (``torchtitan.components.snapshot.partner``, the same
rule gemini uses: partner drawn from this rank's FSDP process group — same
pipeline stage, hence identically shaped state — with the pairing offset
inside that pool chosen to maximize cross-host pairs) over DEDICATED gloo
groups by a small sender
thread — off the training path, never gap-interleaved. The transport is
PER-ITERATION streaming: as the engine's committer confirms each iteration's
event-synced pool span, that span's bytes are enqueued and sent; the
window's final commit then ships only the (small) full metadata blob
(headers, per-iter RNG, ring entry, schedule; torch.save). Streaming — the
reference's own per-iteration replication cadence — keeps the pair's copy
complete within ~ms of the owner's final commit; a single end-of-window bulk
send would leave a multi-second in-flight gap during which a fault strands
the cluster without any common window (survivors drop W-1 at the W+1
boundary while W's replica is still on the wire).

The receiver thread lands the bytes at the SAME offsets in one of two
'remote' shm slots (allocation DEFERRED to ensure_slots — after load()'s
dump read+sweep, or on the first incoming message; never pinned — no CUDA
DMA ever touches them, so the M4 mid-training cudaHostRegister caveat cannot
bite), registered with the shared SnapshotContainer as state keys
"remote0"/"remote1", and commits the copy under key ``r_w{start}`` only when
the final metadata has arrived (invalidate-before-overwrite for torn-write
safety). The slot mirrors the OWNER's pool index, so the receiver's two
slots hold exactly the two windows the owner's own double buffer holds.
Upstream-log rings are NOT replicated (plan §3.7 — faulty ranks replay over
live p2p).

Uniform recovery (both fault types, user decision D3): the faulty rank
NEVER trusts its own dump. ``MoevementCheckpointManager.load()`` first runs a
cross-rank window-agreement vote — an all_reduce(MIN) over a dedicated gloo
group of each rank's "newest restorable window_start" contribution (see
``vote_contribution``) — then every rank restores the SAME agreed window:
non-faulty ranks from their local dumps; faulty ranks by fetching the agreed
window's bundle from their pair (which serves its dumped remote copy) via the
same header/meta/bytes streaming protocol.

Group hygiene: three dedicated gloo groups are built once at lazy_init (a
collective — all ranks construct them in the same order):
  - one per pair direction (low->high and high->low), each touched by exactly
    ONE background thread per process (gloo PGs are not assumed thread-safe);
  - a 'recovery' group used only by the trainer thread (init size exchange,
    the vote, and the recovery fetch/serve).
"""

import io
import logging
import os
import queue
import threading
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from torchtitan.components.moevement.conversion import window_used_nbytes
from torchtitan.components.snapshot import partner, pool_shm

logger = logging.getLogger(__name__)

# Vote sentinels (all_reduce MIN over int64):
#   VOTE_NO_CONSTRAINT — "I restore nothing locally and rely on peer fetch"
#     (faulty ranks). MIN ignores it unless everyone contributes it.
#   VOTE_FRESH_START (0) — "I cannot restore anything and cannot be served"
#     (non-faulty rank with no usable dump; pair-of-faulty with no remote
#     copy; both members of a pair faulty). Forces a world-wide fresh start:
#     window_start is always >= 1, so MIN lands on 0.
VOTE_NO_CONSTRAINT = 1 << 30
VOTE_FRESH_START = 0

_SEND_QUEUE_MAX = 32  # per-iter spans + finals; backpressure via put()
_STOP = object()

# Wire message kinds (first int64 of every 5-int64 header)
_MSG_ITER = 0   # header: [ITER, window_start, pool_idx, span_offset, span_nbytes]
_MSG_FINAL = 1  # header: [FINAL, window_start, pool_idx, meta_nbytes, unused]
_MSG_BYE = 2    # header: [BYE, 0, 0, 0, 0] — sender's close(); ends the peer's receiver


def iter_span(header: dict[str, Any]) -> tuple[int, int]:
    """(offset, nbytes) of the contiguous pool span one iteration's capture
    occupies (min..max over its committed tensor entries; alignment gaps
    inside the span ride along, harmlessly)."""
    lo, hi = None, 0
    for op_meta in header.values():
        for tmeta in op_meta["tensors"].values():
            lo = tmeta["offset"] if lo is None else min(lo, tmeta["offset"])
            hi = max(hi, tmeta["offset"] + tmeta["nbytes"])
    if lo is None:
        return 0, 0
    return lo, hi - lo


def pair_rank(rank: int, world: int) -> int:
    """LEGACY buddy topology over WORLD ranks: (rank + world//2) % world.

    Superseded by the shared rule in ``torchtitan.components.snapshot.partner``
    (mutual, same-pipeline-stage candidate pool, cross-host-maximizing),
    which build_replicator now uses. Kept only as the degenerate fallback
    for a pool that IS the whole world with no host information, and because
    it is the identity this module's tests pin at world=2."""
    return (rank + world // 2) % world


def vote_constraint(
    own: "set[int] | list[int]",
    remote: "set[int] | list[int]",
    self_faulty: bool,
    pair_faulty: bool,
) -> "set[int] | None":
    """The SET of windows this rank can supply, or None = unconstrained.

    Replaces the earlier "contribute your newest, take the world MIN"
    scheme, which could agree on a window that some rank never held: with
    newest-first dump priority a pressured rank keeps only its newest, so
    MIN(newest) may name a window an unpressured rank already dropped.
    Observed live — "rank 1 must serve window w104 to faulty pair rank 5,
    but no remote dump exists". Agreeing on the newest window in the
    INTERSECTION of what every rank can supply removes the failure mode by
    construction.

    - Faulty rank: unconstrained (its state comes from the peer fetch) —
      unless its pair is also faulty, in which case nobody can serve it and
      it forces a fresh start (empty set).
    - Non-faulty rank: the windows it can restore locally; if its pair is
      faulty it must ALSO be able to SERVE the agreed window, so intersect
      with the remote copies it holds for that pair.
    """
    if self_faulty:
        return set() if pair_faulty else None
    c = set(own)
    if pair_faulty:
        c &= set(remote)
    return c


def agreed_window(constraints: "list[set[int] | None]") -> int | None:
    """Newest window every constrained rank can supply; None -> fresh start."""
    constrained = [c for c in constraints if c is not None]
    if not constrained:
        return None
    inter = set.intersection(*(set(c) for c in constrained))
    return max(inter) if inter else None


def serialize_meta(meta: dict[str, Any]) -> torch.Tensor:
    """Window metadata -> uint8 tensor (torch.save; carries per-iter RNG
    tensors and the ring entry)."""
    buf = io.BytesIO()
    torch.save(meta, buf)
    return torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)


def deserialize_meta(data: torch.Tensor) -> dict[str, Any]:
    return torch.load(
        io.BytesIO(data.numpy().tobytes()),
        map_location="cpu",
        weights_only=False,
    )


def _storage_view(storage, offset: int, nbytes: int) -> torch.Tensor:
    view = torch.empty(0, dtype=torch.uint8)
    view.set_(source=storage, storage_offset=offset, size=(nbytes,))
    return view


def _storage_prefix_view(storage, nbytes: int) -> torch.Tensor:
    return _storage_view(storage, 0, nbytes)


def send_window_blob(
    meta: dict[str, Any],
    pool_prefix: torch.Tensor,
    pool_idx: int,
    dst: int,
    group,
) -> None:
    """Ship one window: int64 header [window_start, pool_idx, meta_nbytes,
    used_nbytes], then meta bytes, then the pool prefix."""
    meta_bytes = serialize_meta(meta)
    header = torch.tensor(
        [
            int(meta["window_start"]),
            int(pool_idx),
            meta_bytes.numel(),
            pool_prefix.numel(),
        ],
        dtype=torch.int64,
    )
    dist.send(header, group_dst=dst, group=group)
    dist.send(meta_bytes, group_dst=dst, group=group)
    if pool_prefix.numel() > 0:
        dist.send(pool_prefix, group_dst=dst, group=group)


def recv_window_header(src: int, group) -> tuple[int, int, int, int]:
    header = torch.zeros(4, dtype=torch.int64)
    dist.recv(header, group_src=src, group=group)
    ws, pool_idx, meta_n, used_n = (int(x) for x in header)
    return ws, pool_idx, meta_n, used_n


def recv_window_body(
    meta_nbytes: int, used_nbytes: int, dst_pool_view: torch.Tensor,
    src: int, group,
) -> dict[str, Any]:
    """Receive meta + pool bytes (pool bytes land directly in
    ``dst_pool_view``, a uint8 view of size used_nbytes). Returns meta."""
    meta_bytes = torch.zeros(meta_nbytes, dtype=torch.uint8)
    dist.recv(meta_bytes, group_src=src, group=group)
    if used_nbytes > 0:
        assert dst_pool_view.numel() == used_nbytes
        dist.recv(dst_pool_view, group_src=src, group=group)
    return deserialize_meta(meta_bytes)


class WindowReplicator:
    """Owns the remote slots, the sender/receiver threads, and the
    trainer-thread recovery surface (vote / fetch / serve).

    ``engine`` must expose: ``pool_nbytes``, ``pool_storage(idx)``,
    ``register_extra_pool``, ``commit_key``, ``invalidate_key`` (the
    MoevementSnapshotEngine surface; tests substitute a stub).
    """

    def __init__(
        self,
        engine,
        rank: int,
        world: int,
        mem_fs_folder: str,
        send_group,
        recv_group,
        recovery_group,
        remote_pool_nbytes: int,
        pair: int | None = None,
    ):
        self._engine = engine
        self._rank = rank
        self._world = world
        # The partner comes from the shared rule (build_replicator ->
        # snapshot.partner.plan_pairing); the legacy world-offset is only
        # the fallback for callers that pass no pairing.
        self.pair = pair_rank(rank, world) if pair is None else int(pair)
        self._send_group = send_group
        self._recv_group = recv_group
        self.recovery_group = recovery_group
        self._mem_fs_folder = mem_fs_folder

        self._queue: queue.Queue = queue.Queue(maxsize=_SEND_QUEUE_MAX)
        self._sent_events: dict[str, threading.Event] = {}
        self._sent_lock = threading.Lock()
        self._sender_error: BaseException | None = None
        self._receiver_error: BaseException | None = None

        # Remote slots: DEFERRED like the engine's own window pools (M7
        # structural fix — allocating them at lazy_init, before load() has
        # read+swept the previous attempt's dumps, contributed to the
        # deepseek /dev/shm overflow). ensure_slots() allocates them —
        # invoked from the manager's ensure path (load() after the sweep /
        # first save / promotion init) or lazily by the receiver thread on
        # the first incoming message, whichever runs first. Unpinned
        # (CPU-only traffic: gloo recv in, container dump out) — no CUDA DMA
        # ever touches them, so allocation needs no init-time/fence
        # guarantee (the M4 caveat is about cudaHostRegister only) and is
        # safe on the receiver thread. register->unlink keeps the
        # container's mapping kill-survivable, unchanged. Sized from the
        # PAIR's exchanged pool size — under the shared same-stage partner
        # rule that is normally identical to this rank's own, but the
        # exchange stays authoritative (uneven expert/layer splits, and the
        # pre-unification cross-stage pairing, produce different sizes).
        self.remote_nbytes = remote_pool_nbytes
        self._remote_pools: list = []
        self._remote_slot_keys: list[str | None] = [None, None]
        self._slots_lock = threading.Lock()
        self._slots_ready = threading.Event()
        logger.info(
            "[moevement] replication armed: rank %d <-> pair %d, remote "
            "slots 2 x %.2f MB (unpinned shm, deferred)",
            rank, self.pair, remote_pool_nbytes / (1024 * 1024),
        )

        self._sender = threading.Thread(
            target=self._sender_main,
            name=f"moevement-repl-send-r{rank}",
            daemon=True,
        )
        self._receiver = threading.Thread(
            target=self._receiver_main,
            name=f"moevement-repl-recv-r{rank}",
            daemon=True,
        )
        self._sender.start()
        self._receiver.start()

    def ensure_slots(self) -> None:
        """Idempotent deferred allocation of the two remote slots + their
        container registration (see __init__). Thread-safe: raced by the
        trainer's ensure path and the receiver's first incoming message —
        whichever wins allocates, the loser returns once the lock frees."""
        with self._slots_lock:
            if self._slots_ready.is_set():
                return
            for idx in range(2):
                path = os.path.join(
                    self._mem_fs_folder,
                    f"moevement_remote_rank{self._rank}_v{idx}_pid{os.getpid()}",
                )
                storage = pool_shm.alloc_unpinned(path, self.remote_nbytes)
                self._engine.register_extra_pool(
                    f"remote{idx}", path, self.remote_nbytes
                )
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
                self._remote_pools.append(storage)
            self._slots_ready.set()
            logger.info(
                "[moevement] remote slots allocated: 2 x %.2f MB (unpinned "
                "shm)", self.remote_nbytes / (1024 * 1024),
            )

    # ------------------------------------------------------------------
    # Committer-facing surface
    # ------------------------------------------------------------------

    def enqueue_iter(
        self, key: str, pool_idx: int, header: dict[str, Any]
    ) -> None:
        """Called by the engine's committer right after one iteration's
        commit: stream that iteration's event-confirmed pool span to the
        pair."""
        if self._sender_error is not None:
            return
        offset, nbytes = iter_span(header)
        self._queue.put(("iter", key, pool_idx, offset, nbytes))

    def enqueue_final(self, key: str, meta: dict[str, Any]) -> None:
        """Called by the engine's committer right after a window's final
        commit: ship the full metadata blob (the pool bytes already streamed
        per iteration). Registers send tracking BEFORE queueing so wait_sent
        always finds the event; the event fires once the FINAL message — and
        therefore, FIFO, every iteration span before it — has been sent."""
        event = threading.Event()
        if self._sender_error is not None:
            # Sender is dead (peer loss mid-fault): don't queue, don't block
            # the barrier.
            event.set()
        with self._sent_lock:
            self._sent_events[key] = event
        if self._sender_error is None:
            self._queue.put(("final", key, meta))

    def replicate_window(
        self, key: str, pool_idx: int, meta: dict[str, Any], used_nbytes: int
    ) -> None:
        """One-shot ship of a re-committed (restored) window to the pair
        (M6 re-commit hardening; engine.recommit_window): a single
        iter-span message covering the window's used pool prefix, then the
        final metadata blob, through the normal sender-thread protocol —
        the pair's receiver lands it in slot ``pool_idx % 2`` and commits
        ``r_w{start}``. Registers send tracking so the begin barrier's
        wait_sent applies when this pool is later overwritten. Trainer
        thread only (the committer is idle during load())."""
        if self._sender_error is not None:
            return
        event = threading.Event()
        with self._sent_lock:
            self._sent_events[key] = event
        self._queue.put(("iter", key, pool_idx, 0, used_nbytes))
        self._queue.put(("final", key, meta))

    def wait_sent(self, key: str, timeout: float) -> None:
        """Boundary barrier: block until window ``key``'s send completed (its
        pool is about to be overwritten). No tracking entry (replication
        armed mid-run / legacy window) -> no wait."""
        with self._sent_lock:
            event = self._sent_events.get(key)
        if event is None:
            return
        if not event.wait(timeout):
            raise RuntimeError(
                f"[moevement] replication send of window {key} did not "
                f"complete within {timeout}s — refusing to overwrite its pool"
            )
        with self._sent_lock:
            self._sent_events.pop(key, None)

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _sender_main(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                break
            try:
                if item[0] == "bye":
                    # close() handshake: tell the pair's receiver to exit so
                    # destroy_process_group never tears down a posted recv.
                    header = torch.tensor(
                        [_MSG_BYE, 0, 0, 0, 0], dtype=torch.int64
                    )
                    dist.send(header, group_dst=self.pair,
                              group=self._send_group)
                    break
                if item[0] == "iter":
                    _kind, key, pool_idx, offset, nbytes = item
                    ws = int(key.removeprefix("w"))
                    header = torch.tensor(
                        [_MSG_ITER, ws, pool_idx, offset, nbytes],
                        dtype=torch.int64,
                    )
                    dist.send(header, group_dst=self.pair,
                              group=self._send_group)
                    if nbytes > 0:
                        span = _storage_view(
                            self._engine.pool_storage(pool_idx),
                            offset, nbytes,
                        )
                        dist.send(span, group_dst=self.pair,
                                  group=self._send_group)
                else:
                    _kind, key, meta = item
                    ws = int(meta["window_start"])
                    pool_idx = int(meta["pool"].removeprefix("pool"))
                    meta_bytes = serialize_meta(meta)
                    header = torch.tensor(
                        [_MSG_FINAL, ws, pool_idx, meta_bytes.numel(), 0],
                        dtype=torch.int64,
                    )
                    dist.send(header, group_dst=self.pair,
                              group=self._send_group)
                    dist.send(meta_bytes, group_dst=self.pair,
                              group=self._send_group)
                    with self._sent_lock:
                        evt = self._sent_events.get(key)
                    if evt is not None:
                        evt.set()
                    logger.info(
                        "[moevement] replicated window %s to pair rank %d "
                        "(%.2f MB streamed + metadata)",
                        key, self.pair,
                        window_used_nbytes(meta) / (1024 * 1024),
                    )
            except BaseException as e:
                # Peer death mid-fault is the expected trigger; the whole
                # cluster is being torn down. Unblock any barrier waiter.
                self._sender_error = e
                logger.warning(
                    "[moevement] replication sender stopped (%s): %r",
                    item[1] if len(item) > 1 else item, e,
                )
                with self._sent_lock:
                    for evt in self._sent_events.values():
                        evt.set()
                break

    def _receiver_main(self) -> None:
        filling: list[int | None] = [None, None]  # slot -> window_start
        while True:
            try:
                header = torch.zeros(5, dtype=torch.int64)
                dist.recv(header, group_src=self.pair, group=self._recv_group)
                kind, ws, pool_idx, a, b = (int(x) for x in header)
                if kind == _MSG_BYE:
                    logger.info(
                        "[moevement] replication receiver: pair rank %d said "
                        "BYE; exiting", self.pair,
                    )
                    break
                # Deferred slots: the pair's first span can arrive before
                # this rank's trainer reached its own ensure path (its step
                # cadence lags the pair's) — allocate here in that case
                # (idempotent, CPU-only, thread-safe; see ensure_slots).
                self.ensure_slots()
                slot = pool_idx % 2
                if filling[slot] != ws:
                    # First message of a new window into this slot: drop the
                    # evicted copy's ledger entry BEFORE its bytes are
                    # overwritten (torn-write safety).
                    old_key = self._remote_slot_keys[slot]
                    if old_key is not None:
                        self._engine.invalidate_key(old_key)
                        self._remote_slot_keys[slot] = None
                    filling[slot] = ws
                if kind == _MSG_ITER:
                    offset, nbytes = a, b
                    if offset + nbytes > self.remote_nbytes:
                        raise RuntimeError(
                            f"[moevement] incoming span of remote window "
                            f"w{ws} ({offset}+{nbytes}B) exceeds the remote "
                            f"slot ({self.remote_nbytes}B)"
                        )
                    if nbytes > 0:
                        dst = _storage_view(
                            self._remote_pools[slot], offset, nbytes
                        )
                        dist.recv(dst, group_src=self.pair,
                                  group=self._recv_group)
                else:  # _MSG_FINAL: metadata blob -> the copy becomes durable
                    meta_bytes = torch.zeros(a, dtype=torch.uint8)
                    dist.recv(meta_bytes, group_src=self.pair,
                              group=self._recv_group)
                    meta = deserialize_meta(meta_bytes)
                    # Rewrite the pool reference to the local slot and tag
                    # the copy; commit only now — every span has arrived
                    # (FIFO channel; the sender emits FINAL last).
                    meta["pool"] = f"remote{slot}"
                    meta["kind"] = "remote"
                    meta["owner"] = self.pair
                    r_key = f"r_w{ws}"
                    self._engine.commit_key(r_key, meta)
                    self._remote_slot_keys[slot] = r_key
                    filling[slot] = None
                    logger.info(
                        "[moevement] received remote window %s from pair "
                        "rank %d into slot %d (%.2f MB) — committed",
                        r_key, self.pair, slot,
                        window_used_nbytes(meta) / (1024 * 1024),
                    )
            except BaseException as e:
                self._receiver_error = e
                logger.warning(
                    "[moevement] replication receiver stopped: %r", e
                )
                break

    # ------------------------------------------------------------------
    # Trainer-thread recovery surface (load() path)
    # ------------------------------------------------------------------

    def vote(self, constraint: "set[int] | None") -> "list[set[int] | None]":
        """All-gather every rank's supplyable-window set.

        Sets are tiny (<= a handful of ints), and every rank reduces the
        same gathered list with the same pure function, so the decision is
        world-identical without a second collective."""
        gathered: list = [None] * dist.get_world_size(self.recovery_group)
        dist.all_gather_object(
            gathered,
            None if constraint is None else sorted(constraint),
            group=self.recovery_group,
        )
        return [None if g is None else set(g) for g in gathered]

    def fetch_from_pair(self) -> tuple[dict[str, Any], torch.Tensor]:
        """Faulty-rank side: receive the agreed window's bundle (meta + pool
        prefix bytes) from the pair over the recovery group."""
        ws, _pool_idx, meta_n, used_n = recv_window_header(
            self.pair, self.recovery_group
        )
        pool_bytes = torch.zeros(used_n, dtype=torch.uint8)
        meta = recv_window_body(
            meta_n, used_n, pool_bytes, self.pair, self.recovery_group
        )
        return meta, pool_bytes

    def serve_remote_window(self, window_start: int, rank: int) -> None:
        """Pair-of-faulty side: stream the dumped remote copy of the faulty
        pair's window ``w{window_start}`` (which THIS rank holds) to the pair."""
        path = os.path.join(
            self._mem_fs_folder,
            f"rank_{rank}_moevement_r_w{window_start}.pt",
        )
        if not os.path.exists(path):
            raise RuntimeError(
                f"[moevement] rank {rank} must serve window w{window_start} "
                f"to faulty pair rank {self.pair}, but no remote dump exists "
                f"at {path}"
            )
        saved = torch.load(path, map_location="cpu", weights_only=False)
        meta, pool_bytes = saved["_meta"], saved["_pool_bytes"]
        used = window_used_nbytes(meta)
        send_window_blob(
            meta, pool_bytes[:used], 0, self.pair, self.recovery_group
        )
        logger.info(
            "[moevement] rank %d: served remote window w%d (%.2f MB) to "
            "faulty pair rank %d",
            rank, window_start, used / (1024 * 1024), self.pair,
        )

    def close(self) -> None:
        """Clean shutdown for the NORMAL-completion path (fault paths are
        SIGKILLed and never get here): send BYE to the pair (ends its
        receiver), then join both threads before the trainer destroys the
        process groups — a posted gloo recv surviving into
        destroy_process_group aborts the process (observed in the M5 smoke).
        Every rank closes at end of run, so our receiver gets the pair's BYE
        within the join window; on timeout we log and proceed."""
        try:
            self._queue.put(("bye",), timeout=30.0)
        except queue.Full:
            logger.warning("[moevement] replication close: send queue wedged")
        self._sender.join(timeout=60.0)
        self._receiver.join(timeout=60.0)
        for name, thread in (("sender", self._sender),
                             ("receiver", self._receiver)):
            if thread.is_alive():
                logger.warning(
                    "[moevement] replication %s did not exit before close; "
                    "process-group teardown may abort", name,
                )


def build_replicator(
    engine,
    mem_fs_folder: str,
    replication_enabled: bool,
    pool_ranks: "list[int] | tuple[int, ...] | None" = None,
) -> WindowReplicator | None:
    """Construct the replicator + its three dedicated gloo groups at
    lazy_init time (collective: ALL ranks must call this at the same point,
    in the same order — group construction happens even for ranks that end
    up disabling replication only via the uniform config gate below).

    ``pool_ranks`` is the candidate pool the partner is drawn from — the
    FSDP process group's global ranks (see snapshot.partner: same pipeline
    stage => identically shaped state, and the pairing offset inside the
    pool is chosen to maximize cross-host pairs). None (no FSDP mesh /
    tests) falls back to the whole world, the pre-unification behavior.

    Returns None (with a log line) when replication cannot run: config off,
    no/1-rank world, or a candidate pool that is not evenly pairable.
    """
    if not (dist.is_available() and dist.is_initialized()):
        if replication_enabled:
            logger.info(
                "[moevement] replication disabled: torch.distributed not "
                "initialized"
            )
        return None
    world = dist.get_world_size()
    rank = dist.get_rank()
    if not replication_enabled:
        logger.info("[moevement] replication disabled by config")
        return None
    pool = tuple(int(r) for r in (pool_ranks if pool_ranks else range(world)))
    if world < 2 or not partner.is_pairable(len(pool)):
        logger.info(
            "[moevement] replication disabled: world_size=%d, candidate "
            "pool=%s (needs an even pool >= 2 for the pair topology)",
            world, list(pool),
        )
        return None

    timeout = timedelta(days=7)  # receiver blocks between windows
    send_low_high = dist.new_group(backend="gloo", timeout=timeout)
    send_high_low = dist.new_group(backend="gloo", timeout=timeout)
    recovery = dist.new_group(backend="gloo")

    # ONE init-time all_gather carries the pool size AND the hostname: the
    # remote slots must fit the PAIR's window pool, and the shared partner
    # rule needs each candidate's host. (Same-stage pairing normally makes
    # the two pools equal-sized; the exchange stays authoritative anyway.)
    info = [None] * world
    dist.all_gather_object(
        info,
        (int(engine.pool_nbytes), partner.hostname()),
        group=recovery,
    )
    sizes = [n for n, _host in info]
    host_by_rank = [host for _n, host in info]

    plan = partner.plan_pairing(pool, host_by_rank)
    partner.log_pairing(
        plan, rank, method="moevement", world_hosts=host_by_rank, log=logger
    )
    pair = plan.partner_of(rank)
    remote_nbytes = int(sizes[pair])

    if rank < pair:
        send_group, recv_group = send_low_high, send_high_low
    else:
        send_group, recv_group = send_high_low, send_low_high

    return WindowReplicator(
        engine=engine,
        rank=rank,
        world=world,
        mem_fs_folder=mem_fs_folder,
        send_group=send_group,
        recv_group=recv_group,
        recovery_group=recovery,
        remote_pool_nbytes=remote_nbytes,
        pair=pair,
    )
