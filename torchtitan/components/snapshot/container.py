"""Generic kill-survivable snapshot container.

A per-rank spawned service process that co-owns tmpfs-backed shm pools with
the training process (see pool_shm): it attaches to each pool by path inside
the synchronous REGISTER, after which the trainer unlinks the path, leaving an
anonymous kernel-refcounted mapping that survives SIGKILL of the trainer —
including ranks hung in NCCL collectives. The container registers with leto's
WorkerController as a service process; on SERVICE_ACTION_PERSIST it hands
every *committed* state to its DumpPolicy to be written out and exits, on
SERVICE_ACTION_CLOSE it exits immediately.

Checkpoint-method policy — which states must pair up at commit, dump file
names/formats, what a commit-ledger entry means — lives entirely in the
DumpPolicy passed at construction; the container core is method-agnostic.
Gemini's policy is GeminiDumpPolicy in
torchtitan.components.gemini.snapshot_container.
"""

import logging
import os
import queue
import signal
import time
import gc

import torch
import torch.multiprocessing as mp

from torchtitan.components.snapshot import pool_shm

# Leto integration for graceful drain coordination
try:
    from leto.launch.worker_controller_client import (
        register_service_process,
        poll_service_action,
        get_process_group_id,
        SERVICE_ACTION_PERSIST,
        SERVICE_ACTION_CLOSE,
        SERVICE_ACTION_WORKING,
    )
    _LETO_AVAILABLE = True
except ImportError:
    _LETO_AVAILABLE = False
    register_service_process = None
    poll_service_action = None
    get_process_group_id = None
    SERVICE_ACTION_PERSIST = None
    SERVICE_ACTION_CLOSE = None
    SERVICE_ACTION_WORKING = None

_process_states = None
logger = logging.getLogger(__name__)


def _setup_subprocess_logger(log_dir: str, rank: int) -> None:
    """Configure logging for the subprocess, writing to a file in log_dir."""
    group_id = get_process_group_id() if _LETO_AVAILABLE else 0
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"snapshot_container_rank{rank}_group{group_id}.log")
    handler = logging.FileHandler(log_path, mode='w')
    handler.setFormatter(logging.Formatter(
        f'%(asctime)s [SnapshotContainer rank={rank} group={group_id}] %(levelname)s - %(message)s'
    ))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


class StateView:
    """A registered state inside the container subprocess: an attached shm
    pool plus the opaque layout the trainer registered for it and the
    (pickled-in) CPU metadata blob."""

    def __init__(self, state_key, layout, pool_storage: torch.UntypedStorage):
        self.state_key = state_key
        self.layout = layout
        self.pool_storage = pool_storage
        self.cpu_metadata = None

    def pool_bytes(self) -> torch.Tensor:
        # Zero-copy uint8 view over the whole pool. Saving a single byte view
        # + layout metadata avoids torch.save's "Cannot save multiple tensors
        # that view the same data as different types" error on mixed-dtype
        # pools.
        pool_view = torch.empty(0, dtype=torch.uint8)
        pool_view.set_(source=self.pool_storage, storage_offset=0,
                       size=(self.pool_storage.nbytes(),))
        return pool_view


class DumpPolicy:
    """Method-specific policy, executed inside the container subprocess.
    Must be picklable (it crosses the spawn boundary at construction)."""

    def on_commit(self, states: dict, ledger: dict, commit_key) -> None:
        """Validation hook run before a COMMIT is acknowledged."""

    def dump(self, states: dict, ledger: dict) -> None:
        """Persist every committed state. states: {state_key: StateView};
        ledger: {commit_key: metadata}."""
        raise NotImplementedError


class SnapshotContainer:

    def __init__(
        self,
        dump_policy: DumpPolicy,
        log_dir: str,
        rank: int,
    ):
        self.ctx = mp.get_context("spawn")
        self.closed = False
        self.input_queue = self.ctx.Queue()   # parent -> child
        self.output_queue = self.ctx.Queue()  # child -> parent

        self.process = self.ctx.Process(
            target=SnapshotContainer._subprocess_main,
            args=(
                dump_policy,
                log_dir,
                rank,
                self.input_queue,
                self.output_queue,
            ),
        )
        self.process.start()

        # Wait for init
        self._recv_ack("INIT_COMPLETE")

    def _recv_ack(self, expected):
        """Blocking ack wait with a liveness check.

        A healthy container acks in ms, so this behaves like a plain
        blocking get. But the container may exit under us on the worker
        controller's instruction (PERSIST/CLOSE — the same race close()
        defends against): an unbounded get() would then wedge the calling
        thread forever, and with the moevement engine that thread may hold
        the container lock the trainer blocks on next. Poll + is_alive
        turns that wedge into an exception.
        """
        while True:
            try:
                response = self.output_queue.get(timeout=0.5)
            except queue.Empty:
                if not self.process.is_alive():
                    raise RuntimeError(
                        f"SnapshotContainer exited before acknowledging "
                        f"{expected!r}"
                    )
                continue
            assert response == expected, f"Expected {expected!r}, got {response!r}"
            return

    def register(self, state_key, layout, pool_share_info: tuple):
        self.input_queue.put(('REGISTER', {
            'state_key': state_key,
            'layout': layout,
            'pool_share_info': pool_share_info,
        }))

        # Wait for acknowledgment
        self._recv_ack(('REGISTER_DONE', state_key))

    def snapshot_cpu_metadata(self, state_key, cpu_metadata):
        self.input_queue.put(('SNAPSHOT_METADATA', {
            'state_key': state_key,
            'cpu_metadata': cpu_metadata,  # This gets pickled and copied
        }))

        self._recv_ack(('SNAPSHOT_METADATA_DONE', state_key))

    def invalidate(self, commit_key):
        """Drop a ledger entry (its states are about to be overwritten)."""
        self.input_queue.put(('INVALIDATE', {'commit_key': commit_key}))
        self._recv_ack(('INVALIDATE', commit_key))

    def commit(self, commit_key, metadata):
        self.input_queue.put(('COMMIT', {
            'commit_key': commit_key,
            'metadata': metadata,
        }))
        self._recv_ack(('COMMIT', commit_key))

    # Longest we will wait for a CLOSE ack from a container that is still
    # alive. Only a wedged container reaches this; a healthy one acks in ms.
    CLOSE_ACK_TIMEOUT_S = 30.0

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.input_queue.put(('CLOSE', {}))

        # The container may have ALREADY exited under us: its poll loop breaks
        # and os._exit()s the moment the WorkerController hands it PERSIST or
        # CLOSE (see _subprocess_main), which races this call and leaves nobody
        # to ack. An unbounded get() here wedges the training process forever --
        # observed as a rank stuck in close() with a zombie container child,
        # holding up the whole run's teardown. So wait for the ack, but stop as
        # soon as the child is gone or the timeout expires.
        deadline = time.monotonic() + self.CLOSE_ACK_TIMEOUT_S
        while True:
            try:
                response = self.output_queue.get(timeout=0.5)
            except queue.Empty:
                if not self.process.is_alive():
                    # Exited on the controller's instruction. Drain once more in
                    # case it acked just before dying, then stop waiting.
                    try:
                        self.output_queue.get(timeout=0.2)
                    except Exception:
                        pass
                    logger.info(
                        "SnapshotContainer already exited (controller-driven "
                        "PERSIST/CLOSE); no CLOSE ack to wait for"
                    )
                    return
                if time.monotonic() >= deadline:
                    logger.warning(
                        "Timed out after %.0fs waiting for the SnapshotContainer "
                        "CLOSE ack; container still alive (pid=%s). Continuing "
                        "shutdown.", self.CLOSE_ACK_TIMEOUT_S, self.process.pid
                    )
                    return
                continue
            except (EOFError, OSError) as e:
                logger.info("SnapshotContainer queue closed during CLOSE (%s)", e)
                return
            if response == 'CLOSE':
                return
            # A late ack for an earlier command (REGISTER/COMMIT/...); skip it
            # and keep waiting for ours.
            logger.warning("Ignoring unexpected response while closing: %r", response)

    @staticmethod
    def _subprocess_main(
        dump_policy: DumpPolicy,
        log_dir: str,
        rank: int,
        input_queue,
        output_queue,
    ):
        """Subprocess entry point with command handling and worker controller polling"""
        global _process_states

        _process_states = {
            "states": {},
            "ledger": {},  # {commit_key: metadata}
            "rank": rank,
            "dump_policy": dump_policy,
        }

        _setup_subprocess_logger(log_dir, rank)

        # Ignore SIGTERM (we handle graceful shutdown via WorkerController polling)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        # Register with WorkerController for graceful shutdown coordination
        if _LETO_AVAILABLE:
            register_service_process(_process_states["rank"])
        else:
            logger.info("leto package not available, skipping WorkerController registration")

        output_queue.put("INIT_COMPLETE")

        # Main loop: poll input queue with timeout, check worker controller on timeout
        while True:
            try:
                msg = input_queue.get(timeout=0.5)
            except queue.Empty:
                # No command from training process — poll worker controller
                if _LETO_AVAILABLE:
                    try:
                        action = poll_service_action()
                    except Exception as e:
                        logger.warning(f"poll_service_action() failed: {e}")
                        logger.info("Falling back to poll-until-done loop")
                        SnapshotContainer._poll_worker_controller_until_done()
                        break
                    if action == SERVICE_ACTION_PERSIST:
                        logger.info("Worker controller requested PERSIST")
                        SnapshotContainer._dump_states()
                        break
                    elif action == SERVICE_ACTION_CLOSE:
                        logger.info("Worker controller requested CLOSE")
                        break
                # SERVICE_ACTION_WORKING or no leto — continue polling
                continue
            except Exception as e:
                # Queue broken — training process likely dead
                logger.warning(f"Queue exception (training process likely dead): {e}")
                SnapshotContainer._poll_worker_controller_until_done()
                break

            if not isinstance(msg, tuple) or len(msg) != 2:
                continue

            cmd, data = msg

            if cmd == 'REGISTER':
                # Attach to the self-managed shm pool by path (see pool_shm). The
                # training process unlinks the path right after this synchronous
                # REGISTER, so this mapping keeps the (now anonymous) pool alive.
                pool_path, pool_nbytes = data['pool_share_info']
                pool_storage = pool_shm.attach(pool_path, pool_nbytes)
                view = StateView(
                    state_key=data['state_key'],
                    layout=data['layout'],
                    pool_storage=pool_storage,
                )
                _process_states["states"][data['state_key']] = view
                output_queue.put(('REGISTER_DONE', data['state_key']))
            elif cmd == 'SNAPSHOT_METADATA':
                state_key = data['state_key']
                _process_states["states"][state_key].cpu_metadata = data['cpu_metadata']
                output_queue.put(('SNAPSHOT_METADATA_DONE', state_key))
            elif cmd == 'INVALIDATE':
                commit_key = data['commit_key']
                _process_states['ledger'].pop(commit_key, None)
                output_queue.put(('INVALIDATE', commit_key))
            elif cmd == 'COMMIT':
                commit_key = data['commit_key']
                _process_states['ledger'][commit_key] = data['metadata']
                dump_policy.on_commit(
                    _process_states["states"], _process_states['ledger'], commit_key
                )
                output_queue.put(('COMMIT', commit_key))
            elif cmd == 'CLOSE':
                output_queue.put('CLOSE')
                logger.info("Received CLOSE command")
                break

        # The container's work is done: on PERSIST, _dump_states() has already
        # flushed every checkpoint file (torch.save + metadata json all closed)
        # to mem_fs synchronously above; on CLOSE there is nothing to keep. A
        # normal return would hand control to the multiprocessing-spawn wrapper
        # whose interpreter teardown (gc of the attached shm pool view, torch
        # atexit, etc.) measured ~1.8s per rank — and the promotion path waits
        # for THIS process to die before activating the standby, so that 1.8s
        # lands directly on fault-recovery latency. The data is already durable,
        # so exit immediately and let the kernel reclaim the mappings.
        for h in list(logging.getLogger().handlers):
            try:
                h.flush()
            except Exception:
                pass
        os._exit(0)

    @staticmethod
    def _poll_worker_controller_until_done():
        """Training process is gone. Poll worker controller for instructions."""
        if not _LETO_AVAILABLE:
            # No leto available — fallback: persist
            logger.info("No leto available, persisting as fallback")
            SnapshotContainer._dump_states()
            return

        while True:
            try:
                action = poll_service_action()
            except:
                logger.error("Error while polling worker controller, exiting")
                return
            if action == SERVICE_ACTION_PERSIST:
                logger.info("Worker controller requested PERSIST (WC-only mode)")
                SnapshotContainer._dump_states()
                return
            elif action == SERVICE_ACTION_CLOSE:
                logger.info("Worker controller requested CLOSE (WC-only mode)")
                return
            # SERVICE_ACTION_WORKING — keep polling
            time.sleep(0.5)

    @staticmethod
    def _dump_states():
        global _process_states
        ledger = _process_states.get('ledger', {})
        if not ledger:
            logger.info("Nothing to dump.")
            return

        dump_policy = _process_states["dump_policy"]

        logger.info(f"Start dumping {len(ledger)} committed entrie(s): {ledger}")
        persist_start_time = time.time()

        states = _process_states["states"]
        del _process_states["states"]
        gc.collect()

        dump_policy.dump(states, ledger)

        logger.info(f"Persist took {time.time() - persist_start_time:.2f} seconds.")
        logger.info("End dumping")
