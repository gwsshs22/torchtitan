import json
import logging
import os
import queue
import signal
import time
import gc
from typing import Dict, List, Any, Optional

import torch
import torch.multiprocessing as mp

from torchtitan.components.gemini.utils import InMemStateType

# Leto integration for checkpoint timing reporting
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


class InMemStateView:
    """Holds references to shared memory pool in subprocess"""

    def __init__(
        self,
        state_id: int,
        state_type: str,
        model_keys: List[str],
        model_metadata: Dict[str, dict],
        optim_keys: List[str],
        optim_metadata: Dict[str, dict],
        pool_storage: torch.UntypedStorage,
    ):
        self.state_id = state_id
        self.state_type = state_type

        self.model_keys = model_keys
        self.model_metadata = model_metadata

        self.optim_keys = optim_keys
        self.optim_metadata = optim_metadata

        self.pool_storage = pool_storage

        # CPU metadata (set later via snapshot_cpu_metadata)
        self.cpu_metadata = None

    def dump(self, path):
        # Save pool as a single uint8 view + metadata to avoid torch.save's
        # "Cannot save multiple tensors that view the same data as
        # different types" error when pool has mixed dtypes (e.g. bfloat16
        # model params + float32 optimizer states).
        # pool_bytes is a zero-copy view into pool_storage — no extra memory.
        pool_bytes = torch.empty(0, dtype=torch.uint8)
        pool_bytes.set_(source=self.pool_storage, storage_offset=0,
                        size=(self.pool_storage.nbytes(),))

        state = {
            "_model_tensor_keys": self.model_keys,
            "_optim_tensor_keys": self.optim_keys,
            "_cpu_metadata": self.cpu_metadata,
            "_pool_bytes": pool_bytes,
            "_model_metadata": self.model_metadata,
            "_optim_metadata": self.optim_metadata,
        }

        torch.save(state, path)


class SnapshotContainer:

    def __init__(
        self,
        checkpoint_dir: str,
        log_dir: str,
        rank: int
    ):
        self.ctx = mp.get_context("spawn")
        self.closed = False
        self.input_queue = self.ctx.Queue()   # parent -> child
        self.output_queue = self.ctx.Queue()  # child -> parent

        self.process = self.ctx.Process(
            target=SnapshotContainer._subprocess_main,
            args=(
                checkpoint_dir,
                log_dir,
                rank,
                self.input_queue,
                self.output_queue,
            ),
        )
        self.process.start()

        # Wait for init
        response = self.output_queue.get()
        assert response == "INIT_COMPLETE"

    def register(
        self,
        state_id: int,
        in_mem_state_type: InMemStateType,
        model_tensor_keys: List[str],
        model_tensor_metadata: Dict[str, dict],
        optim_tensor_keys: List[str],
        optim_tensor_metadata: Dict[str, dict],
        pool_share_info: tuple,
    ):
        self.input_queue.put(('REGISTER', {
            'state_id': state_id,
            'state_type': in_mem_state_type,
            'model_keys': model_tensor_keys,
            'model_metadata': model_tensor_metadata,
            'optim_keys': optim_tensor_keys,
            'optim_metadata': optim_tensor_metadata,
            'pool_share_info': pool_share_info,
        }))

        # Wait for acknowledgment
        response = self.output_queue.get()
        assert response == ('REGISTER_DONE', state_id), f"Expected REGISTER_DONE, got {response}"

    def snapshot_cpu_metadata(
        self,
        state_id: int,
        in_mem_state_type: InMemStateType,
        cpu_metadata: Any,
    ):
        self.input_queue.put(('SNAPSHOT_METADATA', {
            'state_id': state_id,
            'state_type': in_mem_state_type,
            'cpu_metadata': cpu_metadata,  # This gets pickled and copied
        }))

        response = self.output_queue.get()
        assert response == ('SNAPSHOT_METADATA_DONE', state_id)

    def invalidate(self, state_id: int):
        """Mark a version as invalid (about to be overwritten)."""
        self.input_queue.put(('INVALIDATE', {'state_id': state_id}))
        response = self.output_queue.get()
        assert response == ('INVALIDATE', state_id)

    def commit(self, state_id: int, snapshot_step: int):
        self.input_queue.put(('COMMIT', {
            'state_id': state_id,
            'snapshot_step': snapshot_step
        }))
        response = self.output_queue.get()
        assert response == ('COMMIT', state_id)

    def close(self):
        if not self.closed:
            self.closed = True
            self.input_queue.put(('CLOSE', {}))
            response = self.output_queue.get()
            assert response == 'CLOSE'

    @staticmethod
    def _subprocess_main(
        checkpoint_dir: str,
        log_dir: str,
        rank: int,
        input_queue,
        output_queue,
    ):
        """Subprocess entry point with command handling and worker controller polling"""
        global _process_states

        _process_states = {
            "in_mem_states": {},
            "version_steps": {},  # {version_id: step}
            "rank": rank,
            "checkpoint_dir": checkpoint_dir,
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
                # Reconstruct pool storage from share info
                pool_storage = torch.UntypedStorage._new_shared_filename_cpu(
                    *data['pool_share_info']
                )
                view = InMemStateView(
                    state_id=data['state_id'],
                    state_type=data['state_type'],
                    model_keys=data['model_keys'],
                    model_metadata=data['model_metadata'],
                    optim_keys=data['optim_keys'],
                    optim_metadata=data['optim_metadata'],
                    pool_storage=pool_storage,
                )
                _process_states["in_mem_states"][(data['state_id'], data['state_type'])] = view
                output_queue.put(('REGISTER_DONE', data['state_id']))
            elif cmd == 'SNAPSHOT_METADATA':
                state_id = data['state_id']
                state_type = data['state_type']
                _process_states["in_mem_states"][(state_id, state_type)].cpu_metadata = data['cpu_metadata']
                output_queue.put(('SNAPSHOT_METADATA_DONE', state_id))
            elif cmd == 'INVALIDATE':
                state_id = data['state_id']
                _process_states['version_steps'].pop(state_id, None)
                output_queue.put(('INVALIDATE', state_id))
            elif cmd == 'COMMIT':
                state_id = data['state_id']
                snapshot_step = data['snapshot_step']
                _process_states['version_steps'][state_id] = snapshot_step
                assert ((state_id, InMemStateType.LOCAL) in _process_states["in_mem_states"])
                assert ((state_id, InMemStateType.REMOTE) in _process_states["in_mem_states"])
                output_queue.put(('COMMIT', state_id))
            elif cmd == 'CLOSE':
                output_queue.put('CLOSE')
                logger.info("Received CLOSE command")
                break

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
        version_steps = _process_states.get('version_steps', {})
        if not version_steps:
            logger.info("Nothing to dump.")
            return

        checkpoint_dir = _process_states["checkpoint_dir"]
        rank = _process_states["rank"]

        logger.info(f"Start dumping {len(version_steps)} version(s): {version_steps}")
        persist_start_time = time.time()

        in_mem_states = _process_states["in_mem_states"]
        del _process_states["in_mem_states"]
        gc.collect()

        for version, step in version_steps.items():
            local_view = in_mem_states[(version, InMemStateType.LOCAL)]
            remote_view = in_mem_states[(version, InMemStateType.REMOTE)]
            local_path = os.path.join(checkpoint_dir, f"rank_{rank}_v{version}_local.pt")
            remote_path = os.path.join(checkpoint_dir, f"rank_{rank}_v{version}_remote.pt")
            local_view.dump(local_path)
            remote_view.dump(remote_path)
            logger.info(f"Dumped version {version} (step {step})")

        # Write metadata file (index-aligned: list[version] = step)
        metadata = {"version_steps": [version_steps.get(0), version_steps.get(1)]}
        metadata_path = os.path.join(checkpoint_dir, f"rank_{rank}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        logger.info(f"Wrote metadata: {metadata}")

        logger.info(f"Persist took {time.time() - persist_start_time:.2f} seconds.")
        logger.info("End dumping")
