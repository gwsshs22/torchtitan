import os
import sys
from pathlib import Path
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
        report_duration,
        DURATION_CHECKPOINT_PERSISTING,
    )
    _LETO_AVAILABLE = True
except ImportError:
    _LETO_AVAILABLE = False
    register_service_process = None
    report_duration = None
    DURATION_CHECKPOINT_PERSISTING = None

_process_states = None

class InMemStateView:
    """Holds references to shared memory tensors in subprocess"""

    def __init__(
        self,
        state_id: int,
        state_type: str,
        model_keys: List[str],
        model_storages: Dict[str, torch.UntypedStorage],
        model_metadata: Dict[str, dict],
        optim_keys: List[str],
        optim_storages: Dict[str, torch.UntypedStorage],
        optim_metadata: Dict[str, dict],
    ):
        self.state_id = state_id
        self.state_type = state_type

        # Model state
        self.model_keys = model_keys
        self.model_storages = model_storages  # Shared memory references
        self.model_metadata = model_metadata

        # Optimizer state
        self.optim_keys = optim_keys
        self.optim_storages = optim_storages  # Shared memory references
        self.optim_metadata = optim_metadata

        # CPU metadata (set later via snapshot_cpu_metadata)
        self.cpu_metadata = None
    
    def dump(self, path):
        model_cpu_tensors = []
        optim_cpu_tensors = []
        state = {
            "_model_tensor_keys": self.model_keys,
            "_model_cpu_tensors": model_cpu_tensors,
            "_optim_tensor_keys": self.optim_keys,
            "_optim_cpu_tensors": optim_cpu_tensors,
            "_cpu_metadata": self.cpu_metadata
        }

        for key in self.model_keys:
            storage = self.model_storages[key]
            meta = self.model_metadata[key]
            tensor = torch.empty(0, dtype=meta['dtype'])
            tensor.set_(
                source=storage,
                storage_offset=meta['storage_offset'],
                size=meta['shape'],
                stride=meta['stride']
            )
            model_cpu_tensors.append(tensor)

        for key in self.optim_keys:
            storage = self.optim_storages[key]
            meta = self.optim_metadata[key]
            tensor = torch.empty(0, dtype=meta['dtype'])
            tensor.set_(
                source=storage,
                storage_offset=meta['storage_offset'],
                size=meta['shape'],
                stride=meta['stride']
            )

            optim_cpu_tensors.append(tensor)

        torch.save(state, path)


class SnapshotContainer:

    def __init__(
        self,
        local_checkpoint_path: str,
        remote_checkpoint_path: str,
        log_file_path: str,
        rank: int
    ):
        self.ctx = mp.get_context("spawn")
        self.closed = False
        self.parent_pipe, child_pipe = self.ctx.Pipe()

        self.process = self.ctx.Process(
            target=SnapshotContainer._subprocess_main,
            args=(
                local_checkpoint_path,
                remote_checkpoint_path,
                log_file_path,
                rank,
                child_pipe),
        )
        self.process.start()
        child_pipe.close()

        # Wait for init
        response = self.parent_pipe.recv()
        assert response == "INIT_COMPLETE"

    def register(
        self,
        state_id: int,
        in_mem_state_type: InMemStateType,
        model_tensor_keys: List[str],
        model_cpu_tensors: List[torch.Tensor],
        optim_tensor_keys: List[str],
        optim_cpu_tensors: List[torch.Tensor],
    ):
        # Validate lengths match
        assert len(model_tensor_keys) == len(model_cpu_tensors), \
            f"Length mismatch: {len(model_tensor_keys)} keys vs {len(model_cpu_tensors)} tensors"
        assert len(optim_tensor_keys) == len(optim_cpu_tensors), \
            f"Length mismatch: {len(optim_tensor_keys)} keys vs {len(optim_cpu_tensors)} tensors"

        # Extract shared storages and metadata from model tensors
        model_storages = {}
        model_metadata = {}
        for key, tensor in zip(model_tensor_keys, model_cpu_tensors):
            # Get underlying storage (must be shared memory!)
            model_storages[key] = tensor.untyped_storage()
            model_metadata[key] = {
                'dtype': tensor.dtype,
                'shape': tuple(tensor.shape),
                'stride': tuple(tensor.stride()),
                'storage_offset': tensor.storage_offset(),
            }

        # Extract shared storages and metadata from optimizer tensors
        optim_storages = {}
        optim_metadata = {}
        for key, tensor in zip(optim_tensor_keys, optim_cpu_tensors):
            optim_storages[key] = tensor.untyped_storage()
            optim_metadata[key] = {
                'dtype': tensor.dtype,
                'shape': tuple(tensor.shape),
                'stride': tuple(tensor.stride()),
                'storage_offset': tensor.storage_offset(),
            }

        # Send to subprocess via pipe
        # Shared memory references are preserved (no copy)
        self.parent_pipe.send(('REGISTER', {
            'state_id': state_id,
            'state_type': in_mem_state_type,
            'model_keys': model_tensor_keys,
            'model_storages': model_storages,      # Shared memory refs
            'model_metadata': model_metadata,
            'optim_keys': optim_tensor_keys,
            'optim_storages': optim_storages,      # Shared memory refs
            'optim_metadata': optim_metadata,
        }))

        # Wait for acknowledgment (RPC-like)
        response = self.parent_pipe.recv()
        assert response == ('REGISTER_DONE', state_id), f"Expected REGISTER_DONE, got {response}"

    def snapshot_cpu_metadata(
        self,
        state_id: int,
        in_mem_state_type: InMemStateType,
        cpu_metadata: Any,
    ):
        self.parent_pipe.send(('SNAPSHOT_METADATA', {
            'state_id': state_id,
            'state_type': in_mem_state_type,
            'cpu_metadata': cpu_metadata,  # This gets pickled and copied
        }))

        response = self.parent_pipe.recv()
        assert response == ('SNAPSHOT_METADATA_DONE', state_id)

    def commit(self, state_id: int, snapshot_step: int):
        self.parent_pipe.send(('COMMIT', {
            'state_id': state_id,
            'snapshot_step': snapshot_step
        }))
        response = self.parent_pipe.recv()
        assert response == ('COMMIT', state_id)

    def close(self):
        if not self.closed:
            self.closed = True
            self.parent_pipe.send(('CLOSE', {}))
            response = self.parent_pipe.recv()
            assert response == 'CLOSE'

    @staticmethod
    def _signal_handler(signum, frame):
        SnapshotContainer._dump_states()

    @staticmethod
    def _subprocess_main(
        local_checkpoint_path: str,
        remote_checkpoint_path: str,
        log_file_path: str,
        rank: int,
        pipe):
        """Subprocess entry point with command handling and orphan detection"""
        # Redirect stdout/stderr to a temporary file for debugging
        global _process_states

        _process_states = {
            "in_mem_states": {},
            "rank": rank
        }
        _process_states["local_checkpoint_path"] = local_checkpoint_path
        _process_states["remote_checkpoint_path"] = remote_checkpoint_path
        log_file = open(log_file_path, 'w')
        sys.stdout = log_file
        sys.stderr = log_file

        # Ignore signals (we handle graceful shutdown via WorkerController)
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        # Register with WorkerController for graceful shutdown coordination
        if _LETO_AVAILABLE:
            register_service_process(_process_states["rank"])
        else:
            print("leto package not available, skipping WorkerController registration", flush=True)

        parent_pid = os.getppid()
        pipe.send("INIT_COMPLETE")

        # Main loop
        while True:
            # Check for commands from parent (with timeout)
            if pipe.poll(timeout=1.0):
                try:
                    msg = pipe.recv()

                    if not isinstance(msg, tuple) or len(msg) != 2:
                        continue

                    cmd, data = msg

                    if cmd == 'REGISTER':
                        # Create InMemStateView with shared memory references
                        view = InMemStateView(
                            state_id=data['state_id'],
                            state_type=data['state_type'],
                            model_keys=data['model_keys'],
                            model_storages=data['model_storages'],
                            model_metadata=data['model_metadata'],
                            optim_keys=data['optim_keys'],
                            optim_storages=data['optim_storages'],
                            optim_metadata=data['optim_metadata'],
                        )
                        _process_states["in_mem_states"][(data['state_id'], data['state_type'])] = view
                        pipe.send(('REGISTER_DONE', data['state_id']))
                    elif cmd == 'SNAPSHOT_METADATA':
                        state_id = data['state_id']
                        state_type = data['state_type']
                        _process_states["in_mem_states"][(state_id, state_type)].cpu_metadata = data['cpu_metadata']
                        pipe.send(('SNAPSHOT_METADATA_DONE', state_id))
                    elif cmd == 'COMMIT':
                        state_id = data['state_id']
                        snapshot_step = data['snapshot_step']
                        _process_states['commited_state_id'] = state_id
                        _process_states['snapshot_step'] = snapshot_step
                        assert ((state_id, InMemStateType.LOCAL) in _process_states["in_mem_states"])
                        assert ((state_id, InMemStateType.REMOTE) in _process_states["in_mem_states"])
                        pipe.send(('COMMIT', state_id))
                    elif cmd == 'CLOSE':
                        pipe.send('CLOSE')
                        print("Container process got CLOSE message", flush=True)
                        break
                except EOFError:
                    # Pipe closed
                    print("Pipe is unexpectedly closed.", flush=True)
                    SnapshotContainer._dump_states()
                    break
                except Exception as e:
                    # Ignore errors and continue
                    print(f"Exception: {e}", flush=True)


    @staticmethod
    def _dump_states():
        global _process_states
        if 'commited_state_id' not in _process_states:
            print("Nothing to dump.", flush=True)
            return
        snapshot_step = _process_states['snapshot_step']
        print(f"Start dumping step={snapshot_step}", flush=True)
        persist_start_time = time.time()

        commited_state_id = _process_states['commited_state_id']
        local_state_view = _process_states["in_mem_states"][(commited_state_id, InMemStateType.LOCAL)]
        remote_state_view = _process_states["in_mem_states"][(commited_state_id, InMemStateType.REMOTE)]
        del _process_states["in_mem_states"]
        gc.collect()
        local_state_view.dump(_process_states["local_checkpoint_path"])
        remote_state_view.dump(_process_states["remote_checkpoint_path"])

        # Report persisting duration to leto
        if _LETO_AVAILABLE:
            persist_duration = time.time() - persist_start_time
            report_duration(DURATION_CHECKPOINT_PERSISTING, persist_duration, snapshot_step)
            print(f"Persist took {persist_duration:.2f} seconds.", flush=True)

        print(f"End dumping step={snapshot_step}", flush=True)
