"""Gemini's view of the shared snapshot container.

The kill-survivable container process itself lives in
torchtitan.components.snapshot.container; this module keeps gemini's exact
historical surface on top of it:

  - GeminiDumpPolicy: the double-buffered LOCAL/REMOTE pairing rules and the
    on-disk dump format (rank_{r}_v{v}_{local,remote}.pt +
    rank_{r}_metadata.json) exactly as SnapshotExecutor.load expects them.
  - SnapshotContainer: the pre-refactor API (state_id + InMemStateType
    keying, commit(state_id, snapshot_step)) used by InMemState and
    SnapshotExecutor, mapped onto the generic container.
"""

import json
import os

import torch

from torchtitan.components.snapshot.container import (
    DumpPolicy,
    SnapshotContainer as _GenericSnapshotContainer,
)
from torchtitan.components.gemini.utils import InMemStateType


class GeminiDumpPolicy(DumpPolicy):
    """Runs inside the container subprocess (picklable: plain str/int state).

    Ledger: {version(int): snapshot_step(int)}. States: keyed
    (version, InMemStateType) — a committed version must have both its LOCAL
    and REMOTE pools registered.
    """

    def __init__(self, checkpoint_dir: str, rank: int):
        self.checkpoint_dir = checkpoint_dir
        self.rank = rank

    def on_commit(self, states, ledger, commit_key):
        assert (commit_key, InMemStateType.LOCAL) in states
        assert (commit_key, InMemStateType.REMOTE) in states

    def _dump_view(self, view, path):
        # Exact pre-refactor InMemStateView.dump format: a single zero-copy
        # uint8 view of the pool + layout metadata, consumed by
        # InMemState.load_state_dict via _reconstruct_tensors_from_pool.
        state = {
            "_model_tensor_keys": view.layout["model_keys"],
            "_optim_tensor_keys": view.layout["optim_keys"],
            "_cpu_metadata": view.cpu_metadata,
            "_pool_bytes": view.pool_bytes(),
            "_model_metadata": view.layout["model_metadata"],
            "_optim_metadata": view.layout["optim_metadata"],
        }
        torch.save(state, path)

    def dump(self, states, ledger):
        import logging
        logger = logging.getLogger(__name__)

        metadata_path = os.path.join(
            self.checkpoint_dir, f"rank_{self.rank}_metadata.json")

        # Withdraw any previous claim BEFORE touching payload files: if this
        # dump is killed mid-way, stale metadata must not point at
        # partially-rewritten pools (a torn checkpoint that recovery would
        # trust). Losing the claim only degrades this rank to "no
        # checkpoint" — its pair still covers the step.
        try:
            os.unlink(metadata_path)
        except FileNotFoundError:
            pass

        for version, step in ledger.items():
            local_view = states[(version, InMemStateType.LOCAL)]
            remote_view = states[(version, InMemStateType.REMOTE)]
            local_path = os.path.join(
                self.checkpoint_dir, f"rank_{self.rank}_v{version}_local.pt")
            remote_path = os.path.join(
                self.checkpoint_dir, f"rank_{self.rank}_v{version}_remote.pt")
            self._dump_view(local_view, local_path)
            self._dump_view(remote_view, remote_path)
            logger.info(f"Dumped version {version} (step {step})")

        # Write metadata file (index-aligned: list[version] = step) LAST and
        # atomically: the claim appears only once every payload byte above is
        # durable, and never as a truncated json.
        metadata = {"version_steps": [ledger.get(0), ledger.get(1)]}
        tmp_path = metadata_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(metadata, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, metadata_path)
        logger.info(f"Wrote metadata: {metadata}")


class SnapshotContainer:
    """Pre-refactor gemini container API over the generic container."""

    def __init__(self, checkpoint_dir: str, log_dir: str, rank: int):
        self._container = _GenericSnapshotContainer(
            GeminiDumpPolicy(checkpoint_dir, rank), log_dir, rank)

    def register(
        self,
        state_id: int,
        in_mem_state_type: InMemStateType,
        model_tensor_keys,
        model_tensor_metadata,
        optim_tensor_keys,
        optim_tensor_metadata,
        pool_share_info: tuple,
    ):
        self._container.register(
            state_key=(state_id, in_mem_state_type),
            layout={
                "model_keys": model_tensor_keys,
                "model_metadata": model_tensor_metadata,
                "optim_keys": optim_tensor_keys,
                "optim_metadata": optim_tensor_metadata,
            },
            pool_share_info=pool_share_info,
        )

    def snapshot_cpu_metadata(
        self,
        state_id: int,
        in_mem_state_type: InMemStateType,
        cpu_metadata,
    ):
        self._container.snapshot_cpu_metadata(
            state_key=(state_id, in_mem_state_type),
            cpu_metadata=cpu_metadata,
        )

    def invalidate(self, state_id: int):
        """Mark a version as invalid (about to be overwritten)."""
        self._container.invalidate(state_id)

    def commit(self, state_id: int, snapshot_step: int):
        self._container.commit(state_id, snapshot_step)

    def close(self):
        self._container.close()
