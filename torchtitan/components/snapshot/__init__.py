"""Shared kill-survivable snapshot storage layer.

Consumed by the gemini and moevement in-memory checkpointing methods:
  - pool_shm: tmpfs-backed, pinned, path-shared (then unlinked) memory pools
  - container: the per-rank SnapshotContainer service process + DumpPolicy
  - partner: THE shared replica-partner (buddy) selection rule — mutual,
    same-pipeline-stage, cross-host-maximizing (import it directly:
    ``from torchtitan.components.snapshot import partner``)
"""

from torchtitan.components.snapshot.container import (  # noqa: F401
    DumpPolicy,
    SnapshotContainer,
    StateView,
)
