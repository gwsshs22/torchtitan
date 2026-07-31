"""MoEvement sparse in-memory checkpointing (checkpoint.method="moevement").

Port of MoEvement's sparse checkpointing / sparse-to-dense conversion /
upstream logging onto torchtitan + leto, storing all durable state in the
shared kill-survivable snapshot layer (torchtitan.components.snapshot).
Design and porting notes: docs/moevement_port_plan.md in the leto repo.
"""

from torchtitan.components.moevement.checkpoint import (  # noqa: F401
    MoevementCheckpointManager,
)
from torchtitan.components.moevement.conversion import (  # noqa: F401
    WindowBundle,
    apply_iteration,
    find_committed_windows,
    restore_rng,
)
from torchtitan.components.moevement.operators import (  # noqa: F401
    Operator,
    OperatorKind,
    discover_operators,
)
from torchtitan.components.moevement.scheduler import (  # noqa: F401
    CheckpointSchedule,
    SchedulableOp,
    SparseCheckpointScheduler,
)
from torchtitan.components.moevement.snapshot_engine import (  # noqa: F401
    MoevementDumpPolicy,
    MoevementSnapshotEngine,
)
from torchtitan.components.moevement.upstream_logger import (  # noqa: F401
    UpstreamLogger,
    UpstreamTeePipelineStage,
    load_persisted_logs,
)
