"""
NoNans — the numerical continuity layer for GPU computing.

When a CUDA kernel produces a numerical singularity, NoNans detects it at
the kernel boundary, resolves it inside our framework, and returns a
finite, optimizer-coherent tensor to the GPU. Training continues at the
next step. No rollback. No checkpoint reload. No code change.

Public API:

    nonans.wrap(model, mode='auto')
    nonans.configure(telemetry=...)

The detection layer is open-source and ships in this package. The
resolution mechanism is patent-pending and ships only inside the licensed
runtime binary (`ghcr.io/nonans/runtime`).

For the benchmark, see https://nonans.com/benchmark.html.
For the technical replication kit, mail infra@nonans.com.
"""

from nonans.wrap import wrap, configure
from nonans.client import (
    resolution_available,
    ResolutionUnavailable,
    TrialExpired,
)
from nonans.detect import (
    SingularityEvent,
    EventKind,
    EventSeverity,
    EventOrigin,
    LocalTelemetryBackend,
    NoOpTelemetryBackend,
    TelemetryRecorder,
)

__version__ = "1.0.4"

__all__ = [
    "wrap",
    "configure",
    "resolution_available",
    "ResolutionUnavailable",
    "TrialExpired",
    "SingularityEvent",
    "EventKind",
    "EventSeverity",
    "EventOrigin",
    "LocalTelemetryBackend",
    "NoOpTelemetryBackend",
    "TelemetryRecorder",
    "__version__",
]
