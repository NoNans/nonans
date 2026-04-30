"""
NoNans detection layer.

Open-source. MIT licensed. Surfaces numerical singularity events at the kernel
boundary and emits structured telemetry. Resolution requires the licensed
runtime binary; without it, this module reports events and exits gracefully.

The detection layer is intentionally fully open. We want every PyTorch user
who has ever lost a training run to a NaN to be able to install this, run
their workload, and see exactly what happened and where. The decision to
resolve the singularity (and continue training) is a separate, gated step.
"""

from nonans.detect.events import (
    SingularityEvent,
    EventKind,
    EventSeverity,
    EventOrigin,
)
from nonans.detect.hooks import (
    register_kernel_hooks,
    unregister_kernel_hooks,
    is_active,
)
from nonans.detect.telemetry import (
    TelemetryRecorder,
    LocalTelemetryBackend,
    NoOpTelemetryBackend,
)

__all__ = [
    "SingularityEvent",
    "EventKind",
    "EventSeverity",
    "EventOrigin",
    "register_kernel_hooks",
    "unregister_kernel_hooks",
    "is_active",
    "TelemetryRecorder",
    "LocalTelemetryBackend",
    "NoOpTelemetryBackend",
]

__version__ = "1.0.4"
