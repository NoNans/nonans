"""
Singularity event taxonomy.

A SingularityEvent is the structured record produced by the detection layer
when a kernel produces a numerical condition that would cascade into NaN
contagion. The event captures everything the resolver needs (or, if the
resolver is not present, everything a developer needs to debug the failure).

The taxonomy is deliberately stable. New event kinds are added at the end
of EventKind to preserve enum ordinality across versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple
import json
import time
import uuid


class EventKind(IntEnum):
    """The numerical condition that triggered the event.

    Ordinality is part of the public API. Do not reorder. Add new kinds at
    the end. Resolver versioning depends on this stability.
    """

    UNKNOWN = 0
    DIV_BY_ZERO = 1            # x / 0 in a kernel output
    GRAD_OVERFLOW = 2          # |grad| exceeded representable range
    GRAD_UNDERFLOW = 3         # |grad| collapsed to subnormal/zero
    SOFTMAX_OVERFLOW = 4       # softmax denominator collapse
    LOG_OF_NONPOSITIVE = 5     # log(x), x <= 0
    SQRT_OF_NEGATIVE = 6       # sqrt(x), x < 0
    POW_INVALID = 7            # 0^0, (-x)^0.5, etc.
    ATTENTION_DENOM_COLLAPSE = 8
    OPTIMIZER_BUFFER_NAN = 9   # NaN observed inside optimizer state
    ACTIVATION_INF = 10        # ±inf in forward pass activation
    LOSS_NAN = 11              # NaN in scalar loss
    USER_FLAGGED = 12          # User-supplied check raised


class EventSeverity(IntEnum):
    """How urgent the event is. Maps to handler escalation policy."""

    INFO = 0       # Detected but recoverable without intervention.
    WARN = 1       # Detected and would propagate; resolution recommended.
    CRITICAL = 2   # Detected; without resolution the run dies this step.
    FATAL = 3      # Detected; resolver could not handle it (rare).


class EventOrigin(IntEnum):
    """Which layer of the stack first observed the event."""

    UNKNOWN = 0
    FORWARD = 1
    BACKWARD = 2
    OPTIMIZER = 3
    LOSS = 4
    CUSTOM_KERNEL = 5
    INFERENCE = 6


@dataclass(frozen=True)
class TensorFingerprint:
    """A small, fixed-size summary of a tensor that doesn't leak its values."""

    shape: Tuple[int, ...]
    dtype: str
    device: str
    finite_fraction: float        # 0.0 to 1.0
    abs_max_finite: Optional[float]
    nan_count: int
    inf_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "finite_fraction": self.finite_fraction,
            "abs_max_finite": self.abs_max_finite,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
        }


@dataclass
class SingularityEvent:
    """A single observation of a numerical singularity.

    Events are immutable once constructed. The detection layer creates them;
    the resolver consumes them. Telemetry serializes them as JSON.
    """

    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)

    kind: EventKind = EventKind.UNKNOWN
    severity: EventSeverity = EventSeverity.WARN
    origin: EventOrigin = EventOrigin.UNKNOWN

    step: Optional[int] = None
    epoch: Optional[int] = None
    layer_name: Optional[str] = None
    operator: Optional[str] = None    # e.g. "aten::div", "softmax"

    input_fingerprints: Tuple[TensorFingerprint, ...] = field(default_factory=tuple)
    output_fingerprint: Optional[TensorFingerprint] = None

    optimizer_state_intact: Optional[bool] = None

    # User-supplied context. Free-form, kept short on the wire.
    user_context: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = int(self.kind)
        d["severity"] = int(self.severity)
        d["origin"] = int(self.origin)
        d["input_fingerprints"] = [
            fp.to_dict() if hasattr(fp, "to_dict") else fp
            for fp in self.input_fingerprints
        ]
        if self.output_fingerprint is not None:
            d["output_fingerprint"] = (
                self.output_fingerprint.to_dict()
                if hasattr(self.output_fingerprint, "to_dict")
                else self.output_fingerprint
            )
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def __repr__(self) -> str:
        return (
            f"SingularityEvent(kind={self.kind.name}, "
            f"severity={self.severity.name}, "
            f"origin={self.origin.name}, "
            f"step={self.step}, layer={self.layer_name!r})"
        )
