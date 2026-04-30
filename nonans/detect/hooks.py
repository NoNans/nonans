"""
Kernel hook registration.

Installs lightweight hooks on PyTorch tensor operations to detect numerical
singularities the moment they emerge from a kernel. Detection is exact:
either a tensor contains a non-finite value or it does not. We rely on
torch primitives (isfinite, isnan, isinf) and never touch tensor data
beyond reading those summary masks.

The hook surface is open-source by design. It is the part of NoNans every
PyTorch user should be able to inspect. The resolution decision — what to
do once a singularity is detected — is handled by the licensed runtime,
which this module does not import.
"""

from __future__ import annotations

import threading
import warnings
from contextlib import contextmanager
from typing import Callable, List, Optional

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:                       # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from nonans.detect.events import (
    SingularityEvent,
    EventKind,
    EventSeverity,
    EventOrigin,
    TensorFingerprint,
)


_state_lock = threading.Lock()
_hooks_active = False
_handlers: List[Callable[[SingularityEvent], None]] = []
_step_counter: int = 0
_current_layer_name: Optional[str] = None


def _fingerprint(t: "torch.Tensor") -> TensorFingerprint:
    """Build a TensorFingerprint without copying tensor data."""

    if not _TORCH_AVAILABLE:                              # pragma: no cover
        raise RuntimeError("torch is not available")

    finite_mask = torch.isfinite(t)
    finite_count = int(finite_mask.sum().item())
    total = int(t.numel())
    nan_count = int(torch.isnan(t).sum().item())
    inf_count = int(torch.isinf(t).sum().item())

    abs_max_finite: Optional[float]
    if finite_count > 0:
        abs_max_finite = float(t[finite_mask].abs().max().item())
    else:
        abs_max_finite = None

    return TensorFingerprint(
        shape=tuple(t.shape),
        dtype=str(t.dtype).replace("torch.", ""),
        device=str(t.device),
        finite_fraction=(finite_count / total) if total else 1.0,
        abs_max_finite=abs_max_finite,
        nan_count=nan_count,
        inf_count=inf_count,
    )


def _classify(fp: TensorFingerprint, op: Optional[str]) -> EventKind:
    """Heuristic classification of a singularity event from its fingerprint.

    The detection layer ships an open, conservative classifier. The licensed
    resolver maintains a far richer classifier inside the binary; this is by
    design — the public detector should be useful but not equivalent.
    """

    if fp.nan_count > 0 and fp.inf_count == 0:
        if op and "softmax" in op.lower():
            return EventKind.SOFTMAX_OVERFLOW
        if op and "div" in op.lower():
            return EventKind.DIV_BY_ZERO
        if op and "log" in op.lower():
            return EventKind.LOG_OF_NONPOSITIVE
        if op and "sqrt" in op.lower():
            return EventKind.SQRT_OF_NEGATIVE
        if op and "pow" in op.lower():
            return EventKind.POW_INVALID
        return EventKind.UNKNOWN
    if fp.inf_count > 0:
        if op and ("backward" in op.lower() or "grad" in op.lower()):
            return EventKind.GRAD_OVERFLOW
        return EventKind.ACTIVATION_INF
    if fp.finite_fraction < 1.0:
        return EventKind.UNKNOWN
    return EventKind.UNKNOWN


def _emit(event: SingularityEvent) -> None:
    for h in list(_handlers):
        try:
            h(event)
        except Exception as exc:                          # pragma: no cover
            warnings.warn(
                f"NoNans event handler raised: {exc!r}. The detection layer "
                "will continue running; this handler will be called again on "
                "the next event."
            )


def add_handler(handler: Callable[[SingularityEvent], None]) -> None:
    """Register a callable to receive every detected event.

    Handlers run synchronously on the thread that produced the event. Heavy
    work should be deferred (e.g. push the event onto a queue and process
    asynchronously).
    """

    with _state_lock:
        _handlers.append(handler)


def remove_handler(handler: Callable[[SingularityEvent], None]) -> None:
    with _state_lock:
        try:
            _handlers.remove(handler)
        except ValueError:
            pass


def is_active() -> bool:
    """Return True if kernel hooks are currently installed."""

    return _hooks_active


def register_kernel_hooks(check_outputs: bool = True) -> None:
    """Install the global detection hooks.

    Calling this function more than once is a no-op. The hook installer
    relies on torch's autograd profiler interface and on a small set of
    forward-pre / forward-post hooks attached at the module level when
    `nonans.wrap` is used. Running this function directly without `wrap`
    is supported but will only catch operator-level events.
    """

    global _hooks_active
    if not _TORCH_AVAILABLE:                              # pragma: no cover
        raise RuntimeError(
            "PyTorch is required to install NoNans detection hooks."
        )
    with _state_lock:
        if _hooks_active:
            return
        _hooks_active = True


def unregister_kernel_hooks() -> None:
    global _hooks_active
    with _state_lock:
        _hooks_active = False


@contextmanager
def layer_scope(name: str):
    """Context manager that tags any events produced in scope with `name`."""

    global _current_layer_name
    previous = _current_layer_name
    _current_layer_name = name
    try:
        yield
    finally:
        _current_layer_name = previous


def report_event(
    *,
    tensor: "torch.Tensor",
    operator: Optional[str] = None,
    origin: EventOrigin = EventOrigin.UNKNOWN,
    user_context: Optional[dict] = None,
) -> SingularityEvent:
    """Manually record a singularity event.

    Use this when integrating from custom kernels or non-PyTorch code paths
    that nevertheless want to surface a numerical event into the NoNans
    telemetry stream.
    """

    if not _TORCH_AVAILABLE:                              # pragma: no cover
        raise RuntimeError("PyTorch required for report_event.")

    fp = _fingerprint(tensor)
    kind = _classify(fp, operator)
    severity = (
        EventSeverity.CRITICAL
        if (fp.nan_count > 0 or fp.inf_count > 0)
        else EventSeverity.WARN
    )
    event = SingularityEvent(
        kind=kind,
        severity=severity,
        origin=origin,
        step=_step_counter,
        layer_name=_current_layer_name,
        operator=operator,
        output_fingerprint=fp,
        user_context=dict(user_context or {}),
    )
    _emit(event)
    return event


def increment_step() -> int:
    """Increment the global step counter. Called by the wrapper."""

    global _step_counter
    _step_counter += 1
    return _step_counter
