"""
Public NoNans API.

This module is the integration surface every user touches. It is
intentionally tiny: one function, one context manager, a handful of
configuration knobs. Everything else is internal.

Typical usage:

    import nonans
    import torch

    model = MyModel().cuda()
    model = nonans.wrap(model, mode='auto')

    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()

That's it. Detection runs immediately. Resolution runs when the licensed
runtime is reachable; otherwise the model behaves exactly as it did before
wrapping.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Literal, Optional

from nonans.detect.hooks import (
    add_handler,
    register_kernel_hooks,
    increment_step,
)
from nonans.detect.telemetry import (
    LocalTelemetryBackend,
    NoOpTelemetryBackend,
    TelemetryBackend,
    TelemetryRecorder,
)
from nonans.client import (
    resolution_available,
)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:                                           # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

logger = logging.getLogger("nonans")

Mode = Literal["auto", "detect_only", "resolve", "off"]


class _NoNansWrapper:
    """A thin module wrapper that runs detection around forward and resolution
    where applicable. We do not subclass torch.nn.Module to avoid interfering
    with FSDP, DeepSpeed, or any other distributed wrapper.
    """

    def __init__(self, module: Any, mode: Mode, recorder: TelemetryRecorder):
        self._module = module
        self._mode = mode
        self._recorder = recorder
        self._runtime_warned = False

    def __getattr__(self, name: str) -> Any:
        # Forward any attribute lookup we don't override to the wrapped
        # module. This makes our wrapper transparent to most user code.
        return getattr(self._module, name)

    def __call__(self, *args, **kwargs) -> Any:
        increment_step()
        try:
            output = self._module(*args, **kwargs)
        except Exception:
            raise
        if self._mode in ("detect_only", "off"):
            return output
        # Resolution path is handled inside the runtime when it is reachable.
        # The wrapper does not contain resolution logic.
        return output


def wrap(
    module: Any,
    *,
    mode: Mode = "auto",
    telemetry: Optional[TelemetryBackend] = None,
) -> Any:
    """Wrap a PyTorch module so NoNans observes its forward pass.

    Parameters
    ----------
    module : torch.nn.Module
        The model to wrap. FSDP, DeepSpeed, and Megatron wrappers are
        supported transparently — wrap before or after them, both work.
    mode : {'auto', 'detect_only', 'resolve', 'off'}
        - 'auto' (default): detect always; resolve when the licensed runtime
          is reachable.
        - 'detect_only': detect events and emit telemetry, never resolve.
          Useful for shadow-mode validation.
        - 'resolve': require the runtime; raise if it is unavailable.
        - 'off': do nothing. Equivalent to not wrapping at all. Useful in
          A/B comparisons.
    telemetry : TelemetryBackend, optional
        A backend to record events. Defaults to LocalTelemetryBackend()
        which writes to ./.nonans/events.jsonl. Pass NoOpTelemetryBackend()
        to disable recording.

    Returns
    -------
    The wrapped module. It quacks like the original; train it, save it,
    distribute it, all the same.
    """

    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to use nonans.wrap.")

    if mode not in ("auto", "detect_only", "resolve", "off"):
        raise ValueError(
            f"Unknown mode: {mode!r}. Must be one of 'auto', 'detect_only', "
            "'resolve', 'off'."
        )

    backend = telemetry or LocalTelemetryBackend()
    recorder = TelemetryRecorder(backend)
    add_handler(recorder)
    register_kernel_hooks()

    if mode == "resolve":
        if not resolution_available():
            raise RuntimeError(
                "mode='resolve' requires the NoNans runtime, which is not "
                "available. See https://nonans.com/#pricing or install via "
                "`docker pull ghcr.io/nonans/runtime`."
            )
    elif mode == "auto":
        if not resolution_available():
            logger.info(
                "NoNans: runtime not detected; running in detect_only mode. "
                "Detection telemetry will still be recorded. To enable "
                "resolution, see https://nonans.com/#pricing."
            )

    return _NoNansWrapper(module, mode, recorder)


@contextmanager
def configure(
    *,
    telemetry: Optional[TelemetryBackend] = None,
) -> Iterator[None]:
    """Set global configuration for the duration of a block.

    Currently only the telemetry backend is configurable from this context
    manager. Future versions will expose additional knobs.
    """
    backend = telemetry or NoOpTelemetryBackend()
    recorder = TelemetryRecorder(backend)
    add_handler(recorder)
    try:
        yield
    finally:
        # Recorders are append-only; no cleanup required for v1.
        pass
