"""
Telemetry recorder.

A telemetry backend is a sink for SingularityEvents. The default backend
records events to a local ring buffer and writes a JSON line per event to a
file under ./.nonans/events.jsonl. A NoOp backend is provided for users
who do not want any telemetry recorded.

Remote telemetry (anonymous aggregation across deployments) is opt-in and
lives in the licensed runtime, not in this module. The detection layer is
deliberately offline-only.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

from nonans.detect.events import SingularityEvent


class TelemetryBackend:
    """Abstract sink for SingularityEvents."""

    def record(self, event: SingularityEvent) -> None:        # pragma: no cover
        raise NotImplementedError

    def flush(self) -> None:                                  # pragma: no cover
        pass


class NoOpTelemetryBackend(TelemetryBackend):
    """Records nothing. Use when telemetry is undesirable."""

    def record(self, event: SingularityEvent) -> None:
        return


class LocalTelemetryBackend(TelemetryBackend):
    """Append-only JSON-lines file plus an in-memory ring buffer.

    Designed to be safe under concurrent recording; event IDs guarantee
    uniqueness so out-of-order writes from multiple workers are fine.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        ring_size: int = 1024,
    ):
        if path is None:
            base = Path(os.environ.get("NONANS_TELEMETRY_DIR", ".nonans"))
            base.mkdir(parents=True, exist_ok=True)
            path = str(base / "events.jsonl")
        self._path = path
        self._lock = threading.Lock()
        self._ring: Deque[SingularityEvent] = deque(maxlen=ring_size)

    def record(self, event: SingularityEvent) -> None:
        line = event.to_json() + "\n"
        with self._lock:
            self._ring.append(event)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)

    def recent(self, n: int = 32) -> List[SingularityEvent]:
        with self._lock:
            return list(self._ring)[-n:]

    def flush(self) -> None:
        # File writes are immediate; nothing to flush. Kept for API parity.
        return


class TelemetryRecorder:
    """Convenience wrapper that owns a backend and exposes a callable."""

    def __init__(self, backend: Optional[TelemetryBackend] = None):
        self.backend = backend or LocalTelemetryBackend()

    def __call__(self, event: SingularityEvent) -> None:
        self.backend.record(event)

    def recent(self, n: int = 32) -> List[SingularityEvent]:
        if isinstance(self.backend, LocalTelemetryBackend):
            return self.backend.recent(n)
        return []
