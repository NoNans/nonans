"""Tests for telemetry recording backends."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest

from nonans.detect.events import (
    EventKind,
    EventOrigin,
    EventSeverity,
    SingularityEvent,
)
from nonans.detect.telemetry import (
    LocalTelemetryBackend,
    NoOpTelemetryBackend,
    TelemetryRecorder,
)


def _evt(step: int = 0) -> SingularityEvent:
    return SingularityEvent(
        kind=EventKind.DIV_BY_ZERO,
        severity=EventSeverity.CRITICAL,
        origin=EventOrigin.BACKWARD,
        step=step,
        layer_name=f"layer.{step}",
    )


class NoOpBackendTests(unittest.TestCase):
    def test_absorbs_events_silently(self) -> None:
        b = NoOpTelemetryBackend()
        for i in range(1000):
            b.record(_evt(i))  # must not raise

    def test_flush_is_noop(self) -> None:
        NoOpTelemetryBackend().flush()


class LocalBackendTests(unittest.TestCase):
    def test_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.jsonl")
            b = LocalTelemetryBackend(path=path)
            for i in range(5):
                b.record(_evt(i))
            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 5)
            for i, line in enumerate(lines):
                parsed = json.loads(line)
                self.assertEqual(parsed["step"], i)
                self.assertEqual(parsed["kind"], int(EventKind.DIV_BY_ZERO))

    def test_ring_buffer_recent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.jsonl")
            b = LocalTelemetryBackend(path=path, ring_size=10)
            for i in range(25):
                b.record(_evt(i))
            recent = b.recent(n=10)
            self.assertEqual(len(recent), 10)
            # The ring keeps the latest events.
            steps = [e.step for e in recent]
            self.assertEqual(steps, list(range(15, 25)))

    def test_recent_with_n_larger_than_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.jsonl")
            b = LocalTelemetryBackend(path=path, ring_size=10)
            for i in range(3):
                b.record(_evt(i))
            recent = b.recent(n=100)
            self.assertEqual(len(recent), 3)

    def test_default_path_uses_env_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["NONANS_TELEMETRY_DIR"] = tmp
            try:
                b = LocalTelemetryBackend()
                b.record(_evt(0))
                self.assertTrue(
                    os.path.exists(os.path.join(tmp, "events.jsonl"))
                )
            finally:
                del os.environ["NONANS_TELEMETRY_DIR"]

    def test_concurrent_writes_dont_corrupt(self) -> None:
        """Many threads writing simultaneously must produce well-formed JSONL."""

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.jsonl")
            b = LocalTelemetryBackend(path=path)

            def writer(start: int) -> None:
                for i in range(100):
                    b.record(_evt(start + i))

            threads = [threading.Thread(target=writer, args=(t * 1000,)) for t in range(8)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()

            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 800, "all writes must be present")
            for line in lines:
                # Every line must be a valid JSON object — no torn writes.
                parsed = json.loads(line)
                self.assertIn("event_id", parsed)


class RecorderTests(unittest.TestCase):
    def test_callable_passes_to_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.jsonl")
            b = LocalTelemetryBackend(path=path)
            r = TelemetryRecorder(b)
            r(_evt(42))
            recent = r.recent()
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0].step, 42)

    def test_recent_returns_empty_for_noop(self) -> None:
        r = TelemetryRecorder(NoOpTelemetryBackend())
        self.assertEqual(r.recent(), [])


if __name__ == "__main__":
    unittest.main()
