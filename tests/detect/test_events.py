"""Tests for the SingularityEvent taxonomy."""

from __future__ import annotations

import json
import unittest

from nonans.detect.events import (
    EventKind,
    EventOrigin,
    EventSeverity,
    SingularityEvent,
    TensorFingerprint,
)


class EventEnumStabilityTests(unittest.TestCase):
    """Enum ordinality is part of the public API and must not drift."""

    def test_event_kind_ordinality(self) -> None:
        # If any of these change, every cross-version protocol message
        # silently misrouts. Locked.
        self.assertEqual(EventKind.UNKNOWN, 0)
        self.assertEqual(EventKind.DIV_BY_ZERO, 1)
        self.assertEqual(EventKind.GRAD_OVERFLOW, 2)
        self.assertEqual(EventKind.GRAD_UNDERFLOW, 3)
        self.assertEqual(EventKind.SOFTMAX_OVERFLOW, 4)
        self.assertEqual(EventKind.LOG_OF_NONPOSITIVE, 5)
        self.assertEqual(EventKind.SQRT_OF_NEGATIVE, 6)
        self.assertEqual(EventKind.POW_INVALID, 7)
        self.assertEqual(EventKind.ATTENTION_DENOM_COLLAPSE, 8)
        self.assertEqual(EventKind.OPTIMIZER_BUFFER_NAN, 9)
        self.assertEqual(EventKind.ACTIVATION_INF, 10)
        self.assertEqual(EventKind.LOSS_NAN, 11)
        self.assertEqual(EventKind.USER_FLAGGED, 12)

    def test_severity_ordinality(self) -> None:
        self.assertEqual(EventSeverity.INFO, 0)
        self.assertEqual(EventSeverity.WARN, 1)
        self.assertEqual(EventSeverity.CRITICAL, 2)
        self.assertEqual(EventSeverity.FATAL, 3)

    def test_origin_ordinality(self) -> None:
        self.assertEqual(EventOrigin.UNKNOWN, 0)
        self.assertEqual(EventOrigin.FORWARD, 1)
        self.assertEqual(EventOrigin.BACKWARD, 2)
        self.assertEqual(EventOrigin.OPTIMIZER, 3)
        self.assertEqual(EventOrigin.LOSS, 4)
        self.assertEqual(EventOrigin.CUSTOM_KERNEL, 5)
        self.assertEqual(EventOrigin.INFERENCE, 6)


class TensorFingerprintTests(unittest.TestCase):
    def test_construction_and_to_dict(self) -> None:
        fp = TensorFingerprint(
            shape=(32, 4096),
            dtype="bfloat16",
            device="cuda:0",
            finite_fraction=0.998,
            abs_max_finite=12.4,
            nan_count=8,
            inf_count=0,
        )
        d = fp.to_dict()
        self.assertEqual(d["shape"], [32, 4096])
        self.assertEqual(d["dtype"], "bfloat16")
        self.assertEqual(d["nan_count"], 8)

    def test_zero_finite_tensor(self) -> None:
        fp = TensorFingerprint(
            shape=(1024,),
            dtype="float32",
            device="cuda:0",
            finite_fraction=0.0,
            abs_max_finite=None,
            nan_count=1024,
            inf_count=0,
        )
        d = fp.to_dict()
        self.assertEqual(d["finite_fraction"], 0.0)
        self.assertIsNone(d["abs_max_finite"])

    def test_immutability(self) -> None:
        fp = TensorFingerprint(
            shape=(8,), dtype="float32", device="cpu",
            finite_fraction=1.0, abs_max_finite=1.0,
            nan_count=0, inf_count=0,
        )
        with self.assertRaises((AttributeError, Exception)):
            fp.shape = (16,)  # type: ignore[misc]


class SingularityEventTests(unittest.TestCase):
    def _make(self, **overrides: object) -> SingularityEvent:
        defaults: dict = dict(
            kind=EventKind.SOFTMAX_OVERFLOW,
            severity=EventSeverity.CRITICAL,
            origin=EventOrigin.FORWARD,
            step=1832,
            layer_name="transformer.layers.7.attn",
            operator="softmax",
        )
        defaults.update(overrides)
        return SingularityEvent(**defaults)  # type: ignore[arg-type]

    def test_event_id_unique(self) -> None:
        ids = {self._make().event_id for _ in range(1000)}
        self.assertEqual(len(ids), 1000, "event_ids must collide-free across 1k events")

    def test_to_dict_preserves_int_enums(self) -> None:
        e = self._make()
        d = e.to_dict()
        self.assertEqual(d["kind"], int(EventKind.SOFTMAX_OVERFLOW))
        self.assertEqual(d["severity"], int(EventSeverity.CRITICAL))
        self.assertEqual(d["origin"], int(EventOrigin.FORWARD))

    def test_json_roundtrip(self) -> None:
        e = self._make()
        s = e.to_json()
        parsed = json.loads(s)
        self.assertEqual(parsed["kind"], int(EventKind.SOFTMAX_OVERFLOW))
        self.assertEqual(parsed["layer_name"], "transformer.layers.7.attn")
        self.assertIn("event_id", parsed)
        self.assertIn("timestamp", parsed)

    def test_json_with_fingerprint(self) -> None:
        fp = TensorFingerprint(
            shape=(32, 4096), dtype="bf16", device="cuda:0",
            finite_fraction=0.99, abs_max_finite=14.2,
            nan_count=8, inf_count=2,
        )
        e = self._make(output_fingerprint=fp, input_fingerprints=(fp,))
        parsed = json.loads(e.to_json())
        self.assertEqual(parsed["output_fingerprint"]["shape"], [32, 4096])
        self.assertEqual(parsed["output_fingerprint"]["nan_count"], 8)
        self.assertEqual(len(parsed["input_fingerprints"]), 1)
        self.assertEqual(parsed["input_fingerprints"][0]["dtype"], "bf16")

    def test_repr_useful_for_debugging(self) -> None:
        e = self._make()
        r = repr(e)
        self.assertIn("SOFTMAX_OVERFLOW", r)
        self.assertIn("CRITICAL", r)
        self.assertIn("1832", r)
        self.assertIn("transformer.layers.7.attn", r)

    def test_user_context_default_isolation(self) -> None:
        # Two events must not share a default user_context dict.
        a = self._make()
        b = self._make()
        a.user_context["debug"] = "first"
        self.assertNotIn("debug", b.user_context)


if __name__ == "__main__":
    unittest.main()
