"""Tests for the hooks module that don't require torch."""

from __future__ import annotations

import unittest

from nonans.detect import hooks
from nonans.detect.events import (
    EventKind,
    EventOrigin,
    EventSeverity,
    SingularityEvent,
    TensorFingerprint,
)


class HandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        # Reset handler state before each test. The handlers list is
        # module-global so tests must clean up after themselves.
        hooks._handlers.clear()

    def tearDown(self) -> None:
        hooks._handlers.clear()

    def test_add_handler_receives_emitted_events(self) -> None:
        received: list[SingularityEvent] = []
        hooks.add_handler(received.append)

        evt = SingularityEvent(
            kind=EventKind.DIV_BY_ZERO,
            severity=EventSeverity.CRITICAL,
            origin=EventOrigin.FORWARD,
        )
        hooks._emit(evt)

        self.assertEqual(len(received), 1)
        self.assertIs(received[0], evt)

    def test_remove_handler_stops_delivery(self) -> None:
        received: list[SingularityEvent] = []
        hooks.add_handler(received.append)
        hooks.remove_handler(received.append)

        hooks._emit(SingularityEvent(kind=EventKind.DIV_BY_ZERO))
        self.assertEqual(received, [])

    def test_remove_nonexistent_handler_is_safe(self) -> None:
        # Removing a handler that was never added must not raise.
        def never_registered(_e: SingularityEvent) -> None:
            pass
        hooks.remove_handler(never_registered)  # must not raise

    def test_handler_exception_does_not_break_emit(self) -> None:
        received: list[SingularityEvent] = []

        def broken(_e: SingularityEvent) -> None:
            raise RuntimeError("intentional")

        hooks.add_handler(broken)
        hooks.add_handler(received.append)

        # The broken handler should warn but not prevent other handlers.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hooks._emit(SingularityEvent(kind=EventKind.DIV_BY_ZERO))

        self.assertEqual(len(received), 1)


class LayerScopeTests(unittest.TestCase):
    def test_scope_sets_and_unsets_layer_name(self) -> None:
        self.assertIsNone(hooks._current_layer_name)
        with hooks.layer_scope("encoder.layer.3"):
            self.assertEqual(hooks._current_layer_name, "encoder.layer.3")
        self.assertIsNone(hooks._current_layer_name)

    def test_nested_scopes_restore_correctly(self) -> None:
        self.assertIsNone(hooks._current_layer_name)
        with hooks.layer_scope("outer"):
            self.assertEqual(hooks._current_layer_name, "outer")
            with hooks.layer_scope("inner"):
                self.assertEqual(hooks._current_layer_name, "inner")
            self.assertEqual(hooks._current_layer_name, "outer")
        self.assertIsNone(hooks._current_layer_name)

    def test_scope_restores_on_exception(self) -> None:
        try:
            with hooks.layer_scope("danger"):
                raise ValueError("boom")
        except ValueError:
            pass
        self.assertIsNone(hooks._current_layer_name)


class StepCounterTests(unittest.TestCase):
    def test_increment_returns_and_advances(self) -> None:
        starting = hooks._step_counter
        result = hooks.increment_step()
        self.assertEqual(result, starting + 1)
        self.assertEqual(hooks._step_counter, starting + 1)


class ClassifierTests(unittest.TestCase):
    """The conservative open classifier must produce sensible output."""

    def _fp(self, *, nan: int = 0, inf: int = 0, finite_frac: float = 1.0) -> TensorFingerprint:
        return TensorFingerprint(
            shape=(8,), dtype="float32", device="cuda:0",
            finite_fraction=finite_frac, abs_max_finite=1.0,
            nan_count=nan, inf_count=inf,
        )

    def test_softmax_nan_classifies_as_softmax_overflow(self) -> None:
        kind = hooks._classify(self._fp(nan=2), "aten::softmax")
        self.assertEqual(kind, EventKind.SOFTMAX_OVERFLOW)

    def test_div_nan_classifies_as_div_by_zero(self) -> None:
        kind = hooks._classify(self._fp(nan=1), "aten::div")
        self.assertEqual(kind, EventKind.DIV_BY_ZERO)

    def test_log_nan_classifies_as_log_of_nonpositive(self) -> None:
        kind = hooks._classify(self._fp(nan=1), "aten::log")
        self.assertEqual(kind, EventKind.LOG_OF_NONPOSITIVE)

    def test_sqrt_nan_classifies_as_sqrt_of_negative(self) -> None:
        kind = hooks._classify(self._fp(nan=1), "aten::sqrt")
        self.assertEqual(kind, EventKind.SQRT_OF_NEGATIVE)

    def test_inf_in_grad_classifies_as_grad_overflow(self) -> None:
        kind = hooks._classify(self._fp(inf=4), "aten::backward")
        self.assertEqual(kind, EventKind.GRAD_OVERFLOW)

    def test_inf_in_forward_classifies_as_activation_inf(self) -> None:
        kind = hooks._classify(self._fp(inf=4), "aten::matmul")
        self.assertEqual(kind, EventKind.ACTIVATION_INF)

    def test_nan_unknown_op_returns_unknown(self) -> None:
        kind = hooks._classify(self._fp(nan=1), "aten::custom_op")
        self.assertEqual(kind, EventKind.UNKNOWN)

    def test_no_anomaly_returns_unknown(self) -> None:
        kind = hooks._classify(self._fp(), "aten::softmax")
        self.assertEqual(kind, EventKind.UNKNOWN)


class ActivationTests(unittest.TestCase):
    def test_register_unregister(self) -> None:
        # Registration is a no-op without torch in the sandbox; the test
        # just verifies we can flip the state flag.
        try:
            hooks.register_kernel_hooks()
            self.assertTrue(hooks.is_active())
        except RuntimeError:
            # Torch missing — that's fine, register raises explicitly.
            self.skipTest("torch not available in this environment")
        finally:
            hooks.unregister_kernel_hooks()
            self.assertFalse(hooks.is_active())


if __name__ == "__main__":
    unittest.main()
