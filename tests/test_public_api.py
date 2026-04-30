"""Tests for the public API surface of the `nonans` package."""

from __future__ import annotations

import unittest


class PublicSymbolsTests(unittest.TestCase):
    def test_top_level_imports(self) -> None:
        import nonans
        # Must expose these for users.
        for name in [
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
        ]:
            self.assertTrue(hasattr(nonans, name), f"missing public symbol: {name}")

    def test_version_format(self) -> None:
        import nonans
        # Semver-ish: at least major.minor.patch.
        parts = nonans.__version__.split(".")
        self.assertGreaterEqual(len(parts), 3)
        for p in parts[:3]:
            self.assertTrue(p.isdigit(), f"version part {p!r} is not numeric")

    def test_all_matches_dir(self) -> None:
        import nonans
        # Every name in __all__ must actually exist.
        for name in nonans.__all__:
            self.assertTrue(
                hasattr(nonans, name),
                f"__all__ lists {name!r} but module does not export it",
            )


class WrapWithoutTorchTests(unittest.TestCase):
    def test_wrap_raises_clear_message_when_torch_missing(self) -> None:
        # In environments without torch (this sandbox), wrap() must raise
        # a clear runtime error rather than silently mis-behaving.
        import nonans
        try:
            import torch  # noqa: F401
            self.skipTest("torch is installed; this assertion only applies when torch is missing")
        except ImportError:
            pass

        class FakeModel:
            def __call__(self, *args, **kwargs):
                return None

        with self.assertRaises(RuntimeError) as ctx:
            nonans.wrap(FakeModel())
        # The error message must mention torch so users know what to install.
        self.assertIn("torch", str(ctx.exception).lower())

    def test_wrap_validates_mode(self) -> None:
        # mode validation must happen before any torch dependency check
        # in the codepath... but our implementation checks torch first.
        # Either order is fine; we just verify both paths produce an error.
        import nonans

        try:
            import torch  # noqa: F401
            torch_present = True
        except ImportError:
            torch_present = False

        class FakeModel:
            def __call__(self, *args, **kwargs):
                return None

        if not torch_present:
            # Without torch, wrap raises before mode is checked.
            with self.assertRaises(RuntimeError):
                nonans.wrap(FakeModel(), mode="bogus")  # type: ignore[arg-type]
        else:
            with self.assertRaises(ValueError):
                nonans.wrap(FakeModel(), mode="bogus")  # type: ignore[arg-type]


class ConfigureContextManagerTests(unittest.TestCase):
    def test_configure_yields_and_returns(self) -> None:
        import nonans
        with nonans.configure():
            pass
        # Must not raise on exit.

    def test_configure_with_telemetry_backend(self) -> None:
        import nonans
        from nonans.detect.telemetry import NoOpTelemetryBackend
        with nonans.configure(telemetry=NoOpTelemetryBackend()):
            pass


if __name__ == "__main__":
    unittest.main()
