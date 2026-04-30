"""Tests for the runtime client and transport discovery."""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import unittest

from nonans import client
from nonans.client import (
    DEFAULT_RUNTIME_SOCKET,
    DEFAULT_RUNTIME_URL,
    ResolutionUnavailable,
    _encode_tensor_handle,
    _HttpsTransport,
    _InProcessTransport,
    _RuntimeHandle,
    _UnixSocketTransport,
    reset_for_tests,
)
from nonans.detect.events import (
    EventKind,
    EventOrigin,
    EventSeverity,
    SingularityEvent,
)

# Unix-domain sockets are not available on Windows in the same way they are
# on POSIX systems. The product code degrades gracefully (the discovery
# cascade falls through to HTTPS), but tests that explicitly bind a Unix
# socket cannot run there. Skip those tests on Windows.
# Windows lacks AF_UNIX on most Python builds. Even on builds where the
# constant exists, binding an AF_UNIX socket can still fail. We test by
# attempting a bind to a temporary path and rolling back: if that succeeds,
# Unix sockets work; if it fails for any reason, we skip the integration
# tests that depend on them. This is more robust than checking sys.platform.
def _unix_sockets_available() -> bool:
    if not hasattr(socket, "AF_UNIX"):
        return False
    if sys.platform.startswith(("win", "cygwin")):
        return False
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as _tmp:
            test_path = os.path.join(_tmp, "_probe.sock")
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.bind(test_path)
                s.close()
                return True
            except (OSError, AttributeError):
                s.close()
                return False
    except Exception:
        return False


UNIX_SOCKETS_AVAILABLE = _unix_sockets_available()


def _evt() -> SingularityEvent:
    return SingularityEvent(
        kind=EventKind.DIV_BY_ZERO,
        severity=EventSeverity.CRITICAL,
        origin=EventOrigin.BACKWARD,
        step=42,
    )


class DefaultsTests(unittest.TestCase):
    def test_default_runtime_url_points_to_nonans(self) -> None:
        self.assertEqual(DEFAULT_RUNTIME_URL, "https://runtime.nonans.com")

    def test_default_socket_in_tmp(self) -> None:
        self.assertEqual(DEFAULT_RUNTIME_SOCKET, "/tmp/nonans-runtime.sock")


class TensorEncodingTests(unittest.TestCase):
    def test_none(self) -> None:
        self.assertEqual(_encode_tensor_handle(None), {"kind": "none"})

    def test_dict(self) -> None:
        out = _encode_tensor_handle({"foo": "bar"})
        self.assertEqual(out["kind"], "dict")
        self.assertEqual(out["foo"], "bar")

    def test_torch_like_object(self) -> None:
        class FakeTensor:
            shape = (32, 4096)
            dtype = "torch.bfloat16"
            device = "cuda:0"

        out = _encode_tensor_handle(FakeTensor())
        self.assertEqual(out["kind"], "torch")
        self.assertEqual(out["shape"], [32, 4096])
        self.assertEqual(out["dtype"], "bfloat16")
        self.assertEqual(out["device"], "cuda:0")

    def test_opaque(self) -> None:
        out = _encode_tensor_handle(object())
        self.assertEqual(out["kind"], "opaque")
        self.assertIn("repr", out)


class InProcessTransportTests(unittest.TestCase):
    def test_open_returns_false_when_extension_missing(self) -> None:
        # The nonans._runtime extension is proprietary and not installed
        # in this environment. open() must return False, not raise.
        t = _InProcessTransport()
        self.assertFalse(t.open())
        self.assertFalse(t.is_open())

    def test_dispatch_without_open_raises(self) -> None:
        t = _InProcessTransport()
        with self.assertRaises(ResolutionUnavailable):
            t.dispatch(b"{}")


class UnixSocketTransportTests(unittest.TestCase):
    def test_open_returns_false_when_socket_missing(self) -> None:
        t = _UnixSocketTransport("/tmp/this-path-should-not-exist-nonans-test")
        self.assertFalse(t.open())

    @unittest.skipUnless(
        UNIX_SOCKETS_AVAILABLE,
        "Unix sockets not available (Windows or restricted runtime)",
    )
    def test_open_succeeds_against_real_socket_and_dispatches(self) -> None:
        """Spin up a tiny echo server on a Unix socket and confirm dispatch
        round-trips a request. This is the integration test for the framing."""

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            sock_path = os.path.join(tmp, "rt.sock")
            server_done = threading.Event()
            received: list[bytes] = []

            def server() -> None:
                srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                srv.bind(sock_path)
                srv.listen(1)
                conn, _addr = srv.accept()
                # Read length-prefixed request.
                header = conn.recv(4)
                length = int.from_bytes(header, "big")
                body = b""
                while len(body) < length:
                    chunk = conn.recv(length - len(body))
                    if not chunk:
                        break
                    body += chunk
                received.append(body)
                # Reply with a length-prefixed response.
                response = json.dumps({"status": "ok", "tensor": {"resolved": True}}).encode()
                conn.sendall(len(response).to_bytes(4, "big") + response)
                conn.close()
                srv.close()
                server_done.set()

            t = threading.Thread(target=server, daemon=True)
            t.start()
            # Wait for the server to bind.
            for _ in range(50):
                if os.path.exists(sock_path):
                    break
                time.sleep(0.01)

            transport = _UnixSocketTransport(sock_path)
            self.assertTrue(transport.open())
            response = transport.dispatch(b'{"hello":"world"}')
            transport.close()

            self.assertTrue(server_done.wait(timeout=5))
            self.assertEqual(received, [b'{"hello":"world"}'])
            parsed = json.loads(response)
            self.assertEqual(parsed["status"], "ok")
            self.assertTrue(parsed["tensor"]["resolved"])


class HttpsTransportTests(unittest.TestCase):
    def test_open_validates_url_scheme(self) -> None:
        good = _HttpsTransport("https://runtime.nonans.com")
        self.assertTrue(good.open())
        self.assertTrue(good.is_open())

        bad = _HttpsTransport("not-a-url")
        self.assertFalse(bad.open())
        self.assertFalse(bad.is_open())

    def test_dispatch_without_open_raises(self) -> None:
        t = _HttpsTransport("https://runtime.nonans.com")
        # Don't call open(); is_open is False, dispatch should raise.
        with self.assertRaises(ResolutionUnavailable):
            t.dispatch(b'{"event":{}}')

    def test_token_environment_picked_up(self) -> None:
        os.environ["NONANS_TOKEN"] = "fake-trial-token-12345"
        try:
            t = _HttpsTransport("https://runtime.nonans.com")
            self.assertEqual(t._token, "fake-trial-token-12345")
        finally:
            del os.environ["NONANS_TOKEN"]


class DiscoveryPriorityTests(unittest.TestCase):
    """Verify the discovery cascade picks the right transport."""

    def setUp(self) -> None:
        # Save and clear any environment knobs that affect discovery.
        self._saved_env = {
            k: os.environ.pop(k, None)
            for k in (
                "NONANS_OFFLINE",
                "NONANS_RUNTIME_SOCKET",
                "NONANS_RUNTIME_URL",
            )
        }
        reset_for_tests()

    def tearDown(self) -> None:
        for k, v in self._saved_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        reset_for_tests()

    def test_default_picks_https_when_no_in_process_or_socket(self) -> None:
        h = _RuntimeHandle()
        h.connect()
        self.assertEqual(h.transport_name, "https")

    def test_offline_env_disables_https(self) -> None:
        os.environ["NONANS_OFFLINE"] = "1"
        h = _RuntimeHandle()
        h.connect()
        self.assertFalse(h.is_connected)
        self.assertEqual(h.transport_name, "none")

    @unittest.skipUnless(
        UNIX_SOCKETS_AVAILABLE,
        "Unix sockets not available (Windows or restricted runtime)",
    )
    def test_socket_takes_priority_over_https(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            sock_path = os.path.join(tmp, "rt.sock")

            # Spin up an idle Unix socket server.
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(sock_path)
            srv.listen(1)
            try:
                os.environ["NONANS_RUNTIME_SOCKET"] = sock_path
                h = _RuntimeHandle()
                h.connect()
                self.assertEqual(h.transport_name, "unix-socket")
                h.force_disconnect()
            finally:
                srv.close()


class HandleResolveTests(unittest.TestCase):
    def test_resolve_when_disconnected_raises(self) -> None:
        h = _RuntimeHandle()
        # Don't connect; immediately ask for resolution.
        with self.assertRaises(ResolutionUnavailable):
            h.resolve(_evt(), None)


class PublicAPITests(unittest.TestCase):
    def test_resolution_available_does_not_raise(self) -> None:
        # Public function that must always return a bool, never raise.
        result = client.resolution_available()
        self.assertIsInstance(result, bool)

    def test_transport_name_returns_string(self) -> None:
        result = client.transport_name()
        self.assertIsInstance(result, str)
        self.assertIn(result, {"none", "in-process", "unix-socket", "https"})


if __name__ == "__main__":
    unittest.main()
