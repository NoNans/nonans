"""
Client for the NoNans resolution runtime.

The resolution runtime is proprietary and lives at runtime.nonans.com by
default. The customer's package (this module) opens a TLS connection to
the hosted runtime, authenticates with a trial token, and submits
SingularityEvent records for resolution.

Three deployment modes are supported, in priority order:

  1. In-process extension (Enterprise on-prem only).  Direct import of
     a `nonans._runtime` extension. The fastest path; reserved for
     Enterprise customers with the on-prem runtime installed.
  2. Sidecar runtime (opt-in).  Local Unix socket at
     $NONANS_RUNTIME_SOCKET (default /tmp/nonans-runtime.sock).
     Pro and Enterprise customers running the sidecar Docker image.
  3. Hosted runtime (default).  TLS to https://runtime.nonans.com.
     Free tier and Pro tier customers use this.

This module contains no resolution logic. It is, by design, a transport
shim. The mechanism by which the runtime resolves a singularity is not
described here, in the public documentation, or in the MNDA replication
kit. It lives only on the NoNans servers (or, for Enterprise on-prem,
inside a Cython-compiled binary distributed under contract).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import ssl
import threading
import time
from typing import Any, Optional

from nonans.detect.events import SingularityEvent

logger = logging.getLogger("nonans.client")


# Public defaults. Customers can override via environment variables.
DEFAULT_RUNTIME_URL = "https://runtime.nonans.com"
DEFAULT_RUNTIME_SOCKET = "/tmp/nonans-runtime.sock"

# Connect/read timeouts in seconds. Resolution is a rare-event path, so
# these are generous; if a request takes longer than this, something is
# genuinely wrong and we want to surface it rather than block the run.
CONNECT_TIMEOUT_SECONDS = 5.0
READ_TIMEOUT_SECONDS = 30.0


class ResolutionUnavailable(RuntimeError):
    """Raised when the resolution runtime cannot be reached.

    When this occurs, the wrapper falls back to PyTorch's native behavior
    (which is, typically, to let the run die). We never silently corrupt
    training; we surface the unavailability and let the user decide.
    """


class TrialExpired(RuntimeError):
    """Raised when the runtime token has expired.

    Standard remediation: contact infra@nonans.com to issue a fresh token,
    or upgrade to Pro through the pricing page.
    """


class _Transport:
    """Abstract transport. Implementations talk to a runtime instance."""

    name: str = "abstract"

    def is_open(self) -> bool:
        raise NotImplementedError

    def dispatch(self, payload: bytes) -> bytes:
        raise NotImplementedError

    def close(self) -> None:
        return


class _InProcessTransport(_Transport):
    """Direct call into a `nonans._runtime` extension. Enterprise on-prem."""

    name = "in-process"

    def __init__(self) -> None:
        self._rt: Any = None

    def open(self) -> bool:
        try:
            import nonans._runtime as rt          # type: ignore[import-not-found]
        except ImportError:
            return False
        self._rt = rt
        return True

    def is_open(self) -> bool:
        return self._rt is not None

    def dispatch(self, payload: bytes) -> bytes:
        if self._rt is None:
            raise ResolutionUnavailable("In-process runtime not initialized.")
        return self._rt.dispatch(payload)


class _UnixSocketTransport(_Transport):
    """Local Unix socket. Sidecar Docker image deployment."""

    name = "unix-socket"

    def __init__(self, path: str):
        self._path = path
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def open(self) -> bool:
        if not os.path.exists(self._path):
            return False
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(CONNECT_TIMEOUT_SECONDS)
        try:
            s.connect(self._path)
        except OSError:
            s.close()
            return False
        s.settimeout(READ_TIMEOUT_SECONDS)
        self._sock = s
        return True

    def is_open(self) -> bool:
        return self._sock is not None

    def dispatch(self, payload: bytes) -> bytes:
        if self._sock is None:
            raise ResolutionUnavailable("Unix socket transport not connected.")
        with self._lock:
            return _send_framed(self._sock, payload)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


class _HttpsTransport(_Transport):
    """TLS to runtime.nonans.com. Default for free + Pro tiers."""

    name = "https"

    def __init__(self, url: str, token: Optional[str] = None):
        self._url = url.rstrip("/")
        self._token = token or os.environ.get("NONANS_TOKEN")
        self._verified = False

    def open(self) -> bool:
        # We do not eagerly establish the connection because urllib does
        # connection pooling for us. We only verify the URL is well-formed.
        if not self._url.startswith(("http://", "https://")):
            return False
        self._verified = True
        return True

    def is_open(self) -> bool:
        return self._verified

    def dispatch(self, payload: bytes) -> bytes:
        if not self._verified:
            raise ResolutionUnavailable("HTTPS transport not initialized.")

        # Lazy import urllib so the module loads cleanly without network access.
        import urllib.error
        import urllib.request

        endpoint = f"{self._url}/v1/resolve"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": _user_agent(),
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        req = urllib.request.Request(
            endpoint,
            data=payload,
            method="POST",
            headers=headers,
        )
        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(
                req,
                timeout=READ_TIMEOUT_SECONDS,
                context=ctx,
            ) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:                                 # pragma: no cover
                body = "<unreadable body>"
            if exc.code == 401 or "expired" in body.lower():
                raise TrialExpired(
                    "NoNans trial token expired or unauthorized. Contact "
                    "infra@nonans.com or upgrade at https://nonans.com/#pricing."
                ) from exc
            raise ResolutionUnavailable(
                f"Runtime returned HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ResolutionUnavailable(
                f"Could not reach runtime at {self._url}: {exc.reason!r}"
            ) from exc


def _user_agent() -> str:
    from nonans import __version__
    return f"nonans-client/{__version__}"


def _send_framed(sock: socket.socket, payload: bytes) -> bytes:
    """Length-prefixed framing over a socket. Used by Unix socket transport."""

    length = len(payload).to_bytes(4, "big")
    sock.sendall(length + payload)
    header = _recv_exact(sock, 4)
    resp_len = int.from_bytes(header, "big")
    if resp_len > 256 * 1024 * 1024:                 # 256 MB safety cap
        raise ResolutionUnavailable(
            f"Runtime response unreasonably large ({resp_len} bytes); "
            "refusing to read."
        )
    return _recv_exact(sock, resp_len)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    received = 0
    while received < n:
        chunk = sock.recv(n - received)
        if not chunk:
            raise ResolutionUnavailable(
                "Runtime closed the connection before sending a complete reply."
            )
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


class _RuntimeHandle:
    """Holds the active transport and handles discovery + reconnection."""

    def __init__(self) -> None:
        self._transport: Optional[_Transport] = None
        self._lock = threading.Lock()
        self._last_attempt: float = 0.0

    def connect(self) -> None:
        """Discover and open a transport, in priority order."""

        with self._lock:
            if self._transport is not None and self._transport.is_open():
                return
            self._last_attempt = time.time()

            # Priority 1: in-process extension (Enterprise on-prem)
            ip = _InProcessTransport()
            if ip.open():
                self._transport = ip
                logger.info("NoNans: using in-process runtime.")
                return

            # Priority 2: Unix socket (sidecar)
            sock_path = os.environ.get(
                "NONANS_RUNTIME_SOCKET", DEFAULT_RUNTIME_SOCKET
            )
            us = _UnixSocketTransport(sock_path)
            if us.open():
                self._transport = us
                logger.info("NoNans: using sidecar runtime at %s", sock_path)
                return

            # Allow opting out of network in environments that should be
            # detect-only (e.g., air-gapped clusters without the sidecar).
            if os.environ.get("NONANS_OFFLINE", "").lower() in ("1", "true", "yes"):
                self._transport = None
                logger.info(
                    "NoNans: NONANS_OFFLINE set; running in detect-only mode."
                )
                return

            # Priority 3: hosted HTTPS (default)
            url = os.environ.get("NONANS_RUNTIME_URL", DEFAULT_RUNTIME_URL)
            https = _HttpsTransport(url)
            if https.open():
                self._transport = https
                logger.info("NoNans: using hosted runtime at %s", url)
                return

            # Nothing worked. Stay disconnected; wrapper will run detect-only.
            self._transport = None
            logger.info(
                "NoNans: no runtime available; running in detect-only mode."
            )

    def force_disconnect(self) -> None:
        """Drop the current transport. Used by tests; resets discovery."""

        with self._lock:
            if self._transport is not None:
                self._transport.close()
            self._transport = None

    @property
    def is_connected(self) -> bool:
        return self._transport is not None and self._transport.is_open()

    @property
    def transport_name(self) -> str:
        return self._transport.name if self._transport else "none"

    def resolve(
        self,
        event: SingularityEvent,
        tensor_handle: Any,
    ) -> Any:
        """Submit a singularity to the runtime; receive a resolved tensor."""

        if not self.is_connected:
            raise ResolutionUnavailable(
                "NoNans runtime is not connected. Detection still works; "
                "resolution requires a runtime. See https://nonans.com/#pricing."
            )

        # Build the wire payload. Tensor data is sent by handle (a CUDA
        # IPC handle for sidecar/in-process modes, or a small CPU buffer
        # for hosted mode); the actual tensor never traverses a public
        # network unless the customer has explicitly opted in.
        envelope = {
            "version": 1,
            "event": event.to_dict(),
            "tensor": _encode_tensor_handle(tensor_handle),
        }
        payload = json.dumps(envelope, separators=(",", ":")).encode("utf-8")

        assert self._transport is not None
        try:
            response_bytes = self._transport.dispatch(payload)
        except (ResolutionUnavailable, TrialExpired):
            raise
        except Exception as exc:                              # pragma: no cover
            raise ResolutionUnavailable(
                f"Transport error: {exc!r}"
            ) from exc

        try:
            response = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResolutionUnavailable(
                "Runtime returned a non-JSON response."
            ) from exc

        if response.get("status") == "trial_expired":
            raise TrialExpired(
                "NoNans trial token has expired. Contact infra@nonans.com "
                "or upgrade at https://nonans.com/#pricing."
            )
        if response.get("status") != "ok":
            raise ResolutionUnavailable(
                f"Runtime declined the request: "
                f"{response.get('error', 'unknown error')}"
            )
        return response.get("tensor")


def _encode_tensor_handle(tensor_handle: Any) -> dict:
    """Encode a tensor reference for the wire.

    For hosted mode we send only metadata + a small CPU mirror buffer.
    For sidecar/in-process modes the runtime can use the CUDA IPC handle
    directly. The exact encoding is part of the runtime protocol and is
    deliberately not documented in this module.
    """
    if tensor_handle is None:
        return {"kind": "none"}
    if hasattr(tensor_handle, "shape") and hasattr(tensor_handle, "dtype"):
        return {
            "kind": "torch",
            "shape": list(getattr(tensor_handle, "shape", [])),
            "dtype": str(getattr(tensor_handle, "dtype", "")).replace("torch.", ""),
            "device": str(getattr(tensor_handle, "device", "")),
        }
    if isinstance(tensor_handle, dict):
        return {"kind": "dict", **tensor_handle}
    return {"kind": "opaque", "repr": repr(tensor_handle)[:256]}


_handle = _RuntimeHandle()


def get_runtime() -> _RuntimeHandle:
    """Return the global runtime handle, connecting if necessary."""

    if not _handle.is_connected:
        _handle.connect()
    return _handle


def resolution_available() -> bool:
    """Quick check; does not raise. True if a runtime transport is open."""

    try:
        return get_runtime().is_connected
    except Exception:                                         # pragma: no cover
        return False


def transport_name() -> str:
    """Return the name of the active transport, or 'none' if disconnected."""

    return get_runtime().transport_name


def reset_for_tests() -> None:
    """Drop and rediscover the runtime. For tests only."""

    _handle.force_disconnect()
