# Security Policy

## Reporting a vulnerability

If you discover a security issue in NoNans, please report it privately:

- **Email:** security@nonans.com
- **PGP key:** Available on request from the same address.

Please do not file a public GitHub issue for security-sensitive reports.

We commit to:

1. Acknowledge receipt within **24 hours**.
2. Provide an initial assessment within **72 hours**.
3. Issue a fix or mitigation within **14 days** for confirmed high-severity
   issues, with continuous status updates.
4. Credit the reporter (with permission) in the release notes.

## Scope

In scope:

- The detection layer in this repository.
- The public Python client (`nonans/client.py`, `nonans/wrap.py`).
- The benchmark harness.
- The runtime binary (`ghcr.io/nonans/runtime`, `nonans-runtime`).
- The runtime IPC protocol.

Out of scope:

- Vulnerabilities in third-party dependencies (please report upstream).
- Issues in user code that uses NoNans (please file a regular issue).

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |
| 0.9.x   | Critical fixes only, end-of-life 2026-12 |
| < 0.9   | No        |

## Coordinated disclosure

We follow a 90-day coordinated disclosure window for non-trivial issues.
Public advisories are posted at:

- https://github.com/nonans/nonans/security/advisories
- https://nonans.com/security

For critical issues affecting in-production deployments, we will reach out
to known customers directly before public disclosure.
