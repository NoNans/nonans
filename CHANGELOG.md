# Changelog

All notable changes to the public NoNans package will be documented here.
The runtime binary maintains a separate changelog distributed with the
runtime image.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.4] — 2026-04-15

### Added
- Public benchmark Docker image (`ghcr.io/nonans/bench:v1.0.4`) with eight
  reference workloads.
- `nonans.configure(...)` context manager for scoped telemetry settings.
- vLLM inference integration example.
- Architecture documentation in `docs/architecture.md`.

### Changed
- Detection layer overhead reduced approximately 40% on the hot path.
- Event taxonomy stabilized at 13 kinds; ordinality is now part of the
  public API.

### Fixed
- Edge case where the wrapper would fail to forward `__getattr__` lookups
  for FSDP-wrapped models.

## [1.0.3] — 2026-03-20

### Added
- DeepSpeed integration example.
- `mode='off'` wrapper option for A/B comparisons.

### Changed
- Trial token issuance moved to first kernel call (was first import) to
  avoid spurious activation during static analysis.

## [1.0.2] — 2026-02-12

### Added
- FSDP integration example.
- `nonans.resolution_available()` predicate.

## [1.0.1] — 2026-01-08

### Fixed
- `LocalTelemetryBackend` now creates the `.nonans/` directory if absent.

## [1.0.0] — 2025-12-15

### Added
- Initial public release.
- Detection layer (open source, MIT).
- Public Python client.
- Resolution runtime via `ghcr.io/nonans/runtime` (separate license).
