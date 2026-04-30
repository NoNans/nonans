# NoNans Runtime — Commercial License

The NoNans **resolution runtime** (distributed as the Docker image
`ghcr.io/nonans/runtime` and the optional Python extension `nonans-runtime`)
is proprietary software, separately licensed from the open-source detection
layer in this repository.

This document is a plain-language summary of the runtime license. The
binding legal terms are issued with each commercial agreement and supersede
this summary in case of conflict.

## Free trial

A 30-day trial token is issued automatically the first time the runtime
runs on a given host. The token:

- Is bound to the GPU UUID of the host that requested it.
- Expires 30 days after issuance.
- Is rate-limited to 8 GPUs per host.
- Includes full functional parity with paid tiers; no gating during trial.

## Paid tiers

| Tier | What you can do | How you pay |
|------|----------------|-------------|
| **Pro** | Unlimited GPUs, FSDP/DeepSpeed/Megatron support, capability tier (FP8, long context, RL stability), telemetry dashboard, priority engineering support | Usage-based: $0.50 per protected GPU-hour, billed monthly. Cloud marketplace billing supported (AWS, GCP, Azure). |
| **Enterprise** | Pro features plus driver-level integration, on-premise deployment, SLA-backed uptime, custom kernel extensions, MNDA + IP indemnification | Custom annual contract. From $250K ARR. Talk to engineering. |

## What you may not do

- Reverse-engineer the runtime binary.
- Redistribute the runtime image or extension to third parties.
- Use the runtime to provide a competing numerical-continuity service to
  third parties without an explicit reseller agreement.
- Strip or modify the trial-token enforcement logic.

## What we commit to

- Patches and updates within the major version for the duration of your
  subscription.
- 99.9% runtime uptime SLA on Pro; bespoke SLA terms on Enterprise.
- Security advisories within 24 hours of confirmed vulnerability.
- Source-code escrow for Enterprise customers, releasable on
  agreement-defined trigger events.

## Contact

For pricing, MNDA, or contract questions: **infra@nonans.com**
For security disclosures: **security@nonans.com**

© 2026 NoNans, Inc. · Patent Pending
