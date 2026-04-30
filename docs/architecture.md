# NoNans Architecture

This document describes how NoNans is structured, what runs where, and which
parts are open versus closed. It is the document we expect ML infrastructure
engineers and security reviewers to read first.

## Overview

NoNans is a numerical continuity layer that sits between PyTorch (or any
framework that ultimately calls into CUDA kernels) and your GPU. When a
kernel produces a numerical singularity, NoNans intercepts the event,
resolves it inside our framework, and returns a finite, optimizer-coherent
tensor to the GPU. Training continues at the next step.

The system is split into three pieces:

```
┌─────────────────────────────────────────────────────────────────┐
│  YOUR TRAINING / INFERENCE CODE                                 │
│  PyTorch · JAX · vLLM · Megatron · DeepSpeed · FSDP             │
└────────────┬────────────────────────────────────────────────────┘
             │ nonans.wrap(model)
             ▼
┌─────────────────────────────────────────────────────────────────┐
│  DETECTION LAYER         (open source · MIT · this repo)        │
│  · Kernel hooks          · Event taxonomy                       │
│  · Tensor fingerprints   · Telemetry pipeline                   │
└────────────┬────────────────────────────────────────────────────┘
             │ binary IPC
             ▼
┌─────────────────────────────────────────────────────────────────┐
│  RESOLUTION RUNTIME      (proprietary · separate binary)        │
│  · Singularity resolution mechanism (patent pending)            │
│  · Optimizer-state coherence preservation                       │
│  · License enforcement                                          │
└────────────┬────────────────────────────────────────────────────┘
             │ CUDA calls
             ▼
┌─────────────────────────────────────────────────────────────────┐
│  CUDA / GPU HARDWARE                                            │
└─────────────────────────────────────────────────────────────────┘
```

## What's open

The detection layer is fully open-source. It does three things:

1. **Hook installation.** Lightweight hooks on PyTorch tensor operations
   that fire when a kernel produces non-finite values.
2. **Event classification.** A taxonomy of singularity kinds (division by
   zero, gradient overflow, softmax denominator collapse, etc.) and a
   conservative classifier that maps kernel outputs to event records.
3. **Telemetry.** A local ring buffer plus an append-only JSON-lines file
   that records every detected event for debugging and audit.

You can run the detection layer on its own. Without the resolution
runtime, NoNans behaves as a high-quality NaN debugger: it tells you
exactly when, where, and why your run produced a singularity. Many
users find this alone valuable.

## What's closed

The resolution runtime is a separate, licensed binary. It contains:

- The **resolution mechanism**: how a singularity is mapped to a finite,
  optimizer-coherent tensor. This is patent-pending and the trade secret
  of the company.
- A **richer classifier** that handles edge cases the open detector does
  not.
- **License enforcement**: trial-token issuance, expiration, GPU-UUID
  binding.
- **Aggregate telemetry**: opt-in anonymous reporting that feeds the
  cross-customer corpus (used to improve the resolver, never sold raw).

The runtime ships as a Docker image (`ghcr.io/nonans/runtime`) and an
optional Python C++ extension (`nonans-runtime` on a private index).
Customers do not need source access to validate the runtime; the
benchmark harness in this repository is sufficient to reproduce all
published behavior.

## Public protocol surface

The detection layer talks to the runtime over a binary protocol. The
shape of that protocol is intentionally not documented in this repository.
The contract surface that *is* public is the Python API:

```python
import nonans

# 1. Wrap a model.
model = nonans.wrap(model, mode='auto')

# 2. Configure telemetry (optional).
with nonans.configure(telemetry=nonans.LocalTelemetryBackend()):
    train(model)

# 3. Check runtime availability (optional).
if nonans.resolution_available():
    print("Running in resolve mode.")
else:
    print("Running in detect-only mode.")
```

That's the entire public surface. Any change to it is a breaking change
and follows semantic versioning.

## Detection without resolution

When the runtime is not present (`mode='detect_only'` or `'auto'` with no
runtime installed), NoNans acts as a transparent observer. The model
behaves exactly as it did before wrapping. The detection layer records
events but does not interfere with execution.

This is the recommended deployment mode for shadow validation: install
NoNans on a production fleet, leave it in detect-only mode for two weeks,
review the event log, then enable resolution where it adds value.

## Detection with resolution

When the runtime is available (`mode='resolve'` or `'auto'` with runtime
installed), NoNans intercepts singularity events at the kernel boundary
and hands them to the runtime. The runtime returns a resolved tensor that:

- Is finite throughout (no NaN, no ±inf).
- Preserves the optimizer's momentum and second-moment buffer norms.
- Is consistent with the gradient direction implied by the local geometry
  of the loss surface.

The resolution path adds <0.3% per-step overhead on the workloads in our
public benchmark, measured on H100 SXM5 hardware.

## Composition with other infrastructure

NoNans composes cleanly with:

- **FSDP, DeepSpeed, Megatron-LM**: wrap before or after; both work.
- **torch.compile**: NoNans hooks operate on the autograd graph, so
  compiled models work normally. Future versions will integrate at the
  compiler IR level for lower overhead.
- **Mixed precision (AMP)**: AMP and NoNans solve different problems.
  Loss scaling avoids underflow in the loss; NoNans resolves the
  singularities that loss scaling cannot prevent. They work together.
- **Gradient clipping**: clipping bounds gradients post-hoc; NoNans
  resolves the singularity at the moment it arises. They are
  complementary.

## Failure modes

NoNans fails open. If the runtime is unreachable, the detection layer
continues to record events and the model executes as if NoNans were not
installed. We never silently corrupt training. Out-of-scope events
(singularities the runtime cannot handle) are surfaced explicitly in
the telemetry and the customer dashboard.

## Threat model

For security reviewers:

- The detection layer reads tensor metadata (shape, dtype, finite-fraction)
  but never tensor values. Tensor data does not leave the device.
- The runtime IPC channel is local-only by default (Unix socket). TCP is
  available for distributed setups but requires explicit configuration.
- The runtime image is signed; verification instructions are in the
  Pro/Enterprise documentation.
- Trial tokens are bound to a GPU UUID and expire after 30 days. The
  binding logic lives inside the runtime binary; it is not in the open
  detection layer.

For full SOC 2 documentation, contact infra@nonans.com.

## Contact

- **Engineering questions:** infra@nonans.com
- **Security disclosures:** security@nonans.com
