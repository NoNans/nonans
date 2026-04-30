# Quickstart

The 3-minute path from `pip install` to your first captured event.

## Install

```bash
pip install nonans
```

This installs the open-source detection layer and the public client.
The runtime is fetched automatically on first use; you don't need to
do anything separate.

## Wrap your model

```python
import nonans
import torch

model = MyModel().cuda()
model = nonans.wrap(model, mode='auto')
```

That's it. The detection layer is now active. NoNans will record any
numerical singularity events to `./.nonans/events.jsonl` for the duration
of the process.

## See what's happening

After running for a while:

```bash
$ tail -n 5 .nonans/events.jsonl
{"event_id":"...","kind":4,"severity":2,"origin":2,"step":1832,"layer_name":"transformer.layers.7.attn",...}
```

Or programmatically:

```python
from nonans import LocalTelemetryBackend, EventKind

backend = LocalTelemetryBackend()
recent = backend.recent(n=10)
for event in recent:
    print(f"{event.kind.name} at step {event.step} in {event.layer_name}")
```

## Modes

```python
# Auto: detect always; resolve when runtime is available. (default)
model = nonans.wrap(model, mode='auto')

# Detect only: record events but never resolve. Good for shadow validation.
model = nonans.wrap(model, mode='detect_only')

# Resolve: require the runtime; fail fast if not available.
model = nonans.wrap(model, mode='resolve')

# Off: do nothing. Useful for A/B comparisons.
model = nonans.wrap(model, mode='off')
```

## When the trial expires

After 30 days, the runtime stops resolving and the wrapper falls back to
detect-only behavior automatically. Detection telemetry continues to
work indefinitely under the MIT license.

To extend or upgrade:

- **Pro** (usage-based): see https://nonans.com/#pricing
- **Enterprise** (annual contract): infra@nonans.com

## Reproducing the benchmark

```bash
docker pull ghcr.io/nonans/bench:v1.0.4
docker run --gpus all ghcr.io/nonans/bench:v1.0.4 baseline
docker run --gpus all ghcr.io/nonans/bench:v1.0.4 nonans
docker run ghcr.io/nonans/bench:v1.0.4 compare
```

Reports land in `./out/` as JSON and Markdown.

## Next steps

- Read the [architecture doc](./architecture.md) to understand the layer
  layout.
- Browse [examples/](../examples/) for FSDP, DeepSpeed, and vLLM
  integrations.
- See the public benchmark at https://nonans.com/benchmark.html.
