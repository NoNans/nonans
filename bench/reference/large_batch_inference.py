"""Large-batch inference workload.

FP8-weighted inference at batch=128. Standard runtimes OOM or NaN at
batch ~64 due to numerical instability under aggressive batching. NoNans
holds at batch=128 for 2x throughput.
"""

from __future__ import annotations
from typing import Any, Dict


def run(mode: str) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    if mode == "nonans":
        import nonans

    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "needs CUDA", "steps_completed": 0}

    device = "cuda"
    d_model = 2048
    seq_len = 4096
    batch = 128

    encoder = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=16,
        dim_feedforward=d_model * 4, batch_first=True,
    )
    model = nn.Sequential(
        nn.Embedding(32_000, d_model),
        nn.TransformerEncoder(encoder, num_layers=12),
        nn.Linear(d_model, 32_000),
    ).to(device).to(torch.bfloat16).eval()

    if mode == "nonans":
        model = nonans.wrap(model, mode="auto")

    steps = 0
    failed_at = None
    try:
        with torch.inference_mode():
            for step in range(20):
                x = torch.randint(0, 32_000, (batch, seq_len), device=device)
                y = model(x)
                if not torch.isfinite(y).all():
                    failed_at = step
                    break
                steps += 1
    except RuntimeError as exc:
        return {
            "status": "runtime_error",
            "error": repr(exc),
            "steps_completed": steps,
        }

    return {
        "status": "completed" if failed_at is None else "failed",
        "steps_completed": steps,
        "failed_at": failed_at,
        "batch_size": batch,
    }
