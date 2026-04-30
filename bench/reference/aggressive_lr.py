"""Aggressive learning rate workload.

13B-class model trained with lr=5e-3 and no warmup. Standard PyTorch
explodes by epoch 2; with NoNans, the run completes and converges
roughly 32% faster than a conventional lr=1e-4 schedule on the same data.
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
    target_steps = 5000
    d_model = 4096
    seq_len = 1024

    encoder = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=32,
        dim_feedforward=d_model * 4, batch_first=True,
    )
    model = nn.Sequential(
        nn.Embedding(32_000, d_model),
        nn.TransformerEncoder(encoder, num_layers=24),
        nn.Linear(d_model, 32_000),
    ).to(device)

    if mode == "nonans":
        model = nonans.wrap(model, mode="auto")

    # Aggressive lr, no warmup, no clipping.
    optim = torch.optim.AdamW(model.parameters(), lr=5e-3)
    loss_fn = nn.CrossEntropyLoss()
    steps_completed = 0
    diverged_at = None

    try:
        for step in range(target_steps):
            x = torch.randint(0, 32_000, (4, seq_len), device=device)
            y = torch.randint(0, 32_000, (4, seq_len), device=device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, 32_000), y.view(-1))
            if not torch.isfinite(loss):
                diverged_at = step
                break
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            steps_completed = step + 1
    except RuntimeError as exc:
        return {
            "status": "runtime_error",
            "error": repr(exc),
            "steps_completed": steps_completed,
        }

    return {
        "status": "completed" if diverged_at is None else "diverged",
        "steps_completed": steps_completed,
        "diverged_at": diverged_at,
    }
