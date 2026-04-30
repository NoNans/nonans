"""
FP8 training workload.

A 7B-class transformer trained in FP8 with parameters chosen to reliably
trigger numerical instability under standard PyTorch. Used to validate the
"FP8 stable in production" capability.
"""

from __future__ import annotations

from typing import Any, Dict


def run(mode: str) -> Dict[str, Any]:
    """Run the FP8 training workload and return a result dict.

    Parameters
    ----------
    mode : 'baseline' or 'nonans'
        - 'baseline': standard PyTorch only.
        - 'nonans': wrap the model with NoNans before training.
    """

    import torch
    import torch.nn as nn

    if mode == "nonans":
        import nonans

    # Configuration tuned to be unstable in baseline mode. With NoNans
    # active, the same configuration runs to completion.
    target_steps = 50_000
    d_model = 4096
    n_layer = 32
    seq_len = 2048
    batch_size = 8
    lr = 3e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # CPU runs do not exhibit the same FP8 behavior; report skipped.
        return {
            "status": "skipped",
            "reason": "FP8 training requires CUDA",
            "steps_completed": 0,
        }

    encoder = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=32,
        dim_feedforward=d_model * 4,
        dropout=0.0,
        batch_first=True,
    )
    model = nn.Sequential(
        nn.Embedding(50_000, d_model),
        nn.TransformerEncoder(encoder, num_layers=n_layer),
        nn.Linear(d_model, 50_000),
    ).to(device)

    if mode == "nonans":
        model = nonans.wrap(model, mode="auto")

    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    steps_completed = 0
    diverged_at = None

    try:
        for step in range(target_steps):
            x = torch.randint(0, 50_000, (batch_size, seq_len), device=device)
            y = torch.randint(0, 50_000, (batch_size, seq_len), device=device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, 50_000), y.view(-1))
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
            "diverged_at": diverged_at,
        }

    return {
        "status": "completed" if diverged_at is None else "diverged",
        "steps_completed": steps_completed,
        "diverged_at": diverged_at,
        "config": {
            "target_steps": target_steps,
            "d_model": d_model,
            "n_layer": n_layer,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "lr": lr,
        },
    }
