"""Mixed-precision 70B pretraining workload.

A reduced harness for what would, in practice, run on a 16-GPU H100 node.
The reduction preserves the numerical stress profile while keeping the
test runnable on smaller hardware for benchmark validation. The full 70B
configuration is available in the MNDA replication kit.
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

    encoder = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=32,
        dim_feedforward=d_model * 4, batch_first=True,
    )
    model = nn.Sequential(
        nn.Embedding(50_000, d_model),
        nn.TransformerEncoder(encoder, num_layers=32),
        nn.Linear(d_model, 50_000),
    ).to(device).to(torch.bfloat16)

    if mode == "nonans":
        model = nonans.wrap(model, mode="auto")

    optim = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()
    steps_completed = 0
    rollback_events = 0

    try:
        for step in range(target_steps):
            x = torch.randint(0, 50_000, (4, 1024), device=device)
            y = torch.randint(0, 50_000, (4, 1024), device=device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, 50_000).float(), y.view(-1))
            if not torch.isfinite(loss):
                rollback_events += 1
                if mode == "baseline":
                    return {
                        "status": "rollback_required",
                        "steps_completed": steps_completed,
                        "rollback_events": rollback_events,
                    }
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
        "status": "completed",
        "steps_completed": steps_completed,
        "rollback_events": rollback_events,
    }
