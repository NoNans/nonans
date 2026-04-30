"""Long-context attention workload.

Tests vanilla softmax attention at sequence length 256K, where standard
PyTorch produces NaN immediately due to denominator collapse.
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
    seq_len = 262_144         # 256K
    d_model = 1024
    target_steps = 10_000

    class VanillaAttn(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.q = nn.Linear(d, d, bias=False)
            self.k = nn.Linear(d, d, bias=False)
            self.v = nn.Linear(d, d, bias=False)
            self.scale = d ** -0.5
        def forward(self, x):
            q, k, v = self.q(x), self.k(x), self.v(x)
            scores = (q @ k.transpose(-2, -1)) * self.scale
            attn = scores.softmax(dim=-1)
            return attn @ v

    model = VanillaAttn(d_model).to(device)
    if mode == "nonans":
        model = nonans.wrap(model, mode="auto")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    steps_completed = 0
    diverged_at = None

    try:
        for step in range(target_steps):
            x = torch.randn(1, seq_len, d_model, device=device)
            y = model(x)
            loss = y.pow(2).mean()
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
    }
