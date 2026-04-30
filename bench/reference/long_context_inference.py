"""Long-context inference workload.

1M-token inference with vanilla softmax attention. Standard runtime fails
at sequence ~256K with softmax overflow; NoNans holds through 1M.
"""

from __future__ import annotations
from typing import Any, Dict


def run(mode: str) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    if mode == "nonans":
        import nonans

    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "needs CUDA", "tokens_processed": 0}

    device = "cuda"
    d_model = 1024
    target_seq = 1_000_000

    class VanillaAttn(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.q = nn.Linear(d, d, bias=False)
            self.k = nn.Linear(d, d, bias=False)
            self.v = nn.Linear(d, d, bias=False)
            self.scale = d ** -0.5
        def forward(self, x):
            q, k, v = self.q(x), self.k(x), self.v(x)
            return (q @ k.transpose(-2, -1) * self.scale).softmax(-1) @ v

    model = VanillaAttn(d_model).to(device).eval()
    if mode == "nonans":
        model = nonans.wrap(model, mode="auto")

    tokens_processed = 0
    failed_at = None
    chunk = 16_384
    try:
        with torch.inference_mode():
            for start in range(0, target_seq, chunk):
                length = min(chunk, target_seq - start)
                x = torch.randn(1, length, d_model, device=device)
                y = model(x)
                if not torch.isfinite(y).all():
                    failed_at = start + length
                    break
                tokens_processed = start + length
    except RuntimeError as exc:
        return {
            "status": "runtime_error",
            "error": repr(exc),
            "tokens_processed": tokens_processed,
        }

    return {
        "status": "completed" if failed_at is None else "failed",
        "tokens_processed": tokens_processed,
        "failed_at": failed_at,
        "steps_completed": tokens_processed,
    }
