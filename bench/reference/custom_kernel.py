"""Custom kernel operator workload.

Tests a user-written kernel-like operator that produces a singularity on
specific input distributions. Standard PyTorch raises CUDA error 700;
NoNans resolves the singularity at the kernel boundary.
"""

from __future__ import annotations
from typing import Any, Dict


def run(mode: str) -> Dict[str, Any]:
    import torch
    if mode == "nonans":
        import nonans

    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "needs CUDA", "steps_completed": 0}

    device = "cuda"
    target_steps = 1000

    # A pathological op that simulates the failure mode of a user-written
    # kernel: division by a tensor with sparse zeros. In a real CUDA
    # extension this manifests as CUDA error 700 (illegal memory access)
    # because the NaN propagates through downstream indexing.
    def custom_op(x: torch.Tensor) -> torch.Tensor:
        denom = (x.abs() < 1e-8).float() * 0.0 + (x.abs() >= 1e-8).float() * x
        return x / denom

    if mode == "nonans":
        # For demonstration, wrap a Module that contains the op so the
        # detection layer sees the operator boundary.
        import torch.nn as nn

        class CustomModule(nn.Module):
            def forward(self, x):
                return custom_op(x)

        module = nn.Sequential(CustomModule()).to(device)
        module = nonans.wrap(module, mode="auto")

    steps = 0
    failed_at = None
    try:
        for step in range(target_steps):
            x = torch.randn(1024, device=device)
            x[::100] = 0.0  # sparse zeros
            if mode == "nonans":
                y = module(x)
            else:
                y = custom_op(x)
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
    }
