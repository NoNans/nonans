"""RLHF / GRPO post-training workload.

Simulates the most numerically unstable regime in modern ML: KL-penalized
policy optimization with reward variance and gradient norms swinging across
orders of magnitude in a single step. Five independent seeds; baseline
typically collapses on 3 of 5.
"""

from __future__ import annotations
from typing import Any, Dict, List


def run(mode: str) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    if mode == "nonans":
        import nonans

    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "needs CUDA", "steps_completed": 0}

    device = "cuda"
    seeds = [0, 1, 2, 3, 4]
    target_steps = 2000
    d_model = 2048

    completions: List[Dict[str, Any]] = []

    for seed in seeds:
        torch.manual_seed(seed)
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=16,
            dim_feedforward=d_model * 4, batch_first=True,
        )
        policy = nn.Sequential(
            nn.Embedding(8000, d_model),
            nn.TransformerEncoder(encoder, num_layers=12),
            nn.Linear(d_model, 8000),
        ).to(device)
        if mode == "nonans":
            policy = nonans.wrap(policy, mode="auto")

        optim = torch.optim.AdamW(policy.parameters(), lr=1e-5)
        steps_completed = 0
        diverged_at = None

        try:
            for step in range(target_steps):
                x = torch.randint(0, 8000, (4, 256), device=device)
                logits = policy(x)
                logp = logits.log_softmax(dim=-1)

                # Synthetic reward + KL penalty signal — produces gradient
                # norm swings characteristic of GRPO.
                reward = torch.randn(4, 256, device=device) * 5.0
                ref_logp = logp.detach() - 0.1
                kl = (logp.exp() * (logp - ref_logp)).sum(-1)

                advantage = reward - reward.mean()
                pg_loss = -(advantage.unsqueeze(-1) * logp).mean()
                loss = pg_loss + 0.05 * kl.mean()

                if not torch.isfinite(loss):
                    diverged_at = step
                    break
                loss.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)
                steps_completed = step + 1
        except RuntimeError as exc:
            completions.append(
                {"seed": seed, "status": "runtime_error", "steps": steps_completed}
            )
            continue

        completions.append(
            {
                "seed": seed,
                "status": "completed" if diverged_at is None else "diverged",
                "steps": steps_completed,
                "diverged_at": diverged_at,
            }
        )

    successful = sum(1 for c in completions if c["status"] == "completed")
    return {
        "status": "summary",
        "completions": completions,
        "successful_runs": successful,
        "total_runs": len(seeds),
        "steps_completed": sum(c.get("steps", 0) for c in completions),
    }
