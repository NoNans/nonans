"""
Minimal PyTorch training example with NoNans.

Run:
    python examples/pytorch_minimal.py

This example trains a tiny transformer on synthetic data with deliberately
unstable settings (high learning rate, no warmup, mixed precision) so the
detection layer surfaces real events. Without resolution, the run will
fail; with the licensed runtime present, it completes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import nonans


def build_unstable_model(d_model: int = 256, n_layer: int = 4) -> nn.Module:
    """A small transformer with intentionally aggressive initialization
    so we hit numerical edges quickly. Useful for a 60-second demo."""
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=8,
        dim_feedforward=d_model * 4,
        dropout=0.1,
        batch_first=True,
    )
    model = nn.Sequential(
        nn.Embedding(32_000, d_model),
        nn.TransformerEncoder(encoder_layer, num_layers=n_layer),
        nn.Linear(d_model, 32_000),
    )
    # Aggressive init increases the chance of hitting a singularity.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.normal_(p, mean=0.0, std=0.5)
    return model


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "Warning: running on CPU. Numerical-stability events behave "
            "differently here; this example is meant for GPU."
        )

    model = build_unstable_model().to(device)
    model = nonans.wrap(model, mode="auto")

    optim = torch.optim.AdamW(model.parameters(), lr=5e-3)

    inputs = torch.randint(0, 32_000, (64, 128)).to(device)
    targets = torch.randint(0, 32_000, (64, 128)).to(device)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)

    loss_fn = nn.CrossEntropyLoss()

    for step, (x, y) in enumerate(loader):
        logits = model(x)
        loss = loss_fn(logits.view(-1, 32_000), y.view(-1))
        loss.backward()
        optim.step()
        optim.zero_grad()
        print(f"step {step:4d}  loss={loss.item():.4f}")

    print("Training complete. See ./.nonans/events.jsonl for the event log.")


if __name__ == "__main__":
    main()
