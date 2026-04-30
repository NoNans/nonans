"""
FSDP integration example.

Run with torchrun:
    torchrun --nproc_per_node=8 examples/fsdp_70b.py

This example demonstrates that nonans.wrap() composes correctly with FSDP.
The wrapper is transparent — it can sit either inside or outside FSDP, and
both placements produce identical detection telemetry.

The model below is a stub; replace the `build_model` function with your
actual 70B+ training setup. The point of the example is the wiring, not
the model.
"""

import os

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import nonans


def build_model() -> nn.Module:
    """Replace with your real model. Stubbed for the example."""
    return nn.Sequential(
        nn.Linear(4096, 4096),
        nn.GELU(),
        nn.Linear(4096, 4096),
    )


def setup_distributed() -> int:
    """Initialize torch.distributed. Returns the local rank."""
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def main() -> None:
    local_rank = setup_distributed()

    model = build_model().to(local_rank)

    # NoNans can sit inside FSDP (per-shard wrapping) or outside it (global
    # wrapping). Outside is simpler and equally effective for v1.
    model = FSDP(model)
    model = nonans.wrap(model, mode="auto")

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(100):
        x = torch.randn(8, 4096, device=local_rank)
        y = model(x)
        loss = y.pow(2).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()
        if local_rank == 0:
            print(f"step {step:4d}  loss={loss.item():.6f}")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
