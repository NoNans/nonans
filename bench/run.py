"""
NoNans public benchmark harness.

This is the same script that produces the numbers on
https://nonans.com/benchmark.html. It is bundled in the public Docker image
and runs against any of the workloads in `bench/reference/`.

Usage:
    python bench/run.py baseline    # standard PyTorch
    python bench/run.py nonans      # PyTorch + NoNans
    python bench/run.py compare     # compare ./out/baseline.json vs nonans.json

Each workload is a Python file in `bench/reference/` that exposes a
`run(mode: str) -> dict` function. The harness collects the dict, writes
it to `./out/<mode>.json`, and produces a human-readable markdown report.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

WORKLOADS = [
    "bench.reference.fp8_training",
    "bench.reference.long_context_attention",
    "bench.reference.aggressive_lr",
    "bench.reference.rlhf_grpo",
    "bench.reference.mixed_precision_70b",
    "bench.reference.long_context_inference",
    "bench.reference.large_batch_inference",
    "bench.reference.custom_kernel",
]

OUT_DIR = Path("./out")


def run_workload(module_path: str, mode: str) -> Dict[str, Any]:
    """Run a single benchmark workload and capture its result + timing."""

    started = time.time()
    record: Dict[str, Any] = {
        "workload": module_path,
        "mode": mode,
        "started_at": started,
    }
    try:
        module = importlib.import_module(module_path)
        result = module.run(mode)
        record.update(result)
        record["status"] = result.get("status", "ok")
    except Exception as exc:
        record["status"] = "harness_error"
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
    record["duration_seconds"] = round(time.time() - started, 3)
    return record


def run_all(mode: str) -> List[Dict[str, Any]]:
    """Run every workload in the suite for the given mode."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for w in WORKLOADS:
        print(f"[{mode}] running {w}")
        record = run_workload(w, mode)
        results.append(record)
        print(f"[{mode}] {w} -> {record.get('status')}")

    out_path = OUT_DIR / f"{mode}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")
    return results


def compare() -> None:
    """Render a side-by-side comparison of baseline.json and nonans.json."""

    base_path = OUT_DIR / "baseline.json"
    nn_path = OUT_DIR / "nonans.json"
    if not base_path.exists() or not nn_path.exists():
        sys.exit(
            "Both baseline.json and nonans.json must exist in ./out/. "
            "Run `bench/run.py baseline` and `bench/run.py nonans` first."
        )

    base = {r["workload"]: r for r in json.loads(base_path.read_text())}
    nn = {r["workload"]: r for r in json.loads(nn_path.read_text())}

    rows = []
    for w in WORKLOADS:
        b = base.get(w, {})
        n = nn.get(w, {})
        rows.append(
            {
                "workload": w.split(".")[-1],
                "baseline": b.get("status", "missing"),
                "nonans": n.get("status", "missing"),
                "baseline_steps": b.get("steps_completed"),
                "nonans_steps": n.get("steps_completed"),
            }
        )

    md_path = OUT_DIR / "report.md"
    with open(md_path, "w") as f:
        f.write("# NoNans Benchmark Comparison\n\n")
        f.write("| Workload | Baseline | NoNans | Baseline steps | NoNans steps |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['workload']} | {r['baseline']} | {r['nonans']} | "
                f"{r['baseline_steps']} | {r['nonans_steps']} |\n"
            )

    print(f"Comparison written to {md_path}")
    for r in rows:
        line = (
            f"  {r['workload']:32s}  baseline={r['baseline']:12s}  "
            f"nonans={r['nonans']:12s}"
        )
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="NoNans benchmark harness.")
    parser.add_argument(
        "mode",
        choices=["baseline", "nonans", "compare"],
        help=(
            "baseline: run with standard PyTorch. "
            "nonans: run with the resolution layer active. "
            "compare: render a comparison report from previous runs."
        ),
    )
    args = parser.parse_args()

    if args.mode == "compare":
        compare()
    else:
        run_all(args.mode)


if __name__ == "__main__":
    main()
