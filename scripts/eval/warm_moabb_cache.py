#!/usr/bin/env python3
"""Warm a MOABB dataset cache for selected splits.

This is intended as a one-off Slurm bootstrap step so transfer-eval jobs do not
all download/process the same dataset concurrently.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import DATASET_CONFIGS, load_moabb_dataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Warm cached MOABB splits")
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=["train", "val", "test"],
        help="Which splits to materialize into cache.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Warming cache for dataset={args.dataset} splits={args.splits}")
    load_moabb_dataset(
        args.dataset,
        batch_size=args.batch_size,
        config={"num_classes": 4},
        split_names=tuple(args.splits),
    )
    print("Cache warm complete.")


if __name__ == "__main__":
    main()
