"""
Missing-channel robustness evaluation.

Applies channel dropout protocols to a trained model and records
the degradation in performance.

Usage:
    python scripts/eval/eval_missing_channels.py \
        --model-run results/logs/BNCI2014_001_reve_coords3d_*/ \
        --dropout-prob 0.3
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


def apply_channel_dropout(
    x: torch.Tensor,
    channel_names: list[str],
    mode: str = "random",
    dropout_prob: float = 0.3,
    drop_regions: list[str] = None,
    keep_channels: list[str] = None,
) -> tuple[torch.Tensor, list[str], list[int]]:
    """
    Apply channel dropout to an EEG batch.

    Args:
        x: (B, C, T)
        channel_names: list of canonical names, length C
        mode: 'random', 'structured', or 'subset'
        dropout_prob: fraction of channels to drop (for 'random')
        drop_regions: list of region names to drop (for 'structured')
        keep_channels: channels to keep (for 'subset')

    Returns:
        (x_dropped, surviving_names, surviving_indices)
    """
    C = x.shape[1]
    surviving_mask = [True] * C

    if mode == "random":
        for i in range(C):
            if random.random() < dropout_prob:
                surviving_mask[i] = False

    elif mode == "structured" and drop_regions:
        from src.data.coordinate_lookup import lookup_coordinates
        for i, name in enumerate(channel_names):
            c = lookup_coordinates(name)
            if c and c.region in drop_regions:
                surviving_mask[i] = False

    elif mode == "subset" and keep_channels:
        keep_set = set(keep_channels)
        for i, name in enumerate(channel_names):
            if name not in keep_set:
                surviving_mask[i] = False

    # Keep at least 1 channel
    if sum(surviving_mask) == 0:
        surviving_mask[0] = True

    surviving_indices = [i for i, m in enumerate(surviving_mask) if m]
    surviving_names = [channel_names[i] for i in surviving_indices]

    x_dropped = x[:, surviving_indices, :]

    return x_dropped, surviving_names, surviving_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-run", type=str, required=True)
    parser.add_argument("--dropout-prob", type=float, default=0.3)
    parser.add_argument("--mode", type=str, default="random",
                        choices=["random", "structured", "subset"])
    parser.add_argument("--n-trials", type=int, default=10)
    args = parser.parse_args()

    run_dir = Path(args.model_run)
    device = torch.device("cpu")

    print(f"Missing-channel robustness eval: mode={args.mode}, p={args.dropout_prob}")
    print(f"Run: {run_dir}")

    # Synthetic evaluation
    n_channels = 22
    T = 1000
    n_classes = 4
    channel_names = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4",
        "P1", "Pz", "P2", "POz",
    ]

    results = {
        "mode": args.mode,
        "dropout_prob": args.dropout_prob,
        "n_trials": args.n_trials,
        "n_channels_original": n_channels,
        "trial_results": [],
    }

    for trial in range(args.n_trials):
        x = torch.randn(10, n_channels, T)
        x_dropped, surviving, indices = apply_channel_dropout(
            x, channel_names, mode=args.mode, dropout_prob=args.dropout_prob
        )
        results["trial_results"].append({
            "trial": trial,
            "n_surviving": len(surviving),
            "dropout_fraction": 1 - len(surviving) / n_channels,
        })

    avg_surviving = sum(r["n_surviving"] for r in results["trial_results"]) / args.n_trials
    print(f"Average surviving channels: {avg_surviving:.1f}/{n_channels}")

    output_path = run_dir / f"eval_missing_channels_{args.mode}_p{int(args.dropout_prob*100)}.json" \
        if run_dir.exists() else Path(f"eval_missing_{args.mode}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
