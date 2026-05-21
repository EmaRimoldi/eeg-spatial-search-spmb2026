"""
Robustness plot: accuracy vs channel dropout rate.

Shows how each spatial variant degrades as channels are randomly dropped.

Usage:
    python src/plotting/plot_robustness.py \
        --results results/tables/channel_dropout_summary.csv \
        --output results/figures/
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VARIANT_ORDER = ["channel_id", "coords3d", "coords3d_reference", "topology_agnostic"]
VARIANT_COLORS = {
    "channel_id": "#4878CF",
    "coords3d": "#6ACC65",
    "coords3d_reference": "#D65F5F",
    "topology_agnostic": "#B47CC7",
}
VARIANT_LABELS = {
    "channel_id": "Channel ID",
    "coords3d": "3D Coords",
    "coords3d_reference": "3D + Reference",
    "topology_agnostic": "Topo-Agnostic",
}


def plot_robustness_curves(results: list[dict], output_dir: Path):
    """
    Plot accuracy vs dropout rate for each variant.
    Results must include a 'dropout_prob' field.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by variant and dropout_prob
    from collections import defaultdict
    import statistics

    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        variant = r.get("spatial_variant", "unknown")
        prob = float(r.get("dropout_prob", 0.0))
        bacc = r.get("test_balanced_accuracy") or r.get("bacc_mean")
        if bacc is not None:
            grouped[variant][prob].append(float(bacc))

    if not grouped:
        print("No dropout results found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for variant in VARIANT_ORDER:
        if variant not in grouped:
            continue
        probs = sorted(grouped[variant].keys())
        means = [statistics.mean(grouped[variant][p]) for p in probs]
        stds = [statistics.stdev(grouped[variant][p]) if len(grouped[variant][p]) > 1 else 0 for p in probs]

        ax.errorbar(
            probs, means, yerr=stds,
            label=VARIANT_LABELS.get(variant, variant),
            color=VARIANT_COLORS.get(variant, "#888888"),
            marker="o", linewidth=2, capsize=4,
        )

    ax.set_xlabel("Channel Dropout Rate", fontsize=12)
    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_title("Robustness to Missing Channels", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = output_dir / f"robustness_curves.{ext}"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=str(PROJECT_ROOT / "results/tables/channel_dropout_summary.csv"))
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "results/figures"))
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    import csv
    with open(results_path) as f:
        results = list(csv.DictReader(f))

    plot_robustness_curves(results, Path(args.output))


if __name__ == "__main__":
    main()
