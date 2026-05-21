"""
Few-shot performance curve plot.

Shows how each spatial variant performs as the fraction of labeled
training data increases from 1% to 100%.

Usage:
    python src/plotting/plot_fewshot.py \
        --results results/tables/few_shot_summary.csv \
        --output results/figures/
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VARIANT_ORDER = ["none", "channel_id", "coords3d", "coords3d_reference", "topology_agnostic"]
VARIANT_COLORS = {
    "none": "#999999",
    "channel_id": "#4878CF",
    "coords3d": "#6ACC65",
    "coords3d_reference": "#D65F5F",
    "topology_agnostic": "#B47CC7",
}
VARIANT_LABELS = {
    "none": "None (ablation)",
    "channel_id": "Channel ID",
    "coords3d": "3D Coords",
    "coords3d_reference": "3D + Reference",
    "topology_agnostic": "Topo-Agnostic",
}

LABEL_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 1.00]
LABEL_FRACTION_STR = ["1%", "5%", "10%", "25%", "100%"]


def plot_fewshot_curves(results: list[dict], output_dir: Path):
    """Plot balanced accuracy vs label fraction for each variant."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import statistics
    except ImportError:
        print("matplotlib/numpy not available — skipping")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        variant = r.get("spatial_variant", "unknown")
        frac = float(r.get("label_fraction", 1.0))
        bacc = r.get("test_balanced_accuracy") or r.get("bacc_mean")
        if bacc is not None:
            grouped[variant][frac].append(float(bacc))

    if not grouped:
        print("No few-shot results found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for variant in VARIANT_ORDER:
        if variant not in grouped:
            continue
        fracs = sorted(grouped[variant].keys())
        means = [statistics.mean(grouped[variant][f]) for f in fracs]
        stds = [statistics.stdev(grouped[variant][f]) if len(grouped[variant][f]) > 1 else 0 for f in fracs]

        ax.errorbar(
            [f * 100 for f in fracs], means, yerr=stds,
            label=VARIANT_LABELS.get(variant, variant),
            color=VARIANT_COLORS.get(variant, "#888888"),
            marker="o", linewidth=2, capsize=4,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training Labels (%)", fontsize=12)
    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_title("Performance vs Label Fraction", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = output_dir / f"fewshot_curves.{ext}"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=str(PROJECT_ROOT / "results/tables/few_shot_summary.csv"))
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "results/figures"))
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    import csv
    with open(results_path) as f:
        results = list(csv.DictReader(f))
    plot_fewshot_curves(results, Path(args.output))


if __name__ == "__main__":
    main()
