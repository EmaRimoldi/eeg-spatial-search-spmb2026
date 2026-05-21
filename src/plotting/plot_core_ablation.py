"""
Core ablation plot: bar chart with error bars across spatial variants.

Produces paper-ready figures showing balanced accuracy for each
spatial variant under different training regimes.

Usage:
    python src/plotting/plot_core_ablation.py \
        --results results/tables/core_ablation_summary.csv \
        --output results/figures/
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

VARIANT_ORDER = [
    "none",
    "channel_id",
    "coords2d",
    "coords3d",
    "coords3d_distbias",
    "coords3d_reference",
    "topology_agnostic",
]

VARIANT_LABELS = {
    "none": "None",
    "channel_id": "Channel ID",
    "coords2d": "2D Coords",
    "coords3d": "3D Coords",
    "coords3d_distbias": "3D + Dist Bias",
    "coords3d_reference": "3D + Reference",
    "topology_agnostic": "Topo-Agnostic",
}

REGIME_COLORS = {
    "frozen": "#4878CF",       # blue
    "head_only": "#6ACC65",    # green
    "partial": "#D65F5F",      # red
    "full": "#B47CC7",         # purple
}

REGIME_LABELS = {
    "frozen": "Linear Probe",
    "head_only": "Head-Only",
    "partial": "Partial Fine-tune",
    "full": "Full Fine-tune",
}


def load_results(results_path: str) -> list[dict]:
    """Load summary CSV."""
    import csv
    rows = []
    with open(results_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in ("bacc_mean", "bacc_std", "acc_mean", "acc_std"):
                if k in row and row[k]:
                    row[k] = float(row[k])
                else:
                    row[k] = None
            rows.append(row)
    return rows


def plot_core_ablation(
    results: list[dict],
    regime_filter: str = None,
    metric: str = "bacc_mean",
    metric_std: str = "bacc_std",
    output_dir: Path = FIGURES_DIR,
    backbone_filter: str = None,
):
    """Create bar chart of spatial variants."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — saving plot data as JSON only")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get unique regimes
    regimes = list(dict.fromkeys([r["freeze_policy"] for r in results]))
    if regime_filter:
        regimes = [r for r in regimes if r == regime_filter]

    # Filter by backbone
    if backbone_filter:
        results = [r for r in results if r.get("backbone") == backbone_filter]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(VARIANT_ORDER))
    n_regimes = len(regimes)
    width = 0.7 / max(n_regimes, 1)

    for i, regime in enumerate(regimes):
        regime_data = {r["spatial_variant"]: r for r in results if r.get("freeze_policy") == regime}

        means = []
        stds = []
        for variant in VARIANT_ORDER:
            row = regime_data.get(variant)
            means.append(row[metric] if row and row.get(metric) else 0.0)
            stds.append(row[metric_std] if row and row.get(metric_std) else 0.0)

        offset = (i - n_regimes / 2 + 0.5) * width
        color = REGIME_COLORS.get(regime, "#888888")
        label = REGIME_LABELS.get(regime, regime)

        bars = ax.bar(
            x + offset, means, width * 0.9,
            yerr=stds, capsize=3,
            color=color, alpha=0.85, label=label,
            error_kw={"elinewidth": 1.5},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in VARIANT_ORDER], rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_title("Spatial Embedding Ablation Study", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0.2, 0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    stem = f"core_ablation_{'_'.join(regimes) if regimes else 'all'}"
    for ext in ("pdf", "png"):
        path = output_dir / f"{stem}.{ext}"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=str(PROJECT_ROOT / "results/tables/core_ablation_summary.csv"))
    parser.add_argument("--output", type=str, default=str(FIGURES_DIR))
    parser.add_argument("--regime", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run first: python src/analysis/aggregate_results.py --experiment core_ablation")
        return

    results = load_results(str(results_path))
    print(f"Loaded {len(results)} rows from {results_path}")

    plot_core_ablation(
        results=results,
        regime_filter=args.regime,
        output_dir=Path(args.output),
        backbone_filter=args.backbone,
    )


if __name__ == "__main__":
    main()
