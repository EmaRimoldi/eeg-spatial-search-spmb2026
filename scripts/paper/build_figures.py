"""
Build all paper figures from experiment results.

Usage:
    python scripts/paper/build_figures.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building paper figures")
    print("=" * 60)

    # Core ablation plot
    ablation_path = tables_dir / "core_ablation_summary.csv"
    if ablation_path.exists():
        print("\nBuilding core ablation plot...")
        from src.plotting.plot_core_ablation import load_results, plot_core_ablation
        results = load_results(str(ablation_path))
        plot_core_ablation(results, output_dir=figures_dir)
    else:
        print(f"Core ablation results not found: {ablation_path}")
        print("Run: python src/analysis/aggregate_results.py --experiment core_ablation")

    # Robustness plot
    dropout_path = tables_dir / "channel_dropout_summary.csv"
    if dropout_path.exists():
        print("\nBuilding robustness plot...")
        from src.plotting.plot_robustness import load_results_csv, plot_robustness_curves
        import csv
        with open(dropout_path) as f:
            results = list(csv.DictReader(f))
        plot_robustness_curves(results, figures_dir)
    else:
        print(f"Channel dropout results not found: {dropout_path}")

    # Few-shot plot
    fewshot_path = tables_dir / "few_shot_summary.csv"
    if fewshot_path.exists():
        print("\nBuilding few-shot plot...")
        from src.plotting.plot_fewshot import plot_fewshot_curves
        import csv
        with open(fewshot_path) as f:
            results = list(csv.DictReader(f))
        plot_fewshot_curves(results, figures_dir)
    else:
        print(f"Few-shot results not found: {fewshot_path}")

    print(f"\nFigures directory: {figures_dir}")


if __name__ == "__main__":
    main()
