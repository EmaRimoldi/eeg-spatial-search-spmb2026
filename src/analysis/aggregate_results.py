"""
Aggregate experiment results from run directories.

Scans results/logs/ for completed runs, extracts metrics,
and builds summary tables for analysis.

Usage:
    python src/analysis/aggregate_results.py --experiment pilot
    python src/analysis/aggregate_results.py --experiment core_ablation
    python src/analysis/aggregate_results.py --all
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"


def parse_run_name(run_name: str) -> dict:
    """
    Parse a run directory name into its components.

    Format: {dataset}_{backbone}_{variant}_{regime}_seed{seed}_{transfer}_{timestamp}
    """
    parts = run_name.split("_")
    info = {"run_name": run_name}

    # Best-effort parsing
    if len(parts) >= 5:
        info["dataset"] = parts[0]
        info["backbone"] = parts[1]
        info["spatial_variant"] = parts[2]
        info["freeze_policy"] = parts[3]

        # Find seed
        for p in parts:
            if p.startswith("seed"):
                try:
                    info["seed"] = int(p[4:])
                except ValueError:
                    pass

    return info


def load_run_metrics(run_dir: Path) -> Optional[dict]:
    """Load metrics.json from a run directory."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except Exception:
        return None


def load_run_config(run_dir: Path) -> Optional[dict]:
    """Load config_resolved.yaml from a run directory."""
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def scan_runs(experiment_filter: Optional[str] = None) -> list[dict]:
    """
    Scan results/logs/ for all completed runs.

    Returns:
        List of dicts with run metadata + metrics.
    """
    if not LOGS_DIR.exists():
        print(f"No logs directory found at {LOGS_DIR}")
        return []

    run_dirs = [d for d in LOGS_DIR.iterdir() if d.is_dir()]
    runs = []

    for run_dir in sorted(run_dirs):
        # Skip non-run directories (logs, etc.)
        if not (run_dir / "metrics.json").exists():
            continue

        # Apply experiment filter
        if experiment_filter and experiment_filter not in run_dir.name:
            # Try to check via config
            pass

        metrics = load_run_metrics(run_dir)
        if metrics is None:
            continue

        config = load_run_config(run_dir) or {}
        run_info = parse_run_name(run_dir.name)
        run_info.update({
            "run_dir": str(run_dir),
            "config": config,
            **{f"test_{k}": v for k, v in metrics.get("test", {}).items()},
            "best_val_bacc": metrics.get("best_val", {}).get("balanced_accuracy"),
            "best_epoch": metrics.get("best_val", {}).get("epoch"),
        })
        runs.append(run_info)

    return runs


def build_summary_table(runs: list[dict]) -> list[dict]:
    """Build a summary table from individual run records."""
    import statistics
    from collections import defaultdict

    # Group by (backbone, spatial_variant, freeze_policy, dataset)
    groups = defaultdict(list)
    for r in runs:
        key = (
            r.get("backbone", "unknown"),
            r.get("spatial_variant", "unknown"),
            r.get("freeze_policy", "unknown"),
            r.get("dataset", "unknown"),
        )
        groups[key].append(r)

    summary = []
    for (backbone, variant, regime, dataset), group_runs in sorted(groups.items()):
        baccs = [r.get("test_balanced_accuracy") for r in group_runs if r.get("test_balanced_accuracy") is not None]
        accs = [r.get("test_accuracy") for r in group_runs if r.get("test_accuracy") is not None]

        row = {
            "backbone": backbone,
            "spatial_variant": variant,
            "freeze_policy": regime,
            "dataset": dataset,
            "n_seeds": len(group_runs),
        }

        if baccs:
            row["bacc_mean"] = statistics.mean(baccs)
            row["bacc_std"] = statistics.stdev(baccs) if len(baccs) > 1 else 0.0
        if accs:
            row["acc_mean"] = statistics.mean(accs)
            row["acc_std"] = statistics.stdev(accs) if len(accs) > 1 else 0.0

        summary.append(row)

    return summary


def save_csv(data: list[dict], path: Path):
    """Save list of dicts to CSV."""
    if not data:
        return
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved {len(data)} rows to {path}")


def print_summary_table(summary: list[dict]):
    """Print a formatted summary table."""
    if not summary:
        print("No results to display.")
        return

    print("\n" + "=" * 90)
    print(f"{'Backbone':<12} {'Variant':<22} {'Regime':<14} {'N':<4} {'BAccuracy':<16} {'Accuracy'}")
    print("-" * 90)

    for row in sorted(summary, key=lambda r: (-r.get("bacc_mean", 0), r.get("spatial_variant", ""))):
        bacc = row.get("bacc_mean")
        bacc_std = row.get("bacc_std", 0)
        acc = row.get("acc_mean")

        bacc_str = f"{bacc:.4f} ± {bacc_std:.4f}" if bacc is not None else "N/A"
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"

        print(
            f"{row['backbone']:<12} "
            f"{row['spatial_variant']:<22} "
            f"{row['freeze_policy']:<14} "
            f"{row['n_seeds']:<4} "
            f"{bacc_str:<16} "
            f"{acc_str}"
        )
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(TABLES_DIR))
    args = parser.parse_args()

    print(f"Scanning {LOGS_DIR}...")
    runs = scan_runs(experiment_filter=args.experiment)

    if not runs:
        print("No completed runs found.")
        return

    print(f"Found {len(runs)} completed runs")

    # Build summary
    summary = build_summary_table(runs)
    print_summary_table(summary)

    # Save outputs
    output_dir = Path(args.output_dir)
    prefix = args.experiment or "all"

    save_csv(runs, output_dir / f"{prefix}_raw_results.csv")
    save_csv(summary, output_dir / f"{prefix}_summary.csv")

    # Save LaTeX table
    save_latex_table(summary, output_dir / f"{prefix}_table.tex")


def save_latex_table(summary: list[dict], path: Path):
    """Generate a LaTeX table from summary results."""
    if not summary:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Spatial Ablation Results}\n")
        f.write("\\begin{tabular}{llll rr}\n")
        f.write("\\hline\n")
        f.write("Backbone & Spatial Variant & Regime & N & BAccuracy & Accuracy \\\\\n")
        f.write("\\hline\n")

        for row in sorted(summary, key=lambda r: (-r.get("bacc_mean", 0))):
            bacc = row.get("bacc_mean")
            bacc_std = row.get("bacc_std", 0)
            acc = row.get("acc_mean")
            n = row.get("n_seeds", 0)

            bacc_str = f"{bacc:.4f} $\\pm$ {bacc_std:.4f}" if bacc is not None else "—"
            acc_str = f"{acc:.4f}" if acc is not None else "—"

            f.write(
                f"{row['backbone']} & {row['spatial_variant']} & "
                f"{row['freeze_policy']} & {n} & "
                f"{bacc_str} & {acc_str} \\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved LaTeX table to {path}")


if __name__ == "__main__":
    main()
