"""
Build paper-ready tables from experiment results.

Generates:
- Main ablation table (LaTeX)
- Transfer table (LaTeX)
- Robustness summary (LaTeX)

Usage:
    python scripts/paper/build_tables.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from src.analysis.aggregate_results import scan_runs, build_summary_table, save_csv
    from src.analysis.stat_tests import run_pairwise_tests

    print("=" * 60)
    print("Building paper tables")
    print("=" * 60)

    runs = scan_runs()
    if not runs:
        print("No completed runs found. Exiting.")
        return

    summary = build_summary_table(runs)
    tables_dir = PROJECT_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Save all
    save_csv(summary, tables_dir / "ablation_summary.csv")

    # Generate LaTeX
    print("\nBuilding LaTeX tables...")
    _build_main_table(summary, tables_dir / "table_main_ablation.tex")
    _build_pairwise_table(runs, tables_dir / "table_pairwise_tests.tex")

    print("\nTables saved to:", tables_dir)


def _build_main_table(summary: list[dict], output_path: Path):
    """Main ablation LaTeX table."""
    # Filter to primary conditions
    primary = [
        r for r in summary
        if r.get("freeze_policy") in ("head_only", "frozen")
        and r.get("backbone") == "reve"
    ]

    if not primary:
        print("No primary condition results found for main table")
        return

    from src.analysis.aggregate_results import save_latex_table
    save_latex_table(primary, output_path)


def _build_pairwise_table(runs: list[dict], output_path: Path):
    """Pairwise statistical tests LaTeX table."""
    from collections import defaultdict
    from src.analysis.stat_tests import run_pairwise_tests

    grouped = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r.get("freeze_policy") == "head_only" and r.get("backbone") == "reve":
            variant = r.get("spatial_variant", "?")
            bacc = r.get("test_balanced_accuracy")
            if bacc is not None:
                grouped[r.get("dataset", "?")][variant].append(float(bacc))

    with open(output_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Pairwise Comparisons (Head-Only, REVE)}\n")
        f.write("\\begin{tabular}{ll rrr r}\n")
        f.write("\\hline\n")
        f.write("Variant A & Variant B & $\\Delta$ BAccuracy & $t$ & $p$ & $d$ \\\\\n")
        f.write("\\hline\n")

        for dataset, variant_scores in grouped.items():
            tests = run_pairwise_tests(dict(variant_scores))
            for r in tests:
                delta = r.get("delta", float("nan"))
                t = r.get("t_statistic", float("nan"))
                p = r.get("p_value", float("nan"))
                d = r.get("cohens_d", float("nan"))

                sig = "^{*}" if p is not None and not (p != p) and p < 0.05 else ""
                delta_str = f"{delta:+.4f}" if delta == delta else "—"
                t_str = f"{t:.3f}" if t == t else "—"
                p_str = f"{p:.3f}{sig}" if p == p else "—"
                d_str = f"{d:.3f}" if d == d else "—"

                f.write(
                    f"{r['variant_a']} & {r['variant_b']} & "
                    f"{delta_str} & {t_str} & {p_str} & {d_str} \\\\\n"
                )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved pairwise table to {output_path}")


if __name__ == "__main__":
    main()
