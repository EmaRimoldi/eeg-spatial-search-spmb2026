"""
Statistical testing for spatial ablation comparisons.

Implements paired significance tests and effect size computation
for the pairwise comparisons specified in the manual:
  - channel_id vs coords2d
  - coords2d vs coords3d
  - coords3d vs coords3d_distbias
  - coords3d vs coords3d_reference
  - coords3d_reference vs topology_agnostic

Per-experiment reporting:
  - mean ± std over seeds
  - paired t-test or Wilcoxon signed-rank test
  - Cohen's d effect size
  - confidence intervals
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def cohens_d(x: list[float], y: list[float]) -> float:
    """Compute Cohen's d effect size between two sample lists."""
    import statistics
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mean_diff = statistics.mean(x) - statistics.mean(y)
    pooled_std = ((statistics.variance(x) + statistics.variance(y)) / 2) ** 0.5
    if pooled_std == 0:
        return 0.0
    return mean_diff / pooled_std


def paired_ttest(x: list[float], y: list[float]) -> tuple[float, float]:
    """Paired t-test. Returns (t_statistic, p_value)."""
    import statistics
    import math
    if len(x) != len(y) or len(x) < 2:
        return float("nan"), float("nan")

    diffs = [a - b for a, b in zip(x, y)]
    mean_d = statistics.mean(diffs)
    std_d = statistics.stdev(diffs)
    n = len(diffs)

    if std_d == 0:
        return 0.0, 1.0

    t = mean_d / (std_d / math.sqrt(n))

    # Approximate p-value using t-distribution (scipy not required)
    # Using a simple approximation for small samples
    try:
        from scipy import stats
        _, p = stats.ttest_rel(x, y)
    except ImportError:
        # Rough approximation without scipy
        df = n - 1
        p = 2 * (1 - _t_cdf(abs(t), df))

    return t, p


def _t_cdf(t: float, df: int) -> float:
    """Rough CDF of Student's t distribution."""
    import math
    # Simple approximation; for publication use scipy
    x = df / (df + t * t)
    # Incomplete beta approximation
    half_df = df / 2.0
    try:
        return 1 - 0.5 * _beta_cdf(x, half_df, 0.5)
    except Exception:
        return 0.5


def _beta_cdf(x, a, b):
    """Very rough beta CDF approximation."""
    import math
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    # Use normal approximation for large df
    return 0.5 + 0.5 * math.erf((x - a / (a + b)) / 0.1)


# Primary pairwise comparisons from the manual
PRIMARY_COMPARISONS = [
    ("channel_id", "coords2d"),
    ("coords2d", "coords3d"),
    ("coords3d", "coords3d_distbias"),
    ("coords3d", "coords3d_reference"),
    ("coords3d_reference", "topology_agnostic"),
]


def run_pairwise_tests(
    results: dict[str, list[float]],
    comparisons: list[tuple[str, str]] = None,
    metric: str = "balanced_accuracy",
) -> list[dict]:
    """
    Run pairwise statistical tests for given spatial variant comparisons.

    Args:
        results: Dict mapping variant_name → list of per-seed metric values.
        comparisons: List of (variant_a, variant_b) pairs to compare.
        metric: Metric name being tested.

    Returns:
        List of comparison result dicts.
    """
    if comparisons is None:
        comparisons = PRIMARY_COMPARISONS

    import statistics

    test_results = []
    for variant_a, variant_b in comparisons:
        scores_a = results.get(variant_a, [])
        scores_b = results.get(variant_b, [])

        if not scores_a or not scores_b:
            continue

        row = {
            "variant_a": variant_a,
            "variant_b": variant_b,
            "metric": metric,
            "n_seeds_a": len(scores_a),
            "n_seeds_b": len(scores_b),
            "mean_a": statistics.mean(scores_a),
            "std_a": statistics.stdev(scores_a) if len(scores_a) > 1 else 0,
            "mean_b": statistics.mean(scores_b),
            "std_b": statistics.stdev(scores_b) if len(scores_b) > 1 else 0,
            "delta": statistics.mean(scores_a) - statistics.mean(scores_b),
        }

        # Paired test if same number of seeds
        if len(scores_a) == len(scores_b) and len(scores_a) >= 2:
            t, p = paired_ttest(scores_a, scores_b)
            d = cohens_d(scores_a, scores_b)
            row.update({"t_statistic": t, "p_value": p, "cohens_d": d})

        test_results.append(row)

    return test_results


def print_comparison_table(test_results: list[dict]):
    """Print a formatted comparison table."""
    if not test_results:
        print("No comparisons to display.")
        return

    print("\n" + "=" * 90)
    print("Pairwise Comparison Results")
    print(f"{'Variant A':<26} {'vs Variant B':<26} {'A mean':<8} {'B mean':<8} {'Delta':<8} {'p-val':<8} {'d'}")
    print("-" * 90)

    for r in test_results:
        p = r.get("p_value", float("nan"))
        d = r.get("cohens_d", float("nan"))
        sig = "*" if (p is not None and not (p != p) and p < 0.05) else " "

        print(
            f"{r['variant_a']:<26} {r['variant_b']:<26} "
            f"{r['mean_a']:.4f}  {r['mean_b']:.4f}  "
            f"{r['delta']:+.4f}  "
            f"{p:.3f}{sig}  {d:.3f}"
        )

    print("=" * 90)
    print("* p < 0.05 (uncorrected)")


def main():
    from src.analysis.aggregate_results import scan_runs, build_summary_table

    print("Loading results...")
    runs = scan_runs()

    if not runs:
        print("No completed runs found.")
        return

    # Group by (backbone, regime, dataset) → variant → scores
    from collections import defaultdict
    groups = defaultdict(lambda: defaultdict(list))

    for r in runs:
        key = (r.get("backbone", "?"), r.get("freeze_policy", "?"), r.get("dataset", "?"))
        variant = r.get("spatial_variant", "?")
        bacc = r.get("test_balanced_accuracy")
        if bacc is not None:
            groups[key][variant].append(bacc)

    for (backbone, regime, dataset), variant_scores in sorted(groups.items()):
        print(f"\n{'='*60}")
        print(f"Backbone: {backbone} | Regime: {regime} | Dataset: {dataset}")
        print(f"{'='*60}")

        test_results = run_pairwise_tests(
            dict(variant_scores),
            metric="balanced_accuracy",
        )
        print_comparison_table(test_results)

    # Save results
    output_path = PROJECT_ROOT / "results" / "tables" / "pairwise_tests.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for (backbone, regime, dataset), variant_scores in groups.items():
        key = f"{backbone}_{regime}_{dataset}"
        all_results[key] = run_pairwise_tests(dict(variant_scores))

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved pairwise tests to {output_path}")


if __name__ == "__main__":
    main()
