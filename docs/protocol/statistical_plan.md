# Statistical Analysis Plan

## Per-experiment reporting

For each (backbone, variant, regime, dataset) cell:
- **Mean** over seeds
- **Standard deviation** over seeds
- **95% CI** (when ≥5 seeds)
- **Paired t-test** vs each comparison partner (same seeds)
- **Cohen's d** effect size

## Primary pairwise comparisons

| Comparison | Scientific question |
|-----------|---------------------|
| `channel_id` vs `coords2d` | Do 2D coordinates help over fixed ID? |
| `coords2d` vs `coords3d` | Does 3D beat 2D? |
| `coords3d` vs `coords3d_distbias` | Does pairwise distance bias add value? |
| `coords3d` vs `coords3d_reference` | Does reference encoding matter? |
| `coords3d_reference` vs `topology_agnostic` | Is robustness vs geometry? |

## Required plots

1. **Core ablation bar chart** — balanced accuracy per variant ± std (all regimes)
2. **Robustness curves** — balanced accuracy vs channel dropout rate
3. **Few-shot learning curves** — balanced accuracy vs label fraction (log scale)
4. **Transfer gap heatmap** — performance degradation under layout shift
5. **Regime comparison matrix** — head_only vs partial vs linear_probe

## Statistical thresholds

- Primary criterion: **balanced accuracy** (handles class imbalance)
- Significance: **p < 0.05** (paired t-test, uncorrected; flag all comparisons)
- Effect size: **Cohen's d** interpretation: small=0.2, medium=0.5, large=0.8
- Bonferroni correction applied when reporting multiple comparisons in paper

## Result interpretation decision tree

See `Phase 16` in the project manual for full interpretation logic.

Key scenarios:
1. `coords3d` clearly best → explicit 3D geometry matters
2. `coords3d_reference` > `coords3d` → reference scheme matters
3. `topology_agnostic` ≈ `coords3d_reference` → it's about robustness, not geometry
4. Effects under head_only > full fine-tune → geometry is a prior / data-efficiency aid
