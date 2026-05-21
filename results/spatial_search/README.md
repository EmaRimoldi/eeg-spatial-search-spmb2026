# EEG-FM-Bench Spatial Search

This directory records staged coordinate-only searches on the fixed EEG-FM-Bench
processed datasets: `bcic_2a`, `motor_mv_img`, and `workload`.

Baseline `none` and `channel_id` runs are not generated here; those baselines
are fixed externally. Final promoted comparisons use operational seeds
`42,43,44,45,46`.

## Stages

- `stage1_screen`: broad seed-42 screen over coordinate-based variants only.
- `stage2_promote`: generated from the stage-1 leaderboard. By default it adds
  seeds `43,44,45,46`, treating the matching seed-42 stage-1 run as the first
  final-comparison seed.

## Current Manifest

- Manifest: `results/spatial_search/manifest.tsv`
- Planned rows: 78
- stage1_screen / biot: 12
- stage1_screen / cbramod: 42
- stage1_screen / labram: 24

## Commands

Generate the stage-1 manifest and configs:

```bash
python scripts/train/eegfm_spatial_search.py generate --stage stage1_screen --force
```

Inspect a submission without launching jobs:

```bash
python scripts/train/eegfm_spatial_search.py submit --stage stage1_screen --dry-run --limit 3
```

Aggregate status and leaderboard:

```bash
python scripts/train/summarize_eegfm_spatial_search.py --results-root results/spatial_search
```

Generate stage-2 promotion rows after stage-1 metrics exist:

```bash
python scripts/train/eegfm_spatial_search.py promote --top-k 1
```
