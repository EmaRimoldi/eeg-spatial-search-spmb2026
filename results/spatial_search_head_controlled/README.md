# EEG-FM-Bench Spatial Search: Head-Controlled CBraMod

This directory records the head-controlled CBraMod rerun for the EEG-FM-Bench
processed datasets where the earlier staged spatial search used a different
adaptation head than the 5-seed baseline: `motor_mv_img` and `workload`.

Baseline `none` and `channel_id` runs are not generated here; those baselines
are fixed externally. Final promoted comparisons use operational seeds
`42,43,44,45,46`.

## Protocol

- Data: local EEG-FM-Bench processed splits loaded by `src/data/eegfm_loader.py`.
- Backbone: CBraMod only.
- Adaptation head: explicit `avg_pool` with `hidden_dims=[128]` and
  `dropout=0.3`, recorded in each row's `hparams_json`.
- Motivation: match the generated 5-seed baseline configs under
  `configs/eegfm_bench/generated/{physiomi,workload}_table2_5seed/` and avoid
  mixing `flatten_mlp` spatial runs against `avg_pool` baseline runs.

## Stages

- `stage1_screen`: broad seed-42 screen over coordinate-based variants only.
- `stage2_promote`: generated from the stage-1 leaderboard. By default it adds
  seeds `43,44,45,46`, treating the matching seed-42 stage-1 run as the first
  final-comparison seed.

## Current Manifest

- Manifest: `results/spatial_search_head_controlled/manifest.tsv`
- Planned rows: 28
- stage1_screen / cbramod: 28

## Commands

Generate the stage-1 manifest and configs:

```bash
python scripts/train/eegfm_spatial_search.py generate \
  --stage stage1_screen \
  --datasets motor_mv_img,workload \
  --backbones cbramod \
  --cbramod-head avg_pool \
  --results-root results/spatial_search_head_controlled \
  --manifest results/spatial_search_head_controlled/manifest.tsv \
  --run-root results/spatial_search_head_controlled/runs \
  --config-root results/spatial_search_head_controlled/generated_configs \
  --force
```

Inspect a submission without launching jobs:

```bash
python scripts/train/eegfm_spatial_search.py submit --stage stage1_screen --dry-run --limit 3
```

Aggregate status and leaderboard:

```bash
python scripts/train/summarize_eegfm_spatial_search.py \
  --results-root results/spatial_search_head_controlled \
  --manifest results/spatial_search_head_controlled/manifest.tsv \
  --run-root results/spatial_search_head_controlled/runs \
  --status-root results/spatial_search_head_controlled/status
```

Generate stage-2 promotion rows after stage-1 metrics exist:

```bash
python scripts/train/eegfm_spatial_search.py promote --top-k 1
```
