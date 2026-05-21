# Explicit Electrode Coordinates for EEG Foundation Models

This repository contains the code, configuration files, and tabular results for
the SPMB 2026 manuscript:

**Do Explicit Electrode Coordinates Improve EEG Foundation Models?**

The experiments evaluate coordinate-aware adapters for BIOT, CBraMod, and
LaBraM on three processed EEG tasks: BCIC-2a, PhysioNet EEG Motor
Movement/Imagery, and EEG workload classification.

## Contents

- `src/`: model wrappers, spatial embedding modules, data loaders, and training
  code.
- `configs/`: baseline and spatial-search configuration files.
- `scripts/`: Slurm submission, training, summarization, and paper utilities.
- `results/spatial_search/`: completed original staged spatial-search manifests,
  status files, and leaderboards.
- `results/spatial_search_head_controlled/`: current head-controlled CBraMod
  rerun manifests, status files, and leaderboards.
- `results/eegfm_bench/summaries/`: baseline replication summaries.

Raw datasets, pretrained checkpoints, and large per-run artifacts are excluded.
The code expects local dataset and checkpoint paths to be configured before
rerunning training jobs.

## Main Result Files

- `results/spatial_search/run_status_paper_ready.tsv`
- `results/spatial_search/leaderboard_paper_ready.tsv`
- `results/spatial_search_head_controlled/run_status.tsv`
- `results/spatial_search_head_controlled/leaderboard.tsv`

## Reproducibility Notes

The local experiments used fixed processed splits at 200 Hz, five baseline
seeds (`42`--`46`), and full-parameter single-task fine-tuning for the
replicated baseline protocol. Spatial-search runs use a staged design: a broad
seed-42 screen followed by promotion of selected configurations to additional
seeds.
