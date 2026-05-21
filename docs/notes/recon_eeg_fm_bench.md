# EEG-FM-Bench Reconnaissance Notes

**Commit**: 9b34c0885a4dbd126b37ca1ac212a4a6166aa177
**Date inspected**: 2026-04-08

## Repository structure

```
EEG-FM-Bench/
├── baseline/               ← Baseline model implementations
├── common/
│   ├── conf.py             ← Configuration management
│   ├── log.py              ← Logging utilities
│   ├── path.py             ← Path management
│   ├── type.py             ← Type definitions
│   └── utils.py            ← Shared utilities
├── data/                   ← Dataset handling
├── analysis_run.py         ← Analysis entrypoint
├── analysis_vis.py         ← Visualization entrypoint
├── baseline_main.py        ← Baseline model training
├── plot_vis.py             ← Plot generation
├── preproc.py              ← Preprocessing entry point
└── scripts/                ← Shell scripts for analysis
```

## Key observations

- EEG-FM-Bench provides a unified evaluation harness
- Supports REVE, LaBraM, CBraMod, EEGPT, and BIOT
- Has consistent config system and analysis tooling

## Integration assessment

EEG-FM-Bench is primarily an analysis/visualization framework.
The actual model training is done via the individual model repos.
It serves as a benchmark comparison layer, not the training engine.

## Integration plan for our paper

Use EEG-FM-Bench primarily for:
- Dataset preprocessing pipeline reference
- Evaluation comparison with standard baselines
- Analysis and visualization scripts

Our custom training will be driven by our own scripts wrapping REVE's dt.py.

## Friction points

- EEG-FM-Bench appears to be a newer/in-progress benchmark
- May not have deep integration hooks for custom spatial modules
- Likely best used as a reference, not as the primary driver

## Decision

Use as reference implementation and evaluation comparison baseline.
Primary training driver: our own wrapper around REVE's downstream task pipeline.
