# Project Scope

**Locked**: 2026-04-08

## Hard constraints (from manual)

- No large-scale self-supervised pretraining from scratch.
- No attempt to reproduce 60k-hour or 20k-hour corpus training.
- No more than **2 pretrained FM backbones** in first pass.
- No more than **3 datasets** in first complete run.
- Paper built around **controlled ablations**, not model novelty-by-scale.

## Locked choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Primary backbone | REVE | Best spatial-encoding FM, HF checkpoint available |
| Secondary backbone | LaBraM | Strong baseline, included checkpoint |
| Non-FM baseline | EEGNet | Lightweight, no pretraining needed |
| Primary dataset | MOABB BNCI2014_001 | Reproducible, easy access, motor imagery |
| Secondary dataset | MOABB PhysionetMI | Different channel layout (64 ch) |
| Tertiary dataset | TUAB (if accessible) | Clinical EEG, layout shift |
| Benchmark harness | EEG-FM-Bench | Reference only |
| Config system | YAML + argparse | Simple and portable |

## What is out of scope

- EEGPT: no clear public checkpoint advantage for our ablation
- CBraMod: deferred until LaBraM proves insufficient
- BIOT: no clear advantage for this paper's question
- New self-supervised pretraining objectives
- New backbone architectures

## Spatial variants (locked)

| Variant | Status |
|---------|--------|
| `none` | Implemented |
| `channel_id` | Implemented |
| `coords2d` | Implemented |
| `coords3d` | Implemented |
| `coords3d_distbias` | Implemented |
| `coords3d_reference` | Implemented |
| `topology_agnostic` | Implemented |

All variants pass forward-pass shape tests.
