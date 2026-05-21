# CBraMod Reconnaissance Notes

**Commit**: 20a57c4a905bc1a4eb5f9f156667f142cf5155b3
**Date inspected**: 2026-04-08

## Repository structure

```
CBraMod/
├── datasets/                   ← Dataset-specific data loaders
│   ├── bciciv2a_dataset.py
│   ├── physio_dataset.py
│   ├── tuab_dataset.py
│   ├── tuev_dataset.py
│   └── pretraining_dataset.py
├── models/
│   ├── cbramod.py              ← Main CBraMod model
│   ├── criss_cross_transformer.py  ← Core architecture
│   └── model_for_*.py          ← Task-specific model variants
├── finetune_main.py            ← Fine-tuning entry point
├── finetune_trainer.py
└── finetune_evaluator.py
```

## Model architecture

- Criss-cross transformer with explicit spatial-temporal structure
- Processes EEG with dedicated spatial and temporal attention mechanisms
- More recent than LaBraM; explicitly designed for diverse EEG formats

## Checkpoint availability

No pre-downloaded checkpoint found in repo.
Need to check official release page or HuggingFace.

## Integration assessment

CBraMod would be useful as the secondary backbone if it has public checkpoints.
Its criss-cross attention design is interesting because it separates spatial
and temporal processing, making spatial ablations more interpretable.

## Decision

**Defer to Phase 2 (Priority 3).**
Focus on REVE (primary) and LaBraM (secondary) first.
Add CBraMod if checkpoint becomes available and LaBraM proves insufficient
as a comparison point.

## If activated:

- Download checkpoint from official release
- Wrap via `CBraModWrapper`
- Map spatial attention components to our spatial embedding variants
