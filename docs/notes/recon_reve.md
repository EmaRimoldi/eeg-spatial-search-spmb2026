# REVE Reconnaissance Notes

**Commit**: 06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7
**Date inspected**: 2026-04-08

## Repository structure

```
reve_eeg/
├── hf/
│   ├── reve-base/          ← HuggingFace-compatible model files
│   │   ├── modeling_reve.py     ← Main model implementation
│   │   ├── configuration_reve.py
│   │   └── config.json
│   └── reve-positions/     ← Position bank (channel → 3D coordinate lookup)
│       ├── position_bank.py
│       ├── positions.json
│       └── configuration_bank.py
├── src/
│   ├── models/
│   │   ├── backbone.py     ← Training backbone with MAE pretraining
│   │   ├── classifier.py   ← Downstream classification head
│   │   └── encoder.py      ← Shared encoder logic
│   ├── downstream_tasks/
│   │   ├── dataloader_moabb.py   ← MOABB downstream data loading
│   │   ├── dataloader_tuh.py     ← TUH downstream data loading
│   │   ├── eval_core.py          ← Evaluation loop
│   │   ├── train_core.py         ← Training loop
│   │   └── position_utils.py     ← Position handling utilities
│   ├── dt.py               ← Downstream task entry point
│   └── eval_dt.py          ← Evaluation entry point
└── preprocessing/          ← Dataset-specific preprocessing scripts
    ├── preprocessing_bciciv2a.py
    ├── preprocessing_physio.py
    └── ...
```

## Model architecture overview

- **Reve** class: a transformer-based EEG encoder
- Input: EEG patches, positions (3D xyz per channel)
- Positional encoding: `FourierEmb4D` (4D = x, y, z, t using Fourier features)
  - This is the key spatial pathway
  - Positions are (B, C, 3) float tensors of xyz coordinates
  - Alternative: `Learnable4DPE` (discrete lookup into a position bank)
- Config: depth=22, embed_dim=512, heads=8, head_dim=64, mlp_dim_ratio=2.66
- Patching: patch_size=200, patch_overlap=20

## Positional encoding details (critical for paper)

REVE's spatial pathway is in `FourierEmb4D`:
```python
class FourierEmb4D(nn.Module):
    def __init__(self, dimension, freqs, increment_time, margin):
        ...
    def forward(self, positions_):
        # positions_: (B, C*T, 4) where 4 = (x, y, z, t)
        # Returns Fourier embedding of shape (B, C*T, dimension)
```

The position bank (`RevePositionBank`) maps channel names → learned 3D position embeddings.
This is the component to modify or ablate for our paper.

## HuggingFace loading path

```python
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "brain-bzh/reve-base",
    trust_remote_code=True
)
positions = AutoModel.from_pretrained(
    "brain-bzh/reve-positions",
    trust_remote_code=True
)
```

## Input format

- EEG tensor: (B, C, T) or patched format
- Position tensor: (B, C, 3) — xyz coordinates for each channel
- The model internally patches the signal and expands positions

## Downstream fine-tuning

- `src/dt.py` is the downstream task entry point
- Configs in `src/configs/task/` (one per dataset)
- Uses Hydra for config management
- Supports MOABB, TUH, ISRUC datasets natively

## Integration plan for our paper

1. Load REVE via HuggingFace API (`from_pretrained`)
2. Intercept the spatial-input pathway (positions tensor)
3. Provide our custom positions from the metadata pipeline
4. Swap `FourierEmb4D` with our spatial embedding variants
5. Wrap in `REVEWrapper` with freeze/unfreeze control

## Friction points

- FourierEmb4D expects positions as (B, C*T, 4) where T is temporal patches
  → need to handle temporal dimension expansion
- Position bank uses discrete lookup; ablating to continuous coords requires
  bypassing the lookup and feeding raw xyz
- Flash attention requires optional install; wrapper should fall back gracefully

## Channel assumptions

REVE supports arbitrary channel arrangements via continuous coordinates.
The position bank covers standard 10-20/10-10 channels.
New/custom channel layouts need coordinate registration in the positions.json
or via our own coordinate lookup.
