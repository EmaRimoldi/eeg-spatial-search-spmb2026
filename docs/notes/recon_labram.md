# LaBraM Reconnaissance Notes

**Commit**: c431221e6cfd23dbfa9950e0180682fb322b0548
**Date inspected**: 2026-04-08

## Repository structure

```
LaBraM/
├── checkpoints/
│   ├── labram-base.pth     ← Pretrained weights (INCLUDED in repo!)
│   └── vqnsp.pth           ← VQ-NSP tokenizer weights
├── data_processor/
│   ├── data_preprocess.py
│   └── dataset.py
├── dataset_maker/
│   ├── make_TUAB.py
│   └── make_TUEV.py
├── modeling_finetune.py    ← Fine-tuning model definition
├── modeling_pretrain.py    ← Pretraining model definition
├── modeling_vqnsp.py       ← VQ-NSP tokenizer
├── run_class_finetuning.py ← Classification fine-tuning entry point
├── engine_for_finetuning.py
└── utils.py
```

## Model architecture

- ViT-style transformer adapted for EEG
- Input: fixed-channel EEG with positional embeddings
- **Key limitation**: Fixed channel order assumed during pretraining
- Positional encoding: standard learned PE per channel position (not spatial coordinates)
- Input shape: (B, C, T) where C channels are in fixed order

## Checkpoint availability

**Critical**: `labram-base.pth` is included in the GitHub repo.
This is extremely convenient — no separate download needed.
Size: needs to be verified (LFS likely used for large checkpoint).

## Loading path

```python
from modeling_finetune import BrainNetViTForFineTuning
import torch

model = BrainNetViTForFineTuning(...)
checkpoint = torch.load('checkpoints/labram-base.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
```

## Fine-tuning entry point

```bash
python run_class_finetuning.py \
    --finetune checkpoints/labram-base.pth \
    --data_path /path/to/data \
    --nb_classes 4 \
    ...
```

## Input format

- Fixed channel layout (channels must be in predefined order)
- Standard 10-20 channel names expected
- Temporal patches: (B, C, T) split into patches

## Integration plan for our paper

LaBraM serves as the **fixed-channel-layout reference point**.
This makes it ideal for:
- Demonstrating that fixed-channel models struggle with layout shift
- Comparing against REVE's flexible positional encoding

Wrapper strategy:
1. Load `labram-base.pth` from `external/LaBraM/checkpoints/`
2. Create `LaBraMWrapper` that accepts our metadata dict
3. Map our canonical channel names to LaBraM's fixed channel order
4. Zero-pad or mask channels not present in LaBraM's vocabulary

## Channel assumptions

- LaBraM assumes a fixed set of EEG channels in a specific order
- Channel names from standard 10-20 system
- Cannot handle truly novel montages without retraining

## Friction points

- Fixed channel order: need remapping from our canonical names
- Missing channels: need masking strategy
- Positional encoding is per-index, not per-coordinate

## Scientific value for the paper

LaBraM represents the "channel-ID / fixed-layout" baseline.
It is the ideal comparison to show whether REVE's flexible positional
encoding (and our ablations) actually help under layout shift.
