# BIOT Reconnaissance Notes

**Commit**: d138e32634e52ae9fa6ec98ac9c4087b14ca869a
**Date inspected**: 2026-04-08

## Repository structure

```
BIOT/
├── model/
│   ├── biot.py             ← Main BIOT model
│   ├── cnn_transformer.py
│   ├── st_transformer.py
│   └── contrawr.py
├── datasets/
│   ├── TUAB/process.py
│   ├── TUEV/process.py
│   ├── CHB-MIT/process1.py
│   └── SHHS/process.py
└── ...
```

## Assessment

BIOT is designed for biosignal learning with heterogeneous sensors.
It uses a tokenization-based approach that could handle varying channels.

## Decision

**Low priority.** BIOT adds integration complexity without a clear
unique scientific value for this paper's ablation story.
Keep as optional fallback if REVE proves insufficient.

## If activated:

- Check HuggingFace for BIOT checkpoint
- Use only if cross-modal/heterogeneous-sensor evaluation is added
