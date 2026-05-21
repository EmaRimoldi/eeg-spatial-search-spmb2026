# Pilot Run Summary

## Status: PILOT JOBS SUBMITTED — Awaiting GPU results (QOSGrpGRES queue)

---

## Current state (2026-04-08, updated)

### SLURM Pilot Jobs Submitted (Job IDs 11539960–11539969)

- **9 jobs** queued: 3 variants (`channel_id`, `coords2d`, `coords3d`) × 3 seeds (42, 123, 456)
- Partition: `pi_tpoggio`, 1 GPU, 2h wall time each
- Status: `PENDING (QOSGrpGRES)` — waiting for GPU quota

Once jobs complete:
```bash
python src/analysis/aggregate_results.py --results-dir results/pilot
```

### End-to-end validated on real BNCI2014_001 data (CPU, 3 epochs)

- `BNCI2014_001_reve_coords3d_head_only_seed42_same_layout_2026-04-08-1149`
- val_bacc=0.2691 (epoch 1), test_bacc=0.2865 — chance level expected (no real REVE weights)
- On GPU with real checkpoint, meaningful results expected (>50% on motor imagery)

### Next experiments (ready to submit once pilot results in)

| Script | Experiment | N jobs |
|---|---|---|
| `scripts/train/slurm_core_ablation.sh` | 7 variants × 3 regimes × 3 seeds | 63 |
| `scripts/train/slurm_few_shot.sh` | 5 variants × 5 fractions × 3 seeds | 75 |
| `scripts/train/slurm_channel_dropout.sh` | 4 variants × 4 dropout rates × 3 seeds | 48 |
| `scripts/eval/slurm_cross_layout.sh` | Cross-montage eval (PhysionetMI) | 1 per trained model |

---

## Current state (2026-04-08)

### Environment

- Python interpreter: `/home/erimoldi/.conda/envs/sparse-hate/bin/python` (Python 3.11)
- No dedicated `eeg-spatial-paper` conda env yet — using `sparse-hate` (fully loaded)
- All required packages installed:
  - torch 2.11.0+cu130, numpy 2.4.4, transformers 5.5.0, timm 1.0.26
  - mne 1.12.0, braindecode 1.4.0, moabb 1.5.0
  - scikit-learn 1.8.0, hydra-core 1.3.2, einops 0.8.2, safetensors

### Test suite

**63/63 tests pass** (`python -m pytest tests/ -v`)

### End-to-end training (synthetic data)

```
python src/training/train.py \
    --backbone reve \
    --spatial-variant coords3d \
    --freeze-policy head_only \
    --dataset synthetic \
    --num-classes 4 \
    --seed 42 \
    --epochs 3 \
    --batch-size 8 \
    --output-dir results/logs/
```

Result: runs successfully, val_acc ~25% (chance-level expected with random REVE weights).

---

## Blockers for real pilot runs

### 1. REVE checkpoint — GATED HuggingFace repo

`brain-bzh/reve-base` is a restricted repository requiring:

1. Visit https://huggingface.co/brain-bzh/reve-base
2. Request access (approve terms of use)
3. Log in: `huggingface-cli login` (paste your HF token)
4. Download checkpoint:
   ```bash
   /home/erimoldi/.conda/envs/sparse-hate/bin/python -c "
   from huggingface_hub import hf_hub_download
   hf_hub_download(
       repo_id='brain-bzh/reve-base',
       filename='model.safetensors',
       local_dir='checkpoints/reve',
   )
   "
   ```

The wrapper will automatically find the checkpoint at `checkpoints/reve/model.safetensors`.

### 2. Real EEG data (MOABB/BNCI2014_001)

MOABB downloads data on first use. Either:
- Run with `--dataset synthetic` for development
- Run `python -c "import moabb; from moabb.datasets import BNCI2014_001; d = BNCI2014_001(); d.download()"` to pre-download

---

## To run the pilot

Once REVE checkpoint is available:

```bash
# 1. Request HF access + login
huggingface-cli login

# 2. Download REVE checkpoint (one time)
bash scripts/download/download_reve.sh

# 3. Run pilot (synthetic data, fast)
bash scripts/train/run_pilot.sh

# OR single run:
python src/training/train.py \
    --backbone reve \
    --spatial-variant coords3d \
    --freeze-policy head_only \
    --dataset synthetic \
    --num-classes 4 \
    --seed 42 \
    --epochs 20 \
    --output-dir results/pilot/
```

---

## Architecture verified

- REVE backbone: **real** (69M params loaded from `hf/reve-base/`) — architecture correct
- REVE backbone frozen (head_only): 3,076 trainable params (head only)
- Spatial variant → pos tensor mapping: all 7 variants tested and correct
- All spatial embedding modules: instantiate and produce correct shapes
- Registry: `build_model('reve', ...)` works
- Training loop: complete with synthetic data

---

## Key design decisions made

1. **Spatial variant → pos tensor mapping for REVE**
   - `none`, `topology_agnostic` → `pos = zeros`
   - `channel_id` → pos from canonical 3D coord lookup by channel name
   - `coords2d` → `pos = (x, y, 0)` (zero-padded z)
   - `coords3d`, `coords3d_distbias`, `reve_default` → `pos = (x, y, z)`
   - `coords3d_reference` → `pos = (x, y, z)` + reference embedding added to features

2. **REVE reference encoding** (`coords3d_reference`):
   Reference type is encoded as a learned embedding (`nn.Embedding(4, embed_dim)`)
   and added to the pooled features before the classifier head.
   This cannot be injected directly into REVE's `FourierEmb4D` without invasive modification.

3. **REVE loading strategy** (in order):
   - Local `checkpoints/reve/model.safetensors` (from HF download)
   - HuggingFace download (requires authentication + gated access)
   - Architecture-only random weights (development/testing)
   - Stub (if HF path doesn't exist)

4. **Environment**: Using `sparse-hate` conda env (Python 3.11) as the project Python.
   All dependencies manually installed via pip. No separate `eeg-spatial-paper` conda env
   created yet (no conda binary available; can be recreated via `env/environment.yml` later).
