# EEG-FM-Bench BCIC-2a triplet replication plan

## Goal

Build a **reliable local baseline** for three models from the EEG-FM-Bench paper on **BCIC-2a**, targeting the paper's **Table 2** protocol:

- **single-task** fine-tuning
- **full-parameter** fine-tuning (`freeze_encoder: false`)
- **`avg_pool`** classification head
- metric of record: **balanced accuracy** on **BCIC-2a**

This is the most practical paper-faithful target with the assets currently available in this workspace.

## Model inventory

The local `EEG-FM-Bench` registry currently implements **10** model types:

1. `bendr`
2. `biot`
3. `cbramod`
4. `csbrain`
5. `eegnet`
6. `eegpt`
7. `labram`
8. `mantis`
9. `moment`
10. `reve`

Of these, the **9 models relevant to the EEG-FM-Bench paper** are:

- `bendr`
- `biot`
- `cbramod`
- `csbrain`
- `eegpt`
- `labram`
- `mantis`
- `moment`
- `reve`

`eegnet` exists locally but is **not** one of the paper's EEG-FM / time-series foundation model baselines.

## Checkpoints available locally now

Immediately usable local pretrained weights/checkpoints:

- **BIOT**
  - `external/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt`
  - `external/BIOT/pretrained-models/EEG-SHHS+PREST-18-channels.ckpt`
  - `external/BIOT/pretrained-models/EEG-PREST-16-channels.ckpt`
- **CBraMod**
  - `external/CBraMod/pretrained_weights/pretrained_weights.pth`
- **LaBraM**
  - `external/LaBraM/checkpoints/labram-base.pth`

Partially present / not ready enough for immediate faithful reproduction:

- **REVE**: position-bank file exists, but the main local safetensors checkpoint expected by the current submit helper is missing.
- **EEGPT / CSBrain / BENDR / Mantis / MOMENT**: code is present in the benchmark tree, but this workspace does not currently expose a ready local checkpoint path analogous to the three models above.

## Chosen 3-model replication set

For a baseline we can reproduce **right now**, the most practical triplet is:

1. **BIOT**
2. **CBraMod**
3. **LaBraM**

Why these three:

- they are explicitly in the EEG-FM-Bench paper,
- they have local checkpoints available now,
- they cover both EEG-specific FMs and a strong time-series baseline,
- they avoid the incomplete REVE checkpoint situation.

## Paper targets (EEG-FM-Bench Table 2, BCIC-2a, B-Acc)

These are the target **balanced accuracies** for the selected triplet under the paper's Table 2 setup:

| Model | Target B-Acc |
|---|---:|
| BIOT | **28.11 ± 0.60** |
| LaBraM | **29.03 ± 0.82** |
| CBraMod | **33.71 ± 0.86** |

For reference, the full Table 2 BCIC-2a B-Acc row is:

- BENDR: 35.21 ± 0.54
- BIOT: 28.11 ± 0.60
- LaBraM: 29.03 ± 0.82
- EEGPT: 44.07 ± 3.27
- CBraMod: 33.71 ± 0.86
- CSBrain: 36.23 ± 0.25
- REVE: 32.73 ± 0.19

## Important local divergences discovered

### 1. Existing local LaBraM BCIC-2a config is not a Table-2-style run
The pre-existing local file `configs/eegfm_bench/bcic2a_labram.yaml` has:

- `freeze_encoder: true`
- `multitask: true`

That makes it **not** a faithful Table 2 reproduction, because Table 2 is **full-parameter single-task**.

### 2. Existing local CBraMod BCIC-2a config is also not strictly Table-2-style
The pre-existing local file `configs/eegfm_bench/bcic2a_cbramod.yaml` has:

- `multitask: true`

With only one dataset this may still run, but it does **not** cleanly encode the paper's single-task protocol.

### 3. EEG-FM-Bench trainer semantics are easy to misread
Inside `baseline/abstract/trainer.py`:

- `multitask: true` → `run_unified_training()`
- `multitask: false` → `run_separate_training()`

For a clean BCIC-2a Table 2 replication, we should set **`multitask: false`**.

## Replication policy used here

The new triplet configs in `configs/eegfm_bench/` intentionally:

- do **not** mutate older exploratory configs,
- use **new experiment names**,
- set `multitask: false`,
- set `freeze_encoder: false`,
- keep `avg_pool` heads,
- point only to locally available checkpoints.

## 5-seed evaluation policy

The paper reports mean ± spread values, so the reliable comparison here is **not** a single lucky seed.
The default replication sweep therefore uses **5 seeds**:

- `42`
- `43`
- `44`
- `45`
- `46`

For each model we report:

- per-seed `test_at_best_eval_bacc`
- per-seed `best_test_bacc`
- **mean ± sample std** across the 5 seeds

The main number to compare against the paper is:

- **mean `test_at_best_eval_bacc` ± std**

## BIOT runtime fix

The first BIOT run did not fail because of model quality; it failed because the Slurm job environment was missing the NVRTC builtins library needed by the STFT backward path:

- missing runtime library: `libnvrtc-builtins.so.13.0`

The job launcher now exports both CUDA/NVRTC library directories from the local conda env in `scripts/train/eegfm_bcic2a_job.sh`, which is required for BIOT's GPU STFT path.

## Recommended execution order

1. Submit the **5-seed** Table-2-style sweep with:
   - `bash scripts/train/submit_eegfm_bcic2a_5seed_sweep.sh`
2. After runs finish, summarize per-seed and aggregated results with:
   - `python scripts/paper/summarize_eegfm_bcic2a_5seed_sweep.py`
3. Compare the resulting **mean ± std** per model to the paper targets above.

## Success criterion

We should treat the triplet as a **usable baseline** if:

- all **15 sweep jobs** (3 models × 5 seeds) finish cleanly,
- checkpoints and logs are produced normally,
- each model's **mean `test_at_best_eval_bacc`** lands roughly in the neighborhood of its Table 2 target,
- the seed-to-seed spread is not wildly unstable,
- no run is obviously broken because of missing weights, accidental freezing, wrong multitask mode, or runtime library issues.

If one of the models misses badly, the next debugging targets should be:

1. checkpoint mismatch,
2. wrong training mode (`multitask` vs single-task),
3. dataset/count drift,
4. optimizer / schedule mismatch,
5. architecture-specific loading warnings.
