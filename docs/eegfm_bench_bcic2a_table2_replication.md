# EEG-FM-Bench BCIC-2a Table-2-Style Replication

This package targets the EEG-FM-Bench full-parameter, single-task fine-tuning protocol with an `avg_pool` MLP classification head on BCIC-2a. In the paper, the main Table 2 reports this protocol for the first seven datasets; BCIC-2a appears in Appendix Table 6 under the same "separate fine-tuning" protocol.

## Local Integration Audit

The vendored EEG-FM-Bench registry implements 10 model types: `eegpt`, `labram`, `bendr`, `biot`, `cbramod`, `reve`, `csbrain`, `eegnet`, `mantis`, and `moment`. The repo-native wrapper registry implements 4 wrappers: `cbramod`, `eegnet`, `labram`, and `reve`.

The EEG-FM-Bench paper evaluates five EEG foundation models: `BENDR`, `BIOT`, `LaBraM`, `EEGPT`, and `CBraMod`.

Local checkpoint files available now:

| Model family | Local checkpoint | Status |
| --- | --- | --- |
| BIOT | `external/BIOT/pretrained-models/EEG-PREST-16-channels.ckpt` | Available |
| BIOT | `external/BIOT/pretrained-models/EEG-SHHS+PREST-18-channels.ckpt` | Available |
| BIOT | `external/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt` | Available, selected |
| LaBraM | `external/LaBraM/checkpoints/labram-base.pth` | Available, selected |
| LaBraM tokenizer | `external/LaBraM/checkpoints/vqnsp.pth` | Available, not used by the EEG-FM-Bench trainer config |
| CBraMod | `external/CBraMod/pretrained_weights/pretrained_weights.pth` | Available, selected |
| REVE | `checkpoints/reve/positions/model.safetensors` | Position bank only; main `checkpoints/reve/model.safetensors` is missing |
| BENDR | none found | Missing |
| EEGPT | none found | Missing |

BCIC-2a data is locally available under `data/eegfm/processed/fs_200/bcic_2a/finetune`. No local processed `fs_256` BCIC-2a cache was found, so the runnable configs use `fs: 200` to match the existing cache without modifying `data/`.

## Practical Three-Model Set

The most practical immediate replication set is:

| Model | Why selected | Config |
| --- | --- | --- |
| BIOT | Paper model, registered in EEG-FM-Bench, local 18-channel checkpoint available | `configs/eegfm_bench/bcic2a_table2_biot.yaml` |
| LaBraM | Paper model, registered in EEG-FM-Bench, local base checkpoint available | `configs/eegfm_bench/bcic2a_table2_labram.yaml` |
| CBraMod | Paper model, registered in EEG-FM-Bench, local pretrained checkpoint available | `configs/eegfm_bench/bcic2a_table2_cbramod.yaml` |

REVE is implemented locally but is not one of the five models in the EEG-FM-Bench paper and its main local encoder checkpoint is missing. EEGNet is implemented locally but is not a paper foundation-model target.

## Paper Targets

BCIC-2a target values from the paper's full-parameter single-task table for the selected models:

| Model | Target B-Acc | Target as log fraction |
| --- | ---: | ---: |
| BIOT | 28.11 +/- 0.60 | 0.2811 |
| LaBraM | 29.03 +/- 0.82 | 0.2903 |
| CBraMod | 33.71 +/- 0.86 | 0.3371 |

Notes:

- Paper values are reported on a 0-100 scale; EEG-FM-Bench trainer logs `balanced_acc` on a 0-1 scale.
- The paper reports means and standard deviations over five runs. The submission helper submits one seed-42 job per selected model.
- The older local `configs/eegfm_bench/bcic2a_labram.yaml` freezes the encoder. The new Table-2-style LaBraM config uses `freeze_encoder: false` for full-parameter fine-tuning.

Source: arXiv `2508.17742v2`, Appendix B.5-B.6 for training/head settings and Appendix Table 6 for BCIC-2a full-parameter single-task values.
