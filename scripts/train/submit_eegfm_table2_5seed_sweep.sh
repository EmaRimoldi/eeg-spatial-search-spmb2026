#!/usr/bin/env bash
# Submit a 5-seed EEG-FM-Bench Table-2-style sweep for a supported dataset.
#
# Supported datasets:
#   - bcic2a   -> bcic_2a
#   - physiomi -> motor_mv_img
#   - workload -> workload
#   - mimul11  -> mimul_11

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPLATE="$PROJECT_ROOT/scripts/train/eegfm_bcic2a_job.sh"
LOG_DIR="$PROJECT_ROOT/results/eegfm_bench/slurm_logs"
PARTITIONS="${PARTITIONS:-mit_preemptable,mit_normal_gpu,ou_bcs_low,ou_bcs_normal,pi_tpoggio}"
DRY_RUN=0
SKIP_DATA_CHECK=0
DATASET=""
MODELS=(biot cbramod labram)
SEEDS=(42 43 44 45 46)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train/submit_eegfm_table2_5seed_sweep.sh --dataset bcic2a|physiomi|workload|mimul11 [--dry-run] [--models biot,cbramod,labram] [--seeds 42,43,44,45,46] [--skip-data-check]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="${2:?--dataset needs a value}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --models)
      IFS=',' read -r -a MODELS <<< "${2:?--models needs a comma-separated value}"
      shift 2
      ;;
    --seeds)
      IFS=',' read -r -a SEEDS <<< "${2:?--seeds needs a comma-separated value}"
      shift 2
      ;;
    --skip-data-check)
      SKIP_DATA_CHECK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$DATASET" ]]; then
  printf '--dataset is required\n' >&2
  usage >&2
  exit 1
fi

dataset_prefix=""
dataset_key=""
processed_rel=""
port_dataset_offset=0

case "$DATASET" in
  bcic2a)
    dataset_prefix="bcic2a"
    dataset_key="bcic_2a"
    processed_rel="data/eegfm/processed/fs_200/bcic_2a/finetune/1.0.0"
    port_dataset_offset=0
    ;;
  physiomi)
    dataset_prefix="physiomi"
    dataset_key="motor_mv_img"
    processed_rel="data/eegfm/processed/fs_200/motor_mv_img/finetune/1.0.0"
    port_dataset_offset=1000
    ;;
  workload)
    dataset_prefix="workload"
    dataset_key="workload"
    processed_rel="data/eegfm/processed/fs_200/workload/finetune/1.0.0"
    port_dataset_offset=3000
    ;;
  mimul11)
    dataset_prefix="mimul11"
    dataset_key="mimul_11"
    processed_rel="data/eegfm/processed/fs_200/mimul_11/finetune/1.0.0"
    port_dataset_offset=2000
    ;;
  *)
    printf 'Unsupported dataset: %s\n' "$DATASET" >&2
    exit 1
    ;;
esac

GENERATED_DIR="$PROJECT_ROOT/configs/eegfm_bench/generated/${dataset_prefix}_table2_5seed"
MANIFEST="$GENERATED_DIR/manifest.tsv"

mkdir -p "$LOG_DIR" "$GENERATED_DIR"
printf 'dataset\tmodel\tseed\tconfig\tjob_name\tdataset_key\n' > "$MANIFEST"

if [[ ! -f "$TEMPLATE" ]]; then
  printf 'Missing SLURM job template: %s\n' "$TEMPLATE" >&2
  exit 1
fi

if (( SKIP_DATA_CHECK == 0 )) && [[ ! -d "$PROJECT_ROOT/$processed_rel" ]]; then
  printf 'Processed dataset not found yet: %s\n' "$PROJECT_ROOT/$processed_rel" >&2
  exit 1
fi

base_config_for_model() {
  printf '%s/configs/eegfm_bench/%s_%s_table2.yaml' "$PROJECT_ROOT" "$dataset_prefix" "$1"
}

weight_check_for_model() {
  local model="$1"
  case "$model" in
    biot)
      printf '%s external/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt 1000000\n' "$PROJECT_ROOT"
      ;;
    cbramod)
      printf '%s external/CBraMod/pretrained_weights/pretrained_weights.pth 1000000\n' "$PROJECT_ROOT"
      ;;
    labram)
      printf '%s external/LaBraM/checkpoints/labram-base.pth 1000000\n' "$PROJECT_ROOT"
      ;;
    *)
      return 1
      ;;
  esac
}

has_required_weights() {
  local model="$1"
  local ok=0
  while read -r root rel min_bytes; do
    local path="$root/$rel"
    if [[ ! -f "$path" ]]; then
      printf '[skip] %s: missing %s\n' "$model" "$path" >&2
      ok=1
      continue
    fi
    local size
    size="$(stat -c%s "$path")"
    if (( size < min_bytes )); then
      printf '[skip] %s: %s is too small (%s bytes, expected >= %s)\n' \
        "$model" "$path" "$size" "$min_bytes" >&2
      ok=1
    fi
  done < <(weight_check_for_model "$model")
  return "$ok"
}

generated_config_for() {
  local model="$1"
  local seed="$2"
  printf '%s/%s_seed%s.yaml' "$GENERATED_DIR" "$model" "$seed"
}

master_port_for() {
  local model="$1"
  local seed="$2"
  local base=$((52000 + port_dataset_offset))
  local offset=0
  case "$model" in
    biot) offset=100 ;;
    cbramod) offset=200 ;;
    labram) offset=300 ;;
    *) offset=900 ;;
  esac
  printf '%s' $((base + offset + seed))
}

write_generated_config() {
  local model="$1"
  local seed="$2"
  local src="$3"
  local dst="$4"
  local port="$5"
  /home/erimoldi/.conda/envs/sparse-hate/bin/python - "$src" "$dst" "$model" "$seed" "$port" "$dataset_prefix" <<'PY'
import re
import sys

src, dst, model, seed, port, dataset_prefix = sys.argv[1:7]
text = open(src, 'r').read()
text = re.sub(r'(?m)^seed:\s*\d+\s*$', 'seed: %s' % seed, text, count=1)
text = re.sub(r'(?m)^master_port:\s*\d+\s*$', 'master_port: %s' % port, text, count=1)
text = re.sub(r'(?m)^(\s*experiment_name:\s*)([^\n]+)$', r'\1%s_%s_table2_seed%s' % (model, dataset_prefix, seed), text, count=1)
open(dst, 'w').write(text)
PY
}

submitted=0
skipped=0

for model in "${MODELS[@]}"; do
  base_config="$(base_config_for_model "$model")"
  if [[ ! -f "$base_config" ]]; then
    printf '[skip] %s: missing base config %s\n' "$model" "$base_config" >&2
    skipped=$((skipped + 1))
    continue
  fi

  if ! has_required_weights "$model"; then
    skipped=$((skipped + 1))
    continue
  fi

  for seed in "${SEEDS[@]}"; do
    config="$(generated_config_for "$model" "$seed")"
    port="$(master_port_for "$model" "$seed")"
    write_generated_config "$model" "$seed" "$base_config" "$config" "$port"

    job_name="efm_${model}_${dataset_prefix}_t2_s${seed}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$dataset_prefix" "$model" "$seed" "$config" "$job_name" "$dataset_key" >> "$MANIFEST"

    cmd=(
      sbatch
      --partition="$PARTITIONS"
      --job-name="$job_name"
      --output="$LOG_DIR/${job_name}_%j.out"
      --error="$LOG_DIR/${job_name}_%j.err"
      --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONF_FILE=$config,MODEL_TYPE=$model"
      "$TEMPLATE"
    )

    if (( DRY_RUN == 1 )); then
      printf '[dry-run] %s\n' "${cmd[*]}"
    else
      "${cmd[@]}"
      submitted=$((submitted + 1))
      sleep 0.2
    fi
  done
done

printf 'submitted=%s skipped=%s dataset=%s manifest=%s log_dir=%s partitions=%s\n' \
  "$submitted" "$skipped" "$dataset_prefix" "$MANIFEST" "$LOG_DIR" "$PARTITIONS"
