#!/usr/bin/env bash
# Submit the three immediate EEG-FM-Bench BCIC-2a Table-2-style replication jobs.
#
# Usage:
#   bash scripts/train/submit_eegfm_bcic2a_table2_replication.sh --dry-run
#   bash scripts/train/submit_eegfm_bcic2a_table2_replication.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPLATE="$PROJECT_ROOT/scripts/train/eegfm_bcic2a_job.sh"
LOG_DIR="$PROJECT_ROOT/results/eegfm_bench/slurm_logs"
PARTITIONS="${PARTITIONS:-mit_normal_gpu,mit_preemptable,ou_bcs_low,ou_bcs_normal}"
DRY_RUN=0

MODELS=(biot labram cbramod)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      exit 1
      ;;
  esac
done

config_for_model() {
  case "$1" in
    biot)
      printf '%s/configs/eegfm_bench/bcic2a_table2_biot.yaml\n' "$PROJECT_ROOT"
      ;;
    labram)
      printf '%s/configs/eegfm_bench/bcic2a_table2_labram.yaml\n' "$PROJECT_ROOT"
      ;;
    cbramod)
      printf '%s/configs/eegfm_bench/bcic2a_table2_cbramod.yaml\n' "$PROJECT_ROOT"
      ;;
    *)
      return 1
      ;;
  esac
}

weight_check_for_model() {
  case "$1" in
    biot)
      printf '%s external/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt 1000000\n' "$PROJECT_ROOT"
      ;;
    labram)
      printf '%s external/LaBraM/checkpoints/labram-base.pth 1000000\n' "$PROJECT_ROOT"
      ;;
    cbramod)
      printf '%s external/CBraMod/pretrained_weights/pretrained_weights.pth 1000000\n' "$PROJECT_ROOT"
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
      printf '[missing] %s: %s\n' "$model" "$path" >&2
      ok=1
      continue
    fi

    local size
    size="$(stat -c%s "$path")"
    if (( size < min_bytes )); then
      printf '[too-small] %s: %s is %s bytes, expected >= %s\n' \
        "$model" "$path" "$size" "$min_bytes" >&2
      ok=1
    fi
  done < <(weight_check_for_model "$model")
  return "$ok"
}

if [[ ! -f "$TEMPLATE" ]]; then
  printf 'Missing SLURM job template: %s\n' "$TEMPLATE" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

submitted=0
for model in "${MODELS[@]}"; do
  config="$(config_for_model "$model")"
  if [[ ! -f "$config" ]]; then
    printf '[missing] %s config: %s\n' "$model" "$config" >&2
    exit 1
  fi

  if ! has_required_weights "$model"; then
    printf 'Aborting before submission because %s weights are not ready.\n' "$model" >&2
    exit 1
  fi

  job_name="efm_tbl2_bcic2a_${model}"
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

printf 'submitted=%s models=%s log_dir=%s\n' "$submitted" "${MODELS[*]}" "$LOG_DIR"
