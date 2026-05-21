#!/usr/bin/env bash
# Submit a paper-faithful EEG-FM-Bench BCIC-2a replication triplet.
#
# Targets EEG-FM-Bench Table 2 protocol for three locally available models:
#   - biot
#   - cbramod
#   - labram
#
# Usage:
#   bash scripts/train/submit_eegfm_bcic2a_triplet.sh --dry-run
#   bash scripts/train/submit_eegfm_bcic2a_triplet.sh
#   PARTITIONS=mit_preemptable bash scripts/train/submit_eegfm_bcic2a_triplet.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPLATE="$PROJECT_ROOT/scripts/train/eegfm_bcic2a_job.sh"
LOG_DIR="$PROJECT_ROOT/results/eegfm_bench/slurm_logs"
PARTITIONS="${PARTITIONS:-pi_tpoggio,mit_preemptable,mit_normal_gpu}"
DRY_RUN=0
MODELS=(biot cbramod labram)

action_usage() {
  cat <<'EOF'
Usage:
  bash scripts/train/submit_eegfm_bcic2a_triplet.sh [--dry-run] [--models biot,cbramod,labram]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --models)
      IFS=',' read -r -a MODELS <<< "${2:?--models needs a comma-separated value}"
      shift 2
      ;;
    -h|--help)
      action_usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      action_usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "$LOG_DIR"

config_for_model() {
  printf '%s/configs/eegfm_bench/bcic2a_%s_table2.yaml' "$PROJECT_ROOT" "$1"
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

submitted=0
skipped=0

for model in "${MODELS[@]}"; do
  config="$(config_for_model "$model")"
  if [[ ! -f "$config" ]]; then
    printf '[skip] %s: missing config %s\n' "$model" "$config" >&2
    skipped=$((skipped + 1))
    continue
  fi

  if ! has_required_weights "$model"; then
    skipped=$((skipped + 1))
    continue
  fi

  job_name="efm_${model}_bcic2a_t2"
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

printf 'submitted=%s skipped=%s log_dir=%s partitions=%s\n' \
  "$submitted" "$skipped" "$LOG_DIR" "$PARTITIONS"
