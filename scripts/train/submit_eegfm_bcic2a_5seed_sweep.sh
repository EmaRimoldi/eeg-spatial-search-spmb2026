#!/usr/bin/env bash
# Submit a 5-seed EEG-FM-Bench BCIC-2a Table-2-style sweep.
#
# Default models:
#   - biot
#   - cbramod
#   - labram
#
# Default seeds:
#   - 42,43,44,45,46
#
# Usage:
#   bash scripts/train/submit_eegfm_bcic2a_5seed_sweep.sh --dry-run
#   bash scripts/train/submit_eegfm_bcic2a_5seed_sweep.sh
#   bash scripts/train/submit_eegfm_bcic2a_5seed_sweep.sh --models cbramod,labram --seeds 42,43,44,45,46

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPLATE="$PROJECT_ROOT/scripts/train/eegfm_bcic2a_job.sh"
LOG_DIR="$PROJECT_ROOT/results/eegfm_bench/slurm_logs"
GENERATED_DIR="$PROJECT_ROOT/configs/eegfm_bench/generated/bcic2a_table2_5seed"
MANIFEST="$GENERATED_DIR/manifest.tsv"
PARTITIONS="${PARTITIONS:-mit_preemptable,mit_normal_gpu,ou_bcs_low,ou_bcs_normal,pi_tpoggio}"
DRY_RUN=0
MODELS=(biot cbramod labram)
SEEDS=(42 43 44 45 46)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train/submit_eegfm_bcic2a_5seed_sweep.sh [--dry-run] [--models biot,cbramod,labram] [--seeds 42,43,44,45,46]
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
    --seeds)
      IFS=',' read -r -a SEEDS <<< "${2:?--seeds needs a comma-separated value}"
      shift 2
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

mkdir -p "$LOG_DIR" "$GENERATED_DIR"
: > "$MANIFEST"
printf 'model\tseed\tconfig\tjob_name\n' > "$MANIFEST"

base_config_for_model() {
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

generated_config_for() {
  local model="$1"
  local seed="$2"
  printf '%s/%s_seed%s.yaml' "$GENERATED_DIR" "$model" "$seed"
}

master_port_for() {
  local model="$1"
  local seed="$2"
  local base=52000
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
  /home/erimoldi/.conda/envs/sparse-hate/bin/python - "$src" "$dst" "$model" "$seed" "$port" <<'PY'
import re
import sys
src, dst, model, seed, port = sys.argv[1:6]
text = open(src, 'r').read()
text = re.sub(r'(?m)^seed:\s*\d+\s*$', f'seed: {seed}', text, count=1)
text = re.sub(r'(?m)^master_port:\s*\d+\s*$', f'master_port: {port}', text, count=1)
text = re.sub(r'(?m)^(\s*experiment_name:\s*)([^\n]+)$', rf'\1{model}_bcic2a_table2_seed{seed}', text, count=1)
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

    job_name="efm_${model}_bcic2a_t2_s${seed}"
    printf '%s\t%s\t%s\t%s\n' "$model" "$seed" "$config" "$job_name" >> "$MANIFEST"

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

printf 'submitted=%s skipped=%s manifest=%s log_dir=%s partitions=%s\n' \
  "$submitted" "$skipped" "$MANIFEST" "$LOG_DIR" "$PARTITIONS"
