#!/usr/bin/env bash
# Submit a targeted CBraMod sweep on BCIC-2a.
#
# Variants:
#   1) officialish_mlp
#   2) officialish_linear
#
# Seeds: 42,43,44,45,46
# Scheduling: 2 jobs per seed batch, next seed waits for both previous jobs.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPLATE="$PROJECT_ROOT/scripts/train/eegfm_bcic2a_job.sh"
LOG_DIR="$PROJECT_ROOT/results/eegfm_bench/slurm_logs"
GENERATED_DIR="$PROJECT_ROOT/configs/eegfm_bench/generated/cbramod_targeted_sweep"
MANIFEST="$GENERATED_DIR/manifest.tsv"
PARTITIONS="${PARTITIONS:-mit_preemptable,mit_normal_gpu,ou_bcs_low,ou_bcs_normal,pi_tpoggio}"
DRY_RUN=0
SEEDS=(42 43 44 45 46)
VARIANTS=(officialish_mlp officialish_linear)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train/submit_cbramod_targeted_sweep.sh [--dry-run] [--seeds 42,43,44,45,46]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
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
printf 'seed\tvariant\tjob_id\tjob_name\tconfig\tdependency\n' > "$MANIFEST"

base_config_for_variant() {
  local variant="$1"
  printf '%s/configs/eegfm_bench/cbramod_%s_seed42.yaml' "$PROJECT_ROOT" "$variant"
}

generated_config_for() {
  local variant="$1"
  local seed="$2"
  printf '%s/cbramod_%s_seed%s.yaml' "$GENERATED_DIR" "$variant" "$seed"
}

master_port_for() {
  local variant="$1"
  local seed="$2"
  local base=56000
  local offset=0
  case "$variant" in
    officialish_mlp) offset=100 ;;
    officialish_linear) offset=200 ;;
    *) offset=900 ;;
  esac
  printf '%s' $((base + offset + seed))
}

write_generated_config() {
  local variant="$1"
  local seed="$2"
  local src="$3"
  local dst="$4"
  local port="$5"
  /home/erimoldi/.conda/envs/sparse-hate/bin/python - "$src" "$dst" "$variant" "$seed" "$port" <<'PY'
import re
import sys
src, dst, variant, seed, port = sys.argv[1:6]
text = open(src, 'r').read()
text = re.sub(r'(?m)^seed:\s*\d+\s*$', f'seed: {seed}', text, count=1)
text = re.sub(r'(?m)^master_port:\s*\d+\s*$', f'master_port: {port}', text, count=1)
text = re.sub(r'(?m)^(\s*experiment_name:\s*)([^\n]+)$', rf'\1cbramod_{variant}_seed{seed}', text, count=1)
open(dst, 'w').write(text)
PY
}

submitted=0
prev_dependency=""
for seed in "${SEEDS[@]}"; do
  current_job_ids=()
  dependency_flag=()
  if [[ -n "$prev_dependency" ]]; then
    dependency_flag=("--dependency=afterany:${prev_dependency}")
  fi

  for variant in "${VARIANTS[@]}"; do
    src="$(base_config_for_variant "$variant")"
    conf="$(generated_config_for "$variant" "$seed")"
    port="$(master_port_for "$variant" "$seed")"
    write_generated_config "$variant" "$seed" "$src" "$conf" "$port"
    job_name="efm_cbramod_${variant}_s${seed}"
    cmd=(
      sbatch
      --partition="$PARTITIONS"
      "${dependency_flag[@]}"
      --job-name="$job_name"
      --output="$LOG_DIR/${job_name}_%j.out"
      --error="$LOG_DIR/${job_name}_%j.err"
      --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONF_FILE=$conf,MODEL_TYPE=cbramod"
      "$TEMPLATE"
    )
    if (( DRY_RUN == 1 )); then
      printf '[dry-run] %s\n' "${cmd[*]}"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$seed" "$variant" "DRY_RUN" "$job_name" "$conf" "${prev_dependency:-none}" >> "$MANIFEST"
    else
      out="$(${cmd[@]})"
      echo "$out"
      job_id="$(printf '%s\n' "$out" | awk '/Submitted batch job/{print $4}')"
      current_job_ids+=("$job_id")
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$seed" "$variant" "$job_id" "$job_name" "$conf" "${prev_dependency:-none}" >> "$MANIFEST"
      submitted=$((submitted + 1))
      sleep 0.2
    fi
  done

  if (( DRY_RUN == 0 )); then
    prev_dependency="$(IFS=:; echo "${current_job_ids[*]}")"
  else
    prev_dependency="DRY_RUN_DEPENDENCY"
  fi
done

printf 'submitted=%s manifest=%s log_dir=%s partitions=%s\n' "$submitted" "$MANIFEST" "$LOG_DIR" "$PARTITIONS"
