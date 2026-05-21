#!/usr/bin/env bash
# Submit 3 targeted CBraMod diagnostic jobs on BCIC-2a.
#
# Diagnostics:
# 1) officialish optimizer/schedule + shared EEG-FM-Bench avg_pool MLP head
# 2) same, but simple linear avg_pool head (closer to official CBraMod repo)
# 3) same as (2) but from scratch (no pretrained weights)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPLATE="$PROJECT_ROOT/scripts/train/eegfm_bcic2a_job.sh"
LOG_DIR="$PROJECT_ROOT/results/eegfm_bench/slurm_logs"
PARTITIONS="${PARTITIONS:-mit_preemptable,mit_normal_gpu,ou_bcs_low,ou_bcs_normal,pi_tpoggio}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train/submit_cbramod_diagnostics.sh [--dry-run]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
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

mkdir -p "$LOG_DIR"

CONFIGS=(
  "cbramod_diag_officialish_mlp_seed42|$PROJECT_ROOT/configs/eegfm_bench/diagnostics/cbramod_diag_officialish_mlp_seed42.yaml"
  "cbramod_diag_officialish_linear_seed42|$PROJECT_ROOT/configs/eegfm_bench/diagnostics/cbramod_diag_officialish_linear_seed42.yaml"
  "cbramod_diag_officialish_linear_scratch_seed42|$PROJECT_ROOT/configs/eegfm_bench/diagnostics/cbramod_diag_officialish_linear_scratch_seed42.yaml"
)

submitted=0
for entry in "${CONFIGS[@]}"; do
  job_tag="${entry%%|*}"
  conf="${entry#*|}"
  job_name="efm_${job_tag}"
  cmd=(
    sbatch
    --partition="$PARTITIONS"
    --job-name="$job_name"
    --output="$LOG_DIR/${job_name}_%j.out"
    --error="$LOG_DIR/${job_name}_%j.err"
    --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONF_FILE=$conf,MODEL_TYPE=cbramod"
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

printf 'submitted=%s log_dir=%s partitions=%s\n' "$submitted" "$LOG_DIR" "$PARTITIONS"
