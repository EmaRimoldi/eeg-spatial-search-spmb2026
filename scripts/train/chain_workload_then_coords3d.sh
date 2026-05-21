#!/usr/bin/env bash
# After Workload preprocessing succeeds, submit the Workload sweep+summary and
# then arm the Stage-1 spatial pilot once both PhysioMI and Workload summaries
# have completed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

PHYSIO_SUMMARY_JOB_ID="${PHYSIO_SUMMARY_JOB_ID:-}"

if [[ -z "$PHYSIO_SUMMARY_JOB_ID" ]]; then
  printf 'PHYSIO_SUMMARY_JOB_ID is required\n' >&2
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/results/pilot/slurm_logs"

CHAIN_OUTPUT="$(bash $PROJECT_ROOT/scripts/train/submit_eegfm_sweep_with_summary.sh --dataset workload)"
printf '%s\n' "$CHAIN_OUTPUT"

WORKLOAD_SUMMARY_JOB_ID="$(printf '%s\n' "$CHAIN_OUTPUT" | awk -F= '/^SUMMARY_JOB_ID=/ {print $2}')"
if [[ -z "$WORKLOAD_SUMMARY_JOB_ID" ]]; then
  printf 'Could not determine workload summary job id\n' >&2
  exit 1
fi

PILOT_JOB_OUTPUT="$(sbatch \
  --dependency="afterok:${PHYSIO_SUMMARY_JOB_ID}:${WORKLOAD_SUMMARY_JOB_ID}" \
  --job-name="coords3d_pilot_submit" \
  --partition="mit_normal,mit_preemptable,ou_bcs_low,ou_bcs_normal,pi_tpoggio" \
  --time="00:30:00" \
  --cpus-per-task=1 \
  --mem=1G \
  --output="$PROJECT_ROOT/results/pilot/slurm_logs/coords3d_pilot_submit_%j.out" \
  --error="$PROJECT_ROOT/results/pilot/slurm_logs/coords3d_pilot_submit_%j.err" \
  --wrap="cd $PROJECT_ROOT && mkdir -p results/pilot/slurm_logs && bash scripts/train/slurm_pilot.sh")"

printf '%s\n' "$PILOT_JOB_OUTPUT"
PILOT_SUBMIT_JOB_ID="$(printf '%s\n' "$PILOT_JOB_OUTPUT" | awk '/^Submitted batch job [0-9]+$/ {print $4}')"
printf 'WORKLOAD_SUMMARY_JOB_ID=%s\n' "$WORKLOAD_SUMMARY_JOB_ID"
printf 'PILOT_SUBMIT_JOB_ID=%s\n' "$PILOT_SUBMIT_JOB_ID"
