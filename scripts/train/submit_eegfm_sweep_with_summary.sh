#!/usr/bin/env bash
# Submit an EEG-FM-Bench 5-seed sweep and immediately schedule a summary job
# that runs after the sweep jobs finish.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
SUMMARY_DIR="$PROJECT_ROOT/results/eegfm_bench/summaries"

DATASET=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train/submit_eegfm_sweep_with_summary.sh --dataset bcic2a|physiomi|workload|mimul11 [extra args passed to submit_eegfm_table2_5seed_sweep.sh]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="${2:?--dataset needs a value}"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$DATASET" ]]; then
  printf 'Missing required --dataset\n' >&2
  usage >&2
  exit 1
fi

mkdir -p "$SUMMARY_DIR"

SUBMIT_OUTPUT="$(bash $PROJECT_ROOT/scripts/train/submit_eegfm_table2_5seed_sweep.sh "${EXTRA_ARGS[@]}")"
printf '%s\n' "$SUBMIT_OUTPUT"

mapfile -t JOB_IDS < <(printf '%s\n' "$SUBMIT_OUTPUT" | awk '/^Submitted batch job [0-9]+$/ {print $4}')

if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
  printf 'No sweep job IDs found in submit output for dataset=%s\n' "$DATASET" >&2
  exit 1
fi

DEP_LIST="$(IFS=:; echo "${JOB_IDS[*]}")"
CSV_LIST="$(IFS=,; echo "${JOB_IDS[*]}")"

SUMMARY_JOB_OUTPUT="$(sbatch \
  --dependency="afterany:${DEP_LIST}" \
  --job-name="efm_${DATASET}_summary" \
  --partition="mit_normal,mit_preemptable,ou_bcs_low,ou_bcs_normal,pi_tpoggio" \
  --time="00:30:00" \
  --cpus-per-task=1 \
  --mem=2G \
  --output="$PROJECT_ROOT/results/eegfm_bench/slurm_logs/efm_${DATASET}_summary_%j.out" \
  --error="$PROJECT_ROOT/results/eegfm_bench/slurm_logs/efm_${DATASET}_summary_%j.err" \
  --wrap="cd $PROJECT_ROOT && mkdir -p $SUMMARY_DIR && sacct -j $CSV_LIST --format=JobID,JobName%28,State,Elapsed,ExitCode,Partition%18,AllocCPUS -P > $SUMMARY_DIR/${DATASET}_sacct.tsv && python3 scripts/train/summarize_eegfm_table2_sweep.py --dataset $DATASET --per-seed > $SUMMARY_DIR/${DATASET}_summary.md")"

printf '%s\n' "$SUMMARY_JOB_OUTPUT"
SUMMARY_JOB_ID="$(printf '%s\n' "$SUMMARY_JOB_OUTPUT" | awk '/^Submitted batch job [0-9]+$/ {print $4}')"

printf 'SWEEP_JOB_IDS=%s\n' "$CSV_LIST"
printf 'SUMMARY_JOB_ID=%s\n' "$SUMMARY_JOB_ID"
