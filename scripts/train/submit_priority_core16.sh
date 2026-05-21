#!/usr/bin/env bash
# Submit the reduced 16-job core plan for the EEG spatial paper.
# Priorities:
#   P1: finish the main geometry chain under full fine-tuning
#       - coords2d full seed456 (complete existing nearly-finished cell)
#       - coords3d full seeds 42/123/456
#       - coords3d_reference full seeds 42/123/456
#   P2: test new variants in the frozen-backbone regime
#       - coords3d_reference head_only seeds 42/123/456
#       - topology_agnostic head_only seeds 42/123/456
#   P3: complete topology_agnostic under full fine-tuning
#       - topology_agnostic full seeds 42/123/456
#
# Usage:
#   bash scripts/train/submit_priority_core16.sh
#   bash scripts/train/submit_priority_core16.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
CORE_DIR="$PROJECT_ROOT/results/core_ablation"
LOG_DIR="$PROJECT_ROOT/results/slurm_logs"
TEMPLATE="$SCRIPT_DIR/job_template.sh"
PARTITION="mit_normal_gpu,mit_preemptable,pi_tpoggio"
TIME_LIMIT="06:00:00"
CPUS="8"
MEM="64G"
PYTHON="/home/erimoldi/.conda/envs/sparse-hate/bin/python"
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *)
      printf 'Unknown option: %s\n' "$arg" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$CORE_DIR" "$LOG_DIR"

# priority|job_name|variant|policy|seed|reason
JOBS=(
  "P1|c-c2d-full-s456|coords2d|full|456|Complete coords2d full cell"
  "P1|c-c3d-full-s42|coords3d|full|42|Establish 3D vs 2D under full tuning"
  "P1|c-c3d-full-s123|coords3d|full|123|Establish 3D vs 2D under full tuning"
  "P1|c-c3d-full-s456|coords3d|full|456|Establish 3D vs 2D under full tuning"
  "P1|c-c3r-full-s42|coords3d_reference|full|42|Test reference-aware geometry under full tuning"
  "P1|c-c3r-full-s123|coords3d_reference|full|123|Test reference-aware geometry under full tuning"
  "P1|c-c3r-full-s456|coords3d_reference|full|456|Test reference-aware geometry under full tuning"
  "P2|c-c3r-head-s42|coords3d_reference|head_only|42|Test reference-aware geometry with frozen backbone"
  "P2|c-c3r-head-s123|coords3d_reference|head_only|123|Test reference-aware geometry with frozen backbone"
  "P2|c-c3r-head-s456|coords3d_reference|head_only|456|Test reference-aware geometry with frozen backbone"
  "P2|c-top-head-s42|topology_agnostic|head_only|42|Test layout-robust encoding with frozen backbone"
  "P2|c-top-head-s123|topology_agnostic|head_only|123|Test layout-robust encoding with frozen backbone"
  "P2|c-top-head-s456|topology_agnostic|head_only|456|Test layout-robust encoding with frozen backbone"
  "P3|c-top-full-s42|topology_agnostic|full|42|Complete layout-robust encoding under full tuning"
  "P3|c-top-full-s123|topology_agnostic|full|123|Complete layout-robust encoding under full tuning"
  "P3|c-top-full-s456|topology_agnostic|full|456|Complete layout-robust encoding under full tuning"
)

has_completed_run() {
  local variant="$1"
  local policy="$2"
  local seed="$3"
  local pattern="$CORE_DIR/BNCI2014_001_reve_${variant}_${policy}_seed${seed}_same_layout_*/metrics.json"
  compgen -G "$pattern" > /dev/null
}

is_active_job() {
  local job_name="$1"
  squeue -h -u "${USER:-erimoldi}" -o "%j" 2>/dev/null | grep -Fxq "$job_name"
}

submit_one() {
  local priority="$1"
  local job_name="$2"
  local variant="$3"
  local policy="$4"
  local seed="$5"
  local reason="$6"

  if has_completed_run "$variant" "$policy" "$seed"; then
    printf '[skip-complete] %s %s (%s)\n' "$priority" "$job_name" "$reason"
    return 0
  fi

  if is_active_job "$job_name"; then
    printf '[skip-active]   %s %s already queued/running\n' "$priority" "$job_name"
    return 0
  fi

  local train_args
  train_args="--backbone reve --dataset BNCI2014_001 --num-classes 4 --epochs 50 --batch-size 32 --output-dir $CORE_DIR --spatial-variant $variant --freeze-policy $policy --seed $seed"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run]       %s %s -> %s\n' "$priority" "$job_name" "$train_args"
    return 0
  fi

  local submit_output
  submit_output="$({
    sbatch --parsable \
      --job-name="$job_name" \
      --partition="$PARTITION" \
      --time="$TIME_LIMIT" \
      --cpus-per-task="$CPUS" \
      --mem="$MEM" \
      --gres=gpu:1 \
      --chdir="$PROJECT_ROOT" \
      --output="$LOG_DIR/${job_name}_%j.out" \
      --error="$LOG_DIR/${job_name}_%j.err" \
      --export=ALL,TRAIN_ARGS="$train_args" \
      "$TEMPLATE"
  } 2>&1)"

  printf '[submitted]     %s %s -> %s\n' "$priority" "$job_name" "$submit_output"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$priority" "$job_name" "$variant" "$policy" "$seed" "$submit_output" >> "$MANIFEST"
  sleep 0.2
}

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
MANIFEST="$LOG_DIR/priority_core16_submissions_${TIMESTAMP}.tsv"
printf 'priority\tjob_name\tvariant\tpolicy\tseed\tslurm_job_id\n' > "$MANIFEST"

printf 'Project root: %s\n' "$PROJECT_ROOT"
printf 'Core output:  %s\n' "$CORE_DIR"
printf 'Log dir:      %s\n' "$LOG_DIR"
printf 'Template:     %s\n' "$TEMPLATE"
printf 'Partition:    %s\n' "$PARTITION"
printf 'Dry run:      %s\n' "$DRY_RUN"
printf 'Manifest:     %s\n\n' "$MANIFEST"

for spec in "${JOBS[@]}"; do
  IFS='|' read -r priority job_name variant policy seed reason <<< "$spec"
  submit_one "$priority" "$job_name" "$variant" "$policy" "$seed" "$reason"
done

printf '\nDone. Monitor with:\n'
printf '  squeue -u %s --format="%%.10i %%.16j %%.8T %%.10M %%R"\n' "${USER:-erimoldi}"
printf '  tail -f %s/c-c3r-full-s42_*.out\n' "$LOG_DIR"
printf 'Manifest saved to: %s\n' "$MANIFEST"
