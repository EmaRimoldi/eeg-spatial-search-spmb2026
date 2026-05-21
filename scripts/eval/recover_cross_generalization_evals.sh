#!/usr/bin/env bash
# Recover cross-generalization evals after fixing the evaluation pipeline.
#
# - Warms the target dataset cache once (default: PhysionetMI test split).
# - Submits fresh eval jobs with dependency on BOTH the original training job
#   and the cache-warm job.
# - Reuses the latest cross_generalization_full manifest for train job ids.
#
# Usage:
#   bash scripts/eval/recover_cross_generalization_evals.sh
#   bash scripts/eval/recover_cross_generalization_evals.sh --dry-run
#   bash scripts/eval/recover_cross_generalization_evals.sh /path/to/manifest.tsv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_ROOT/results/slurm_logs"
EVAL_OUT="$PROJECT_ROOT/results/cross_layout"
EVAL_LOG="$EVAL_OUT/slurm_logs"
TRAIN_OUT="$PROJECT_ROOT/results/cross_generalization_train"
CACHE_TEMPLATE="$PROJECT_ROOT/scripts/eval/cache_warm_job_template.sh"
EVAL_TEMPLATE="$PROJECT_ROOT/scripts/eval/cross_eval_job_template.sh"
CACHE_PARTITION="mit_normal"
CACHE_TIME="04:00:00"
CACHE_CPUS="4"
CACHE_MEM="32G"
EVAL_PARTITIONS="mit_normal_gpu,mit_preemptable,pi_tpoggio"
EVAL_TIME="04:00:00"
EVAL_CPUS="4"
EVAL_MEM="24G"
EVAL_DATASET="PhysionetMI"
TRAIN_DATASET="BNCI2014_001"
DRY_RUN=0
MANIFEST=""
CACHE_JOB_OVERRIDE=""
CACHE_READY_FILE=""

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --cache-job-id=*) CACHE_JOB_OVERRIDE="${arg#*=}" ;;
    *.tsv) MANIFEST="$arg" ;;
    *)
      printf 'Unknown option: %s\n' "$arg" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MANIFEST" ]]; then
  MANIFEST="$(ls -1t "$LOG_DIR"/cross_generalization_full_*.tsv 2>/dev/null | head -1 || true)"
fi
if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
  printf 'No cross-generalization manifest found in %s\n' "$LOG_DIR" >&2
  exit 1
fi

mkdir -p "$EVAL_OUT" "$EVAL_LOG"

short_tag() {
  case "$1" in
    none) echo "none" ;;
    channel_id) echo "chan" ;;
    coords2d) echo "c2d" ;;
    coords3d) echo "c3d" ;;
    coords3d_reference) echo "c3r" ;;
    topology_agnostic) echo "top" ;;
    *) echo "$1" ;;
  esac
}

existing_eval_glob() {
  local variant="$1" seed="$2"
  printf '%s/%s_reve_%s_full_seed%s_same_layout_*_cross_%s.json' \
    "$EVAL_OUT" "$TRAIN_DATASET" "$variant" "$seed" "$EVAL_DATASET"
}

checkpoint_glob() {
  local variant="$1" seed="$2"
  printf '%s/%s_reve_%s_full_seed%s_same_layout_*/best_model.pt' \
    "$TRAIN_OUT" "$TRAIN_DATASET" "$variant" "$seed"
}

latest_run_glob() {
  local variant="$1" seed="$2"
  printf '%s/%s_reve_%s_full_seed%s_same_layout_*' \
    "$TRAIN_OUT" "$TRAIN_DATASET" "$variant" "$seed"
}

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_MANIFEST="$LOG_DIR/cross_generalization_eval_recovery_${TIMESTAMP}.tsv"
printf 'variant\tseed\ttrain_job_id\tcache_job_id\teval_job_id\n' > "$OUT_MANIFEST"

printf 'Using source manifest: %s\n' "$MANIFEST"
printf 'Eval dataset:          %s\n' "$EVAL_DATASET"
printf 'Eval partitions:       %s\n' "$EVAL_PARTITIONS"
printf 'Cache partition:       %s\n' "$CACHE_PARTITION"
printf 'Cache override:        %s\n' "${CACHE_JOB_OVERRIDE:-<none>}"
printf 'Output manifest:       %s\n' "$OUT_MANIFEST"
printf 'Dry run:               %s\n\n' "$DRY_RUN"

CACHE_READY_FILE="$PROJECT_ROOT/data/processed/$EVAL_DATASET/processed_test.npz"

CACHE_JOB_ID=""
if [[ -n "$CACHE_JOB_OVERRIDE" ]]; then
  CACHE_JOB_ID="$CACHE_JOB_OVERRIDE"
  printf '[reuse-cache-job] %s\n\n' "$CACHE_JOB_ID"
elif [[ "$DRY_RUN" -eq 1 ]]; then
  CACHE_JOB_ID="DRYRUN"
  printf '[dry-cache] submit cache warm for %s test split\n' "$EVAL_DATASET"
else
  CACHE_JOB_ID="$({
    sbatch --parsable \
      --job-name="cw-phys-test" \
      --partition="$CACHE_PARTITION" \
      --time="$CACHE_TIME" \
      --cpus-per-task="$CACHE_CPUS" \
      --mem="$CACHE_MEM" \
      --chdir="$PROJECT_ROOT" \
      --output="$EVAL_LOG/cw-phys-test_%j.out" \
      --error="$EVAL_LOG/cw-phys-test_%j.err" \
      --export=ALL,CACHE_DATASET="$EVAL_DATASET",CACHE_SPLITS="test",CACHE_BATCH_SIZE="32" \
      "$CACHE_TEMPLATE"
  } 2>&1)"
  printf '[submitted-cache] %s\n\n' "$CACHE_JOB_ID"
fi

while IFS=$'\t' read -r variant seed train_job_id old_eval_job_id; do
  if [[ "$variant" == "variant" || -z "$variant" ]]; then
    continue
  fi

  if compgen -G "$(existing_eval_glob "$variant" "$seed")" > /dev/null; then
    printf '[skip-eval-exists] %s seed=%s\n' "$variant" "$seed"
    continue
  fi

  tag="$(short_tag "$variant")"
  run_glob="$(latest_run_glob "$variant" "$seed")"
  ckpt_glob="$(checkpoint_glob "$variant" "$seed")"
  train_active=0
  if squeue -h -j "$train_job_id" >/dev/null 2>&1; then
    train_active=1
  fi

  dep_parts=()
  if [[ ! -f "$CACHE_READY_FILE" ]]; then
    dep_parts+=("$CACHE_JOB_ID")
  fi

  if [[ "$train_active" -eq 1 ]]; then
    dep_parts+=("$train_job_id")
  elif compgen -G "$ckpt_glob" > /dev/null; then
    :
  else
    printf '[skip-no-checkpoint] %s seed=%s (train job %s not active, no checkpoint yet)\n' "$variant" "$seed" "$train_job_id"
    continue
  fi

  dep_string=""
  if [[ ${#dep_parts[@]} -gt 0 ]]; then
    dep_string="afterok:$(IFS=:; echo "${dep_parts[*]}")"
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-eval] %s seed=%s dep=%s\n' "$variant" "$seed" "${dep_string:-<none>}"
    printf '%s\t%s\t%s\t%s\t%s\n' "$variant" "$seed" "$train_job_id" "$CACHE_JOB_ID" "DRYRUN" >> "$OUT_MANIFEST"
    continue
  fi

  eval_cmd=(sbatch --parsable
    --job-name="xf-${tag}-s${seed}"
    --partition="$EVAL_PARTITIONS"
    --time="$EVAL_TIME"
    --cpus-per-task="$EVAL_CPUS"
    --mem="$EVAL_MEM"
    --gres=gpu:1
    --chdir="$PROJECT_ROOT"
    --output="$EVAL_LOG/xf-${tag}-s${seed}_%j.out"
    --error="$EVAL_LOG/xf-${tag}-s${seed}_%j.err"
    --export=ALL,RUN_GLOB="$run_glob",EVAL_DATASET="$EVAL_DATASET",OUTPUT_DIR="$EVAL_OUT")
  if [[ -n "$dep_string" ]]; then
    eval_cmd+=(--dependency="$dep_string")
  fi
  eval_cmd+=("$EVAL_TEMPLATE")

  eval_job_id="$(${eval_cmd[@]} 2>&1)"
  printf '[submitted-eval] %s seed=%s -> %s\n' "$variant" "$seed" "$eval_job_id"
  printf '%s\t%s\t%s\t%s\t%s\n' "$variant" "$seed" "$train_job_id" "$CACHE_JOB_ID" "$eval_job_id" >> "$OUT_MANIFEST"
  sleep 0.2
done < "$MANIFEST"

printf '\nDone. Monitor with:\n'
printf '  squeue -u %s --format="%%.10i %%.18j %%.8T %%.10M %%.24P %%R"\n' "${USER:-erimoldi}"
printf 'Output manifest: %s\n' "$OUT_MANIFEST"
