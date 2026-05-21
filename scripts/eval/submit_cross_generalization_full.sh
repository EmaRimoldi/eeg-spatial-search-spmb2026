#!/usr/bin/env bash
# Submit a focused cross-generalization battery for the most informative FULL models.
# Rationale: same-layout gains for 3D variants emerged mainly under full fine-tuning,
# so we test transfer there first.
#
# Train dataset: BNCI2014_001
# Eval dataset:  PhysionetMI
# Variants:      none, channel_id, coords2d, coords3d, coords3d_reference, topology_agnostic
# Seeds:         42, 123, 456
#
# Each training job persists best_model.pt, then a dependent cross-layout/cross-dataset
# evaluation job runs automatically.
#
# Usage:
#   bash scripts/eval/submit_cross_generalization_full.sh
#   bash scripts/eval/submit_cross_generalization_full.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TRAIN_TEMPLATE="$PROJECT_ROOT/scripts/train/job_template.sh"
EVAL_TEMPLATE="$PROJECT_ROOT/scripts/eval/cross_eval_job_template.sh"
TRAIN_OUT="$PROJECT_ROOT/results/cross_generalization_train"
EVAL_OUT="$PROJECT_ROOT/results/cross_layout"
TRAIN_LOG="$PROJECT_ROOT/results/slurm_logs"
EVAL_LOG="$EVAL_OUT/slurm_logs"
PARTITIONS="mit_normal_gpu,mit_preemptable,pi_tpoggio"
TRAIN_TIME="06:00:00"
EVAL_TIME="02:00:00"
TRAIN_CPUS="8"
TRAIN_MEM="64G"
EVAL_CPUS="4"
EVAL_MEM="24G"
TRAIN_DATASET="BNCI2014_001"
EVAL_DATASET="PhysionetMI"
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

mkdir -p "$TRAIN_OUT" "$EVAL_OUT" "$TRAIN_LOG" "$EVAL_LOG"

VARIANTS=(none channel_id coords2d coords3d coords3d_reference topology_agnostic)
SEEDS=(42 123 456)
POLICY="full"

existing_checkpoint_glob() {
  local variant="$1" seed="$2"
  printf '%s/%s_reve_%s_%s_seed%s_same_layout_*/best_model.pt' \
    "$TRAIN_OUT" "$TRAIN_DATASET" "$variant" "$POLICY" "$seed"
}

existing_eval_glob() {
  local variant="$1" seed="$2"
  printf '%s/%s_reve_%s_%s_seed%s_same_layout_*_cross_%s.json' \
    "$EVAL_OUT" "$TRAIN_DATASET" "$variant" "$POLICY" "$seed" "$EVAL_DATASET"
}

latest_run_glob() {
  local variant="$1" seed="$2"
  printf '%s/%s_reve_%s_%s_seed%s_same_layout_*' \
    "$TRAIN_OUT" "$TRAIN_DATASET" "$variant" "$POLICY" "$seed"
}

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
MANIFEST="$TRAIN_LOG/cross_generalization_full_${TIMESTAMP}.tsv"
printf 'variant\tseed\ttrain_job_id\teval_job_id\n' > "$MANIFEST"

printf 'Project root:   %s\n' "$PROJECT_ROOT"
printf 'Train out:      %s\n' "$TRAIN_OUT"
printf 'Eval out:       %s\n' "$EVAL_OUT"
printf 'Partitions:     %s\n' "$PARTITIONS"
printf 'Train dataset:  %s\n' "$TRAIN_DATASET"
printf 'Eval dataset:   %s\n' "$EVAL_DATASET"
printf 'Manifest:       %s\n' "$MANIFEST"
printf 'Dry run:        %s\n\n' "$DRY_RUN"

for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    ckpt_glob="$(existing_checkpoint_glob "$variant" "$seed")"
    eval_glob="$(existing_eval_glob "$variant" "$seed")"
    run_glob="$(latest_run_glob "$variant" "$seed")"

    if compgen -G "$eval_glob" > /dev/null; then
      printf '[skip-eval-exists] %s seed=%s\n' "$variant" "$seed"
      continue
    fi

    train_job_id=""
    if compgen -G "$ckpt_glob" > /dev/null; then
      printf '[reuse-checkpoint]  %s seed=%s\n' "$variant" "$seed"
    else
      train_args="--backbone reve --dataset $TRAIN_DATASET --num-classes 4 --epochs 50 --batch-size 32 --output-dir $TRAIN_OUT --spatial-variant $variant --freeze-policy $POLICY --seed $seed --save-checkpoint"
      if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '[dry-train]         %s seed=%s -> %s\n' "$variant" "$seed" "$train_args"
        train_job_id="DRYRUN"
      else
        train_job_id="$({
          sbatch --parsable \
            --job-name="xg-${variant:0:5}-s${seed}" \
            --partition="$PARTITIONS" \
            --time="$TRAIN_TIME" \
            --cpus-per-task="$TRAIN_CPUS" \
            --mem="$TRAIN_MEM" \
            --gres=gpu:1 \
            --chdir="$PROJECT_ROOT" \
            --output="$TRAIN_LOG/xg-${variant:0:5}-s${seed}_%j.out" \
            --error="$TRAIN_LOG/xg-${variant:0:5}-s${seed}_%j.err" \
            --export=ALL,TRAIN_ARGS="$train_args" \
            "$TRAIN_TEMPLATE"
        } 2>&1)"
        printf '[submitted-train]   %s seed=%s -> %s\n' "$variant" "$seed" "$train_job_id"
        sleep 0.2
      fi
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
      printf '[dry-eval]          %s seed=%s -> run_glob=%s\n' "$variant" "$seed" "$run_glob"
      printf '%s\t%s\t%s\t%s\n' "$variant" "$seed" "$train_job_id" "DRYRUN" >> "$MANIFEST"
      continue
    fi

    eval_cmd=(sbatch --parsable
      --job-name="xe-${variant:0:5}-s${seed}"
      --partition="$PARTITIONS"
      --time="$EVAL_TIME"
      --cpus-per-task="$EVAL_CPUS"
      --mem="$EVAL_MEM"
      --gres=gpu:1
      --chdir="$PROJECT_ROOT"
      --output="$EVAL_LOG/xe-${variant:0:5}-s${seed}_%j.out"
      --error="$EVAL_LOG/xe-${variant:0:5}-s${seed}_%j.err"
      --export=ALL,RUN_GLOB="$run_glob",EVAL_DATASET="$EVAL_DATASET",OUTPUT_DIR="$EVAL_OUT")

    if [[ -n "$train_job_id" ]]; then
      eval_cmd+=(--dependency="afterok:$train_job_id")
    fi
    eval_cmd+=("$EVAL_TEMPLATE")

    eval_job_id="$(${eval_cmd[@]} 2>&1)"
    printf '[submitted-eval]    %s seed=%s -> %s\n' "$variant" "$seed" "$eval_job_id"
    printf '%s\t%s\t%s\t%s\n' "$variant" "$seed" "$train_job_id" "$eval_job_id" >> "$MANIFEST"
    sleep 0.2
  done
done

printf '\nDone. Monitor with:\n'
printf '  squeue -u %s --format="%%.10i %%.18j %%.8T %%.10M %%.24P %%R"\n' "${USER:-erimoldi}"
printf 'Manifest saved to: %s\n' "$MANIFEST"
