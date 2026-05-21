#!/usr/bin/env bash
# Submit REVE dataset-expansion runs for BNCI2014_004.
#
# Default sweep:
#   variants: none, coords2d, coords3d_reference
#   regimes:  full, head_only
#   seeds:    42, 123, 456
#
# Usage:
#   bash scripts/train/slurm_bnci2014_004_reve.sh
#   bash scripts/train/slurm_bnci2014_004_reve.sh --regime full
#   bash scripts/train/slurm_bnci2014_004_reve.sh --dry-run
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="BNCI2014_004"
NUM_CLASSES=2
BACKBONE="reve"
EPOCHS=50
BATCH_SIZE=32
VARIANTS=("none" "coords2d" "coords3d_reference")
REGIMES=("full" "head_only")
SEEDS=(42 123 456)
PARTITION="pi_tpoggio"
OUTPUT_DIR="$PROJECT_ROOT/results/dataset_expansion/bnci2014_004"
LOG_DIR="$OUTPUT_DIR/slurm_logs"
JOB_TEMPLATE="$PROJECT_ROOT/scripts/train/job_template.sh"

DRY_RUN=0
REGIME_FILTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --regime)
            REGIME_FILTER="$2"
            shift 2
            ;;
        --regime=*)
            REGIME_FILTER="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

submit_one() {
    local variant="$1"
    local regime="$2"
    local seed="$3"

    local variant_tag="$variant"
    case "$variant" in
        none) variant_tag="none" ;;
        coords2d) variant_tag="c2d" ;;
        coords3d_reference) variant_tag="c3r" ;;
    esac

    local regime_tag="$regime"
    [[ "$regime" == "head_only" ]] && regime_tag="head"

    local job_name="b4_${variant_tag}_${regime_tag}_s${seed}"
    local time_limit="03:00:00"
    [[ "$regime" == "full" ]] && time_limit="06:00:00"

    local train_args
    train_args="--backbone $BACKBONE --dataset $DATASET --num-classes $NUM_CLASSES --epochs $EPOCHS --batch-size $BATCH_SIZE --output-dir $OUTPUT_DIR --spatial-variant $variant --freeze-policy $regime --seed $seed --save-checkpoint"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] $job_name :: $train_args"
        return
    fi

    sbatch \
        --job-name="$job_name" \
        --partition="$PARTITION" \
        --time="$time_limit" \
        --cpus-per-task=8 \
        --mem=32G \
        --gres=gpu:1 \
        --output="$LOG_DIR/${job_name}_%j.out" \
        --error="$LOG_DIR/${job_name}_%j.err" \
        --export=ALL,TRAIN_ARGS="$train_args" \
        "$JOB_TEMPLATE"

    sleep 0.2
}

echo "=== BNCI2014_004 REVE sweep ==="
echo "Variants: ${VARIANTS[*]}"
echo "Regimes:  ${REGIMES[*]}"
echo "Seeds:    ${SEEDS[*]}"
echo "Output:   $OUTPUT_DIR"
echo "Dry-run:  $DRY_RUN"

total=0
for regime in "${REGIMES[@]}"; do
    [[ -n "$REGIME_FILTER" && "$regime" != "$REGIME_FILTER" ]] && continue
    for variant in "${VARIANTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            submit_one "$variant" "$regime" "$seed"
            total=$((total + 1))
        done
    done
done

echo "Done. Consider monitoring with: squeue -u $USER --format='%.10i %.16j %.8T %.9M %R'"
echo "Submitted/previewed jobs: $total"
