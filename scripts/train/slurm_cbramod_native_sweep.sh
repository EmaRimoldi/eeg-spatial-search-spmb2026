#!/usr/bin/env bash
# Submit native-spatial CBraMod comparison jobs.
#
# Default matrix:
#   Datasets: BNCI2014_001, BNCI2014_004, Cho2017(seed 42 only)
#   Optional: Shin2017A (enable with --include-shin or target explicitly via --dataset Shin2017A)
#   Variants: none, coords2d, coords3d_reference
#   Regime:   full fine-tuning via configs/models/cbramod_full.yaml
#   Seeds:    42, 123 for BNCI datasets; 42 for Cho2017
#
# Usage:
#   bash scripts/train/slurm_cbramod_native_sweep.sh
#   bash scripts/train/slurm_cbramod_native_sweep.sh --dry-run
#   bash scripts/train/slurm_cbramod_native_sweep.sh --dataset BNCI2014_001
#   bash scripts/train/slurm_cbramod_native_sweep.sh --epochs 25
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$PROJECT_ROOT/configs/models/cbramod_full.yaml"
OUTPUT_ROOT="$PROJECT_ROOT/results/cbramod_native"
LOG_DIR="$OUTPUT_ROOT/slurm_logs"
JOB_TEMPLATE="$PROJECT_ROOT/scripts/train/job_template.sh"
PARTITION="pi_tpoggio"
VARIANTS=("none" "coords2d" "coords3d_reference")
BNCI_SEEDS=(42 123)
CHO_SEEDS=(42)
SHIN_SEEDS=(42)
EPOCHS=20
DRY_RUN=0
DATASET_FILTER=""
INCLUDE_CHO=1
INCLUDE_SHIN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --dataset)
            DATASET_FILTER="$2"
            shift 2
            ;;
        --dataset=*)
            DATASET_FILTER="${1#*=}"
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --epochs=*)
            EPOCHS="${1#*=}"
            shift
            ;;
        --no-cho)
            INCLUDE_CHO=0
            shift
            ;;
        --include-shin)
            INCLUDE_SHIN=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

dataset_num_classes() {
    case "$1" in
        BNCI2014_001) echo 4 ;;
        BNCI2014_004|Cho2017|Shin2017A) echo 2 ;;
        *)
            echo "Unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}

dataset_batch_size() {
    case "$1" in
        BNCI2014_001|BNCI2014_004) echo 32 ;;
        Cho2017|Shin2017A) echo 16 ;;
        *)
            echo "Unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}

dataset_time_limit() {
    case "$1" in
        BNCI2014_001|BNCI2014_004) echo "04:00:00" ;;
        Cho2017) echo "06:00:00" ;;
        Shin2017A) echo "08:00:00" ;;
        *)
            echo "Unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}

variant_tag() {
    case "$1" in
        none) echo "none" ;;
        coords2d) echo "c2d" ;;
        coords3d_reference) echo "c3r" ;;
        *) echo "$1" ;;
    esac
}

submit_one() {
    local dataset="$1"
    local variant="$2"
    local seed="$3"
    local num_classes batch_size time_limit dtag vtag job_name train_args

    num_classes="$(dataset_num_classes "$dataset")"
    batch_size="$(dataset_batch_size "$dataset")"
    time_limit="$(dataset_time_limit "$dataset")"
    dtag="$(echo "$dataset" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]')"
    vtag="$(variant_tag "$variant")"
    job_name="cbn_${dtag}_${vtag}_s${seed}"

    train_args="--backbone cbramod --dataset $dataset --num-classes $num_classes --epochs $EPOCHS --batch-size $batch_size --output-dir $OUTPUT_ROOT --save-checkpoint --config $CONFIG --spatial-variant $variant --seed $seed"

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

queue_dataset() {
    local dataset="$1"
    shift
    local seeds=("$@")
    [[ -n "$DATASET_FILTER" && "$dataset" != "$DATASET_FILTER" ]] && return

    for variant in "${VARIANTS[@]}"; do
        for seed in "${seeds[@]}"; do
            submit_one "$dataset" "$variant" "$seed"
        done
    done
}

echo "=== Native-spatial CBraMod sweep ==="
echo "Config:    $CONFIG"
echo "Variants:  ${VARIANTS[*]}"
echo "BNCI seeds:${BNCI_SEEDS[*]}"
echo "Cho seeds: ${CHO_SEEDS[*]}"
echo "Shin seeds:${SHIN_SEEDS[*]}"
echo "Epochs:    $EPOCHS"
echo "Output:    $OUTPUT_ROOT"
echo "Dry-run:   $DRY_RUN"
echo "Filter:    ${DATASET_FILTER:-<none>}"

queue_dataset "BNCI2014_001" "${BNCI_SEEDS[@]}"
queue_dataset "BNCI2014_004" "${BNCI_SEEDS[@]}"
if [[ $INCLUDE_CHO -eq 1 ]]; then
    queue_dataset "Cho2017" "${CHO_SEEDS[@]}"
fi
if [[ $INCLUDE_SHIN -eq 1 || "${DATASET_FILTER:-}" == "Shin2017A" ]]; then
    queue_dataset "Shin2017A" "${SHIN_SEEDS[@]}"
fi

echo "Done. Monitor with: squeue -u $USER --format='%.10i %.22j %.8T %.10M %R'"
