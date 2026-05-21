#!/usr/bin/env bash
# Submit pilot experiment as a SLURM job array.
#
# Usage:
#   bash scripts/train/slurm_pilot.sh                    # submit all variants
#   bash scripts/train/slurm_pilot.sh --dry-run          # print sbatch commands only
#
# Each job: one spatial variant × one seed
# Partition: pi_tpoggi (your lab partition observed in squeue)
#
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/erimoldi/.conda/envs/sparse-hate/bin/python"

# --- Experiment config ---
BACKBONE="reve"
DATASET="BNCI2014_001"       # Real motor imagery data (MOABB, auto-downloaded)
NUM_CLASSES=4
EPOCHS=50
BATCH_SIZE=32
FREEZE_POLICY="head_only"
SPATIAL_VARIANTS=("channel_id" "coords2d" "coords3d")   # Pilot: 3 variants
SEEDS=(42 123 456)
OUTPUT_DIR="$PROJECT_ROOT/results/pilot"

# --- SLURM settings ---
PARTITION="pi_tpoggio"
TIME="02:00:00"
CPUS=4
MEM="32G"
GPU="1"               # 1 GPU per job
LOG_DIR="$PROJECT_ROOT/results/pilot/slurm_logs"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "=============================="
echo "EEG SPATIAL PAPER — PILOT SWEEP"
echo "Backbone:  $BACKBONE"
echo "Dataset:   $DATASET"
echo "Variants:  ${SPATIAL_VARIANTS[*]}"
echo "Seeds:     ${SEEDS[*]}"
echo "Partition: $PARTITION"
echo "Dry-run:   $DRY_RUN"
echo "=============================="

SUBMITTED=0

for VARIANT in "${SPATIAL_VARIANTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        JOB_NAME="eeg_${VARIANT:0:6}_s${SEED}"

        SBATCH_CMD=(
            sbatch
            --job-name="$JOB_NAME"
            --partition="$PARTITION"
            --time="$TIME"
            --cpus-per-task="$CPUS"
            --mem="$MEM"
            --gres="gpu:$GPU"
            --output="$LOG_DIR/${JOB_NAME}_%j.out"
            --error="$LOG_DIR/${JOB_NAME}_%j.err"
            --wrap="
                source /home/erimoldi/.bashrc || true
                cd $PROJECT_ROOT
                $PYTHON src/training/train.py \
                    --backbone $BACKBONE \
                    --spatial-variant $VARIANT \
                    --freeze-policy $FREEZE_POLICY \
                    --dataset $DATASET \
                    --num-classes $NUM_CLASSES \
                    --seed $SEED \
                    --epochs $EPOCHS \
                    --batch-size $BATCH_SIZE \
                    --output-dir $OUTPUT_DIR
            "
        )

        if [[ $DRY_RUN -eq 1 ]]; then
            echo "[DRY-RUN] ${SBATCH_CMD[*]}"
        else
            "${SBATCH_CMD[@]}"
            echo "  Submitted: variant=$VARIANT seed=$SEED → $JOB_NAME"
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
done

echo ""
if [[ $DRY_RUN -eq 0 ]]; then
    echo "Submitted $SUBMITTED jobs. Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f $LOG_DIR/*.out"
    echo ""
    echo "Once done, aggregate results:"
    echo "  $PYTHON src/analysis/aggregate_results.py --results-dir $OUTPUT_DIR"
fi
