#!/usr/bin/env bash
# Channel dropout robustness: train with dropout, test degradation.
# Stage 4: 4 variants × 4 dropout rates × 3 seeds = 48 runs
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/erimoldi/.conda/envs/sparse-hate/bin/python"

BACKBONE="reve"
DATASET="BNCI2014_001"
NUM_CLASSES=4
EPOCHS=50
BATCH_SIZE=32
FREEZE_POLICY="head_only"
SPATIAL_VARIANTS=("channel_id" "coords3d" "coords3d_reference" "topology_agnostic")
DROPOUT_RATES=("0.0" "0.1" "0.3" "0.5")
SEEDS=(42 123 456)
OUTPUT_DIR="$PROJECT_ROOT/results/channel_dropout"
LOG_DIR="$OUTPUT_DIR/slurm_logs"
PARTITION="pi_tpoggio"

DRY_RUN=0; [[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
echo "Channel dropout: ${#SPATIAL_VARIANTS[@]} variants × ${#DROPOUT_RATES[@]} rates × ${#SEEDS[@]} seeds = $((${#SPATIAL_VARIANTS[@]}*${#DROPOUT_RATES[@]}*${#SEEDS[@]})) runs"

SUBMITTED=0
for VARIANT in "${SPATIAL_VARIANTS[@]}"; do
  for RATE in "${DROPOUT_RATES[@]}"; do
    RATE_TAG="${RATE/./p}"
    for SEED in "${SEEDS[@]}"; do
      JOB_NAME="cd_${VARIANT:0:5}_${RATE_TAG}_s${SEED}"
      CMD=(sbatch --job-name="$JOB_NAME" --partition="$PARTITION"
           --time="02:00:00" --cpus-per-task=4 --mem="32G" --gres="gpu:1"
           --output="$LOG_DIR/${JOB_NAME}_%j.out"
           --error="$LOG_DIR/${JOB_NAME}_%j.err"
           --wrap="source /home/erimoldi/.bashrc || true; cd $PROJECT_ROOT;
               $PYTHON src/training/train.py
               --backbone $BACKBONE --spatial-variant $VARIANT
               --freeze-policy $FREEZE_POLICY --dataset $DATASET
               --num-classes $NUM_CLASSES --seed $SEED
               --epochs $EPOCHS --batch-size $BATCH_SIZE
               --channel-dropout $RATE --output-dir $OUTPUT_DIR")
      if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] $JOB_NAME rate=$RATE"
      else
        "${CMD[@]}"; SUBMITTED=$((SUBMITTED+1)); sleep 0.2
      fi
    done
  done
done
echo "Submitted $SUBMITTED jobs."
