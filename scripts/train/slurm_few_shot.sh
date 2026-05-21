#!/usr/bin/env bash
# Few-shot experiment: vary label fraction across top spatial variants.
# Stage 5: 5 variants × 5 label fractions × 3 seeds = 75 runs
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/erimoldi/.conda/envs/sparse-hate/bin/python"

BACKBONE="reve"
DATASET="BNCI2014_001"
NUM_CLASSES=4
EPOCHS=50
BATCH_SIZE=32
FREEZE_POLICY="head_only"
SPATIAL_VARIANTS=("none" "channel_id" "coords3d" "coords3d_reference" "topology_agnostic")
LABEL_FRACTIONS=("0.01" "0.05" "0.10" "0.25" "1.00")
SEEDS=(42 123 456)
OUTPUT_DIR="$PROJECT_ROOT/results/few_shot"
LOG_DIR="$OUTPUT_DIR/slurm_logs"
PARTITION="pi_tpoggio"

DRY_RUN=0; [[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
echo "Few-shot: ${#SPATIAL_VARIANTS[@]} variants × ${#LABEL_FRACTIONS[@]} fractions × ${#SEEDS[@]} seeds = $((${#SPATIAL_VARIANTS[@]}*${#LABEL_FRACTIONS[@]}*${#SEEDS[@]})) runs"

SUBMITTED=0
for VARIANT in "${SPATIAL_VARIANTS[@]}"; do
  for FRAC in "${LABEL_FRACTIONS[@]}"; do
    FRAC_TAG="${FRAC/./p}"   # 0.05 → 0p05
    for SEED in "${SEEDS[@]}"; do
      JOB_NAME="fs_${VARIANT:0:5}_${FRAC_TAG}_s${SEED}"
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
               --label-fraction $FRAC --output-dir $OUTPUT_DIR")
      if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] $JOB_NAME frac=$FRAC"
      else
        "${CMD[@]}"; SUBMITTED=$((SUBMITTED+1)); sleep 0.2
      fi
    done
  done
done
echo "Submitted $SUBMITTED jobs."
