#!/usr/bin/env bash
# Pilot experiment: REVE + MOABB + {channel_id, coords2d, coords3d} + head_only + 3 seeds
# Stage 1 — verify the pipeline works before large sweeps
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

BACKBONE="reve"
DATASET="BNCI2014_001"
NUM_CLASSES=4
EPOCHS=50
BATCH_SIZE=32
FREEZE_POLICY="head_only"
SPATIAL_VARIANTS=("channel_id" "coords2d" "coords3d")
SEEDS=(42 123 456)
OUTPUT_DIR="results/logs"

echo "=============================="
echo "PILOT EXPERIMENT"
echo "Backbone: $BACKBONE"
echo "Dataset: $DATASET"
echo "Variants: ${SPATIAL_VARIANTS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "=============================="

for VARIANT in "${SPATIAL_VARIANTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo ">>> Running: variant=$VARIANT, seed=$SEED"
        python src/training/train.py \
            --backbone "$BACKBONE" \
            --spatial-variant "$VARIANT" \
            --freeze-policy "$FREEZE_POLICY" \
            --dataset "$DATASET" \
            --num-classes "$NUM_CLASSES" \
            --seed "$SEED" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --output-dir "$OUTPUT_DIR" \
            2>&1 | tee -a "results/logs/pilot_stdout.log"
    done
done

echo ""
echo "=============================="
echo "Pilot complete. Results in: $OUTPUT_DIR"
echo "Run analysis: python src/analysis/aggregate_results.py --experiment pilot"
echo "=============================="
