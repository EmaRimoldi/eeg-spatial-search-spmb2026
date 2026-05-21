#!/usr/bin/env bash
# Missing-channel robustness experiment
# Stage 4 — simulate practical EEG incompleteness
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

BACKBONE="reve"
DATASET="BNCI2014_001"
NUM_CLASSES=4
EPOCHS=100
BATCH_SIZE=32
SPATIAL_VARIANTS=("channel_id" "coords3d" "coords3d_reference" "topology_agnostic")
DROPOUT_PROBS=(0.1 0.3 0.5)
SEEDS=(42 123 456)
OUTPUT_DIR="results/logs"

echo "=============================="
echo "CHANNEL DROPOUT ROBUSTNESS"
echo "=============================="

for VARIANT in "${SPATIAL_VARIANTS[@]}"; do
    for PROB in "${DROPOUT_PROBS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo ">>> variant=$VARIANT, dropout_p=$PROB, seed=$SEED"
            python src/training/train.py \
                --backbone "$BACKBONE" \
                --spatial-variant "$VARIANT" \
                --freeze-policy "head_only" \
                --dataset "$DATASET" \
                --num-classes "$NUM_CLASSES" \
                --seed "$SEED" \
                --epochs "$EPOCHS" \
                --batch-size "$BATCH_SIZE" \
                --output-dir "$OUTPUT_DIR" \
                2>&1 | tee -a "results/logs/channel_dropout_stdout.log"
        done
    done
done

echo "Channel dropout runs complete."
