#!/usr/bin/env bash
# Core ablation: all 7 spatial variants × 3 regimes × 3 seeds
# Stage 2 — answer the main paper question
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

BACKBONE="reve"
DATASET="BNCI2014_001"
NUM_CLASSES=4
EPOCHS=100
BATCH_SIZE=32
SPATIAL_VARIANTS=("none" "channel_id" "coords2d" "coords3d" "coords3d_distbias" "coords3d_reference" "topology_agnostic")
FREEZE_POLICIES=("frozen" "head_only" "partial")
SEEDS=(42 123 456)
OUTPUT_DIR="results/logs"

TOTAL_RUNS=$(( ${#SPATIAL_VARIANTS[@]} * ${#FREEZE_POLICIES[@]} * ${#SEEDS[@]} ))

echo "=============================="
echo "CORE ABLATION"
echo "Backbone: $BACKBONE"
echo "Dataset: $DATASET"
echo "Variants: ${SPATIAL_VARIANTS[*]}"
echo "Regimes: ${FREEZE_POLICIES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total runs: $TOTAL_RUNS"
echo "=============================="

RUN_COUNT=0
for VARIANT in "${SPATIAL_VARIANTS[@]}"; do
    for REGIME in "${FREEZE_POLICIES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            RUN_COUNT=$((RUN_COUNT + 1))
            echo ""
            echo ">>> [$RUN_COUNT/$TOTAL_RUNS] variant=$VARIANT, regime=$REGIME, seed=$SEED"
            python src/training/train.py \
                --backbone "$BACKBONE" \
                --spatial-variant "$VARIANT" \
                --freeze-policy "$REGIME" \
                --dataset "$DATASET" \
                --num-classes "$NUM_CLASSES" \
                --seed "$SEED" \
                --epochs "$EPOCHS" \
                --batch-size "$BATCH_SIZE" \
                --output-dir "$OUTPUT_DIR" \
                2>&1 | tee -a "results/logs/core_ablation_stdout.log"
        done
    done
done

echo ""
echo "=============================="
echo "Core ablation complete."
echo "Run analysis: python src/analysis/aggregate_results.py --experiment core_ablation"
echo "=============================="
