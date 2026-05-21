#!/usr/bin/env bash
# Submit core ablation as SLURM jobs.
#
# Stage 2: all 7 spatial variants × 3 freeze regimes × 3 seeds = 63 runs
# Estimated wall time: ~2h per run on GPU (head_only faster; full finetuning slower)
#
# Usage:
#   bash scripts/train/slurm_core_ablation.sh             # submit all
#   bash scripts/train/slurm_core_ablation.sh --dry-run   # preview
#   bash scripts/train/slurm_core_ablation.sh --regime head_only  # one regime only
#
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/erimoldi/.conda/envs/sparse-hate/bin/python"

# --- Experiment config ---
BACKBONE="reve"
DATASET="BNCI2014_001"
NUM_CLASSES=4
EPOCHS=50
BATCH_SIZE=32
SPATIAL_VARIANTS=("none" "channel_id" "coords2d" "coords3d" "coords3d_distbias" "coords3d_reference" "topology_agnostic")
FREEZE_POLICIES=("head_only" "partial" "full")
SEEDS=(42 123 456)
OUTPUT_DIR="$PROJECT_ROOT/results/core_ablation"

# --- SLURM settings ---
PARTITION="pi_tpoggio"
SEEDS=(42 123 456)
LOG_DIR="$PROJECT_ROOT/results/core_ablation/slurm_logs"

# Time limits per regime (head_only is fast, full is slow)
declare -A TIME_LIMITS
TIME_LIMITS["head_only"]="02:00:00"
TIME_LIMITS["partial"]="04:00:00"
TIME_LIMITS["full"]="08:00:00"

DRY_RUN=0
REGIME_FILTER=""
for arg in "$@"; do
    case $arg in
        --dry-run)    DRY_RUN=1 ;;
        --regime)     shift; REGIME_FILTER="$1" ;;
        --regime=*)   REGIME_FILTER="${arg#*=}" ;;
    esac
done

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "=============================="
echo "EEG SPATIAL PAPER — CORE ABLATION"
echo "Backbone:  $BACKBONE"
echo "Dataset:   $DATASET"
echo "Variants:  ${SPATIAL_VARIANTS[*]}"
echo "Regimes:   ${FREEZE_POLICIES[*]}"
echo "Seeds:     ${SEEDS[*]}"
echo "Total:     $((${#SPATIAL_VARIANTS[@]} * ${#FREEZE_POLICIES[@]} * ${#SEEDS[@]})) runs"
echo "Dry-run:   $DRY_RUN"
echo "=============================="

SUBMITTED=0

for VARIANT in "${SPATIAL_VARIANTS[@]}"; do
    for POLICY in "${FREEZE_POLICIES[@]}"; do
        [[ -n "$REGIME_FILTER" && "$POLICY" != "$REGIME_FILTER" ]] && continue

        TIME="${TIME_LIMITS[$POLICY]:-04:00:00}"
        MEM="32G"
        [[ "$POLICY" == "full" ]] && MEM="48G"

        for SEED in "${SEEDS[@]}"; do
            JOB_NAME="eeg_${VARIANT:0:5}_${POLICY:0:4}_s${SEED}"

            SBATCH_CMD=(
                sbatch
                --job-name="$JOB_NAME"
                --partition="$PARTITION"
                --time="$TIME"
                --cpus-per-task=4
                --mem="$MEM"
                --gres="gpu:1"
                --output="$LOG_DIR/${JOB_NAME}_%j.out"
                --error="$LOG_DIR/${JOB_NAME}_%j.err"
                --wrap="
                    source /home/erimoldi/.bashrc || true
                    cd $PROJECT_ROOT
                    $PYTHON src/training/train.py \
                        --backbone $BACKBONE \
                        --spatial-variant $VARIANT \
                        --freeze-policy $POLICY \
                        --dataset $DATASET \
                        --num-classes $NUM_CLASSES \
                        --seed $SEED \
                        --epochs $EPOCHS \
                        --batch-size $BATCH_SIZE \
                        --output-dir $OUTPUT_DIR
                "
            )

            if [[ $DRY_RUN -eq 1 ]]; then
                echo "[DRY-RUN] variant=$VARIANT policy=$POLICY seed=$SEED  time=$TIME"
            else
                "${SBATCH_CMD[@]}"
                echo "  Submitted: $JOB_NAME"
                SUBMITTED=$((SUBMITTED + 1))
                sleep 0.2   # avoid overwhelming scheduler
            fi
        done
    done
done

echo ""
if [[ $DRY_RUN -eq 0 ]]; then
    echo "Submitted $SUBMITTED jobs."
    echo "Monitor: squeue -u \$USER"
    echo "Logs:    $LOG_DIR/"
    echo ""
    echo "Once done, aggregate:"
    echo "  $PYTHON src/analysis/aggregate_results.py --results-dir $OUTPUT_DIR"
    echo "  $PYTHON src/analysis/stat_tests.py --results-dir $OUTPUT_DIR"
    echo "  $PYTHON src/plotting/plot_core_ablation.py --results-dir $OUTPUT_DIR"
fi
