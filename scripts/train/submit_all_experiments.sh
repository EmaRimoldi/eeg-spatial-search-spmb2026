#!/usr/bin/env bash
# Master experiment submission — uses all available partitions for fastest start.
#
# Submits all experiments in order of dependency:
#   Stage 1: Pilot   (9 jobs  — 3 variants × 3 seeds)
#   Stage 2: Core    (63 jobs — 7 variants × 3 regimes × 3 seeds)
#   Stage 3: FewShot (75 jobs — 5 variants × 5 fractions × 3 seeds)
#   Stage 4: ChDrop  (48 jobs — 4 variants × 4 dropout rates × 3 seeds)
#   Total  : 195 jobs
#
# Partition strategy (fastest-first):
#   mit_normal_gpu  — 43+ free GPU nodes, 6h limit  ← primary pool
#   ou_bcs_normal   — 18+ free nodes, 1-day limit
#   ou_bcs_low      — 26+ free nodes, 1-day limit
#   ou_bcs_high     — highest priority BCS, 4h limit
#   pi_tpoggio      — lab A100 (8x), 7-day limit
#
# SLURM multi-partition: --partition=A,B,C,D  → first available wins
#
# Usage:
#   bash scripts/train/submit_all_experiments.sh             # submit all
#   bash scripts/train/submit_all_experiments.sh --dry-run   # preview
#   bash scripts/train/submit_all_experiments.sh --stage pilot
#   bash scripts/train/submit_all_experiments.sh --stage core
#   bash scripts/train/submit_all_experiments.sh --stage fewshot
#   bash scripts/train/submit_all_experiments.sh --stage chdrop
#
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/erimoldi/.conda/envs/sparse-hate/bin/python"

# Multi-partition: SLURM picks the first with available GPUs.
# Excluded ou_bcs_high: MaxSubmitPU=4 and nodes unreliable for NFS home.
# Limits: mit_normal_gpu=64, ou_bcs_normal=256, ou_bcs_low=256, pi_tpoggio=large
PARTITIONS="mit_normal_gpu,ou_bcs_normal,ou_bcs_low,pi_tpoggio"

# Time: mit_normal_gpu max=6h, ou_bcs_normal/low max=1day, pi_tpoggio max=7days
TIME="06:00:00"

BACKBONE="reve"
DATASET="BNCI2014_001"
NUM_CLASSES=4
EPOCHS=50
BATCH_SIZE=32

OUTPUT_BASE="$PROJECT_ROOT/results"
LOG_BASE="$PROJECT_ROOT/results/slurm_logs"

DRY_RUN=0
STAGE_FILTER=""
for arg in "$@"; do
    case $arg in
        --dry-run)       DRY_RUN=1 ;;
        --stage=*)       STAGE_FILTER="${arg#*=}" ;;
        --stage)         shift; STAGE_FILTER="$1" ;;
    esac
done

mkdir -p "$LOG_BASE"

TOTAL=0

# ─── submit helper ────────────────────────────────────────────────────────────

submit() {
    local JOB_NAME="$1"; local OUTPUT_DIR="$2"; shift 2
    local TRAIN_ARGS="$*"

    local LOG_DIR="$LOG_BASE"
    mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [DRY] $JOB_NAME  | $TRAIN_ARGS"
        return
    fi

    sbatch \
        --job-name="$JOB_NAME" \
        --partition="$PARTITIONS" \
        --time="$TIME" \
        --cpus-per-task=8 \
        --mem=32G \
        --gres=gpu:1 \
        --output="$LOG_DIR/${JOB_NAME}_%j.out" \
        --error="$LOG_DIR/${JOB_NAME}_%j.err" \
        --wrap="
            set -e
            source /home/erimoldi/.bashrc 2>/dev/null || true
            # Load CUDA if available (safe no-op if not found)
            module load cuda 2>/dev/null || true
            cd $PROJECT_ROOT
            echo \"=== \$(hostname) | GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) ===\"
            $PYTHON src/training/train.py $TRAIN_ARGS
        "
    TOTAL=$((TOTAL + 1))
    sleep 0.15   # avoid scheduler flood
}

# ─── Stage 1: Pilot (3 variants × 3 seeds) ────────────────────────────────────

if [[ -z "$STAGE_FILTER" || "$STAGE_FILTER" == "pilot" ]]; then
    echo "=== STAGE 1: PILOT (9 jobs) ==="
    PILOT_DIR="$OUTPUT_BASE/pilot"

    for VARIANT in channel_id coords2d coords3d; do
        for SEED in 42 123 456; do
            TAG="${VARIANT:0:6}_s${SEED}"
            submit "p_${TAG}" "$PILOT_DIR" \
                --backbone $BACKBONE \
                --spatial-variant $VARIANT \
                --freeze-policy head_only \
                --dataset $DATASET \
                --num-classes $NUM_CLASSES \
                --seed $SEED \
                --epochs $EPOCHS \
                --batch-size $BATCH_SIZE \
                --output-dir "$PILOT_DIR"
        done
    done
fi

# ─── Stage 2: Core ablation (7 variants × 3 regimes × 3 seeds) ───────────────

if [[ -z "$STAGE_FILTER" || "$STAGE_FILTER" == "core" ]]; then
    echo "=== STAGE 2: CORE ABLATION (63 jobs) ==="
    CORE_DIR="$OUTPUT_BASE/core_ablation"

    for VARIANT in none channel_id coords2d coords3d coords3d_distbias coords3d_reference topology_agnostic; do
        for POLICY in head_only partial full; do
            for SEED in 42 123 456; do
                TAG="${VARIANT:0:5}_${POLICY:0:4}_s${SEED}"
                submit "c_${TAG}" "$CORE_DIR" \
                    --backbone $BACKBONE \
                    --spatial-variant $VARIANT \
                    --freeze-policy $POLICY \
                    --dataset $DATASET \
                    --num-classes $NUM_CLASSES \
                    --seed $SEED \
                    --epochs $EPOCHS \
                    --batch-size $BATCH_SIZE \
                    --output-dir "$CORE_DIR"
            done
        done
    done
fi

# ─── Stage 3: Few-shot (5 variants × 5 fractions × 3 seeds) ──────────────────

if [[ -z "$STAGE_FILTER" || "$STAGE_FILTER" == "fewshot" ]]; then
    echo "=== STAGE 3: FEW-SHOT (75 jobs) ==="
    FS_DIR="$OUTPUT_BASE/few_shot"

    for VARIANT in none channel_id coords3d coords3d_reference topology_agnostic; do
        for FRAC in 0.01 0.05 0.10 0.25 1.00; do
            FRAC_TAG="${FRAC/./p}"
            for SEED in 42 123 456; do
                TAG="${VARIANT:0:5}_${FRAC_TAG}_s${SEED}"
                submit "f_${TAG}" "$FS_DIR" \
                    --backbone $BACKBONE \
                    --spatial-variant $VARIANT \
                    --freeze-policy head_only \
                    --dataset $DATASET \
                    --num-classes $NUM_CLASSES \
                    --seed $SEED \
                    --epochs $EPOCHS \
                    --batch-size $BATCH_SIZE \
                    --label-fraction $FRAC \
                    --output-dir "$FS_DIR"
            done
        done
    done
fi

# ─── Stage 4: Channel dropout robustness (4 variants × 4 rates × 3 seeds) ────

if [[ -z "$STAGE_FILTER" || "$STAGE_FILTER" == "chdrop" ]]; then
    echo "=== STAGE 4: CHANNEL DROPOUT (48 jobs) ==="
    CD_DIR="$OUTPUT_BASE/channel_dropout"

    for VARIANT in channel_id coords3d coords3d_reference topology_agnostic; do
        for RATE in 0.0 0.1 0.3 0.5; do
            RATE_TAG="${RATE/./p}"
            for SEED in 42 123 456; do
                TAG="${VARIANT:0:5}_${RATE_TAG}_s${SEED}"
                submit "d_${TAG}" "$CD_DIR" \
                    --backbone $BACKBONE \
                    --spatial-variant $VARIANT \
                    --freeze-policy head_only \
                    --dataset $DATASET \
                    --num-classes $NUM_CLASSES \
                    --seed $SEED \
                    --epochs $EPOCHS \
                    --batch-size $BATCH_SIZE \
                    --channel-dropout $RATE \
                    --output-dir "$CD_DIR"
            done
        done
    done
fi

# ─── Summary ──────────────────────────────────────────────────────────────────

echo ""
if [[ $DRY_RUN -eq 1 ]]; then
    echo "DRY-RUN complete. Re-run without --dry-run to submit."
else
    echo "Submitted $TOTAL jobs across: $PARTITIONS"
    echo ""
    echo "Monitor:"
    echo "  squeue -u \$USER --format='%.10i %.12j %.8T %.8M %R' | head -30"
    echo "  watch -n 30 'squeue -u \$USER | grep -c RUNNING'"
    echo ""
    echo "Once pilot (stage 1) completes:"
    echo "  $PYTHON src/analysis/aggregate_results.py --results-dir $OUTPUT_BASE/pilot"
    echo ""
    echo "Once all stages complete:"
    echo "  bash scripts/eval/slurm_cross_layout.sh"
fi
