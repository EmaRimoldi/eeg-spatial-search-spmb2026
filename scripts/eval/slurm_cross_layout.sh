#!/usr/bin/env bash
# Submit cross-layout evaluation jobs after core ablation completes.
#
# Finds all completed head_only runs in results/core_ablation/ and submits
# one SLURM eval job per run to evaluate on PhysionetMI.
#
# Usage:
#   bash scripts/eval/slurm_cross_layout.sh              # submit all
#   bash scripts/eval/slurm_cross_layout.sh --dry-run    # preview
#
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/erimoldi/.conda/envs/sparse-hate/bin/python"

RESULTS_DIR="$PROJECT_ROOT/results/core_ablation"
OUTPUT_DIR="$PROJECT_ROOT/results/cross_layout"
LOG_DIR="$OUTPUT_DIR/slurm_logs"
PARTITION="pi_tpoggio"

DRY_RUN=0; [[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

SUBMITTED=0
SKIPPED=0

for RUN_DIR in "$RESULTS_DIR"/*/; do
    [[ -f "$RUN_DIR/best_model.pt" ]] || { SKIPPED=$((SKIPPED+1)); continue; }
    [[ -f "$RUN_DIR/config_resolved.yaml" ]] || { SKIPPED=$((SKIPPED+1)); continue; }

    RUN_NAME="$(basename "$RUN_DIR")"
    JOB_NAME="cl_${RUN_NAME:0:12}"

    CMD=(sbatch --job-name="$JOB_NAME" --partition="$PARTITION"
         --time="01:00:00" --cpus-per-task=4 --mem="16G" --gres="gpu:1"
         --output="$LOG_DIR/${JOB_NAME}_%j.out"
         --error="$LOG_DIR/${JOB_NAME}_%j.err"
         --wrap="source /home/erimoldi/.bashrc || true; cd $PROJECT_ROOT;
             $PYTHON src/evaluation/cross_layout_eval.py
             --model-dir $RUN_DIR
             --eval-dataset PhysionetMI
             --output-dir $OUTPUT_DIR")

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] $JOB_NAME  ←  $RUN_NAME"
    else
        "${CMD[@]}"
        SUBMITTED=$((SUBMITTED+1))
        sleep 0.2
    fi
done

echo ""
echo "Submitted: $SUBMITTED  Skipped (no checkpoint): $SKIPPED"
echo "Run after jobs complete:"
echo "  python src/analysis/aggregate_results.py --results-dir $OUTPUT_DIR"
