#!/usr/bin/env bash
# Cross-layout evaluation: test a model trained on one montage layout on another.
#
# Experiment:
#   Train on BNCI2014_001 (22-ch standard motor imagery montage)
#   Evaluate on PhysionetMI (64-ch layout)
#
# This tests whether 3D coordinate embedding generalizes to unseen electrode layouts
# (the core generalization claim of the paper).
#
# Usage:
#   bash scripts/eval/eval_cross_layout.sh --model-dir results/core_ablation/BNCI2014_001_reve_coords3d_head_only_seed42_*
#   bash scripts/eval/eval_cross_layout.sh --dry-run
#
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/erimoldi/.conda/envs/sparse-hate/bin/python"

MODEL_DIR=""
DRY_RUN=0

for arg in "$@"; do
    case $arg in
        --model-dir=*) MODEL_DIR="${arg#*=}" ;;
        --model-dir)   shift; MODEL_DIR="$1" ;;
        --dry-run)     DRY_RUN=1 ;;
    esac
done

if [[ -z "$MODEL_DIR" && $DRY_RUN -eq 0 ]]; then
    echo "Usage: $0 --model-dir <path-to-run-dir>"
    echo "  or:  $0 --dry-run"
    exit 1
fi

echo "=============================="
echo "EEG SPATIAL PAPER — CROSS-LAYOUT EVAL"
echo "Train:  BNCI2014_001 (22ch)"
echo "Test:   PhysionetMI  (64ch)"
echo "Model:  $MODEL_DIR"
echo "=============================="

CMD=(
    "$PYTHON"
    src/evaluation/cross_layout_eval.py
    --model-dir "$MODEL_DIR"
    --eval-dataset PhysionetMI
    --output-dir results/cross_layout
)

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] ${CMD[*]}"
else
    "${CMD[@]}"
fi
