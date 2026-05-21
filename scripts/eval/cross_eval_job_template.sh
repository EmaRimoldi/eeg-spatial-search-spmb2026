#!/bin/bash
# SLURM job template for cross-layout / cross-dataset evaluation.
# Requires env vars:
#   RUN_GLOB      glob that resolves to exactly one latest run directory
#   EVAL_DATASET  target dataset name (e.g. PhysionetMI)
#   OUTPUT_DIR    destination directory for JSON results

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/erimoldi/openclaw_remote/projects/eeg-spatial-paper}"
export PATH="/home/erimoldi/.conda/envs/sparse-hate/bin:$PATH"
export PYTHONNOUSERSITE=1
PYTHON="/home/erimoldi/.conda/envs/sparse-hate/bin/python"

cd "$PROJECT_ROOT"

if [[ -z "${RUN_GLOB:-}" ]]; then
  echo "RUN_GLOB is required" >&2
  exit 2
fi
if [[ -z "${EVAL_DATASET:-}" ]]; then
  echo "EVAL_DATASET is required" >&2
  exit 2
fi
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "OUTPUT_DIR is required" >&2
  exit 2
fi

mapfile -t MATCHES < <(compgen -G "$RUN_GLOB" | sort)
if [[ ${#MATCHES[@]} -eq 0 ]]; then
  echo "No run directory matches: $RUN_GLOB" >&2
  exit 3
fi
RUN_DIR="${MATCHES[-1]}"

mkdir -p "$OUTPUT_DIR"

echo "=== Cross-eval job start ==="
echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-unset}"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "RUN_DIR: $RUN_DIR"
echo "EVAL_DATASET: $EVAL_DATASET"
echo "Python: $($PYTHON --version 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo ""

$PYTHON src/evaluation/cross_layout_eval.py \
  --model-dir "$RUN_DIR" \
  --eval-dataset "$EVAL_DATASET" \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Cross-eval job complete ==="
