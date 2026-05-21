#!/bin/bash
# SLURM job template for warming MOABB caches without wasting GPU slots.
# Requires env vars:
#   CACHE_DATASET   dataset name
#   CACHE_SPLITS    space-separated splits, e.g. "test" or "train val test"
#   CACHE_BATCH_SIZE optional batch size (default 32)

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/erimoldi/openclaw_remote/projects/eeg-spatial-paper}"
export PATH="/home/erimoldi/.conda/envs/sparse-hate/bin:$PATH"
export PYTHONNOUSERSITE=1
PYTHON="/home/erimoldi/.conda/envs/sparse-hate/bin/python"

cd "$PROJECT_ROOT"

if [[ -z "${CACHE_DATASET:-}" ]]; then
  echo "CACHE_DATASET is required" >&2
  exit 2
fi
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-32}"
CACHE_SPLITS="${CACHE_SPLITS:-test}"

echo "=== Cache warm job start ==="
echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-unset}"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "CACHE_DATASET: $CACHE_DATASET"
echo "CACHE_SPLITS: $CACHE_SPLITS"
echo "Python: $($PYTHON --version 2>&1)"
echo ""

# shellcheck disable=SC2206
SPLIT_ARGS=( $CACHE_SPLITS )
$PYTHON scripts/eval/warm_moabb_cache.py --dataset "$CACHE_DATASET" --batch-size "$CACHE_BATCH_SIZE" --splits "${SPLIT_ARGS[@]}"

echo ""
echo "=== Cache warm job complete ==="
