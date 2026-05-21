#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/erimoldi/openclaw_remote/projects/eeg-spatial-paper}"
MANIFEST="${MANIFEST:?MANIFEST is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
PYTHON="${PYTHON:-/home/erimoldi/.conda/envs/sparse-hate/bin/python}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

line_no=$((TASK_ID + 2))
line="$(sed -n "${line_no}p" "$MANIFEST")"
if [[ -z "$line" ]]; then
  echo "No manifest row for task ${TASK_ID} (line ${line_no})"
  exit 1
fi

IFS=$'\t' read -r dataset_key backbone variant seed config dataset <<< "$line"

echo "TASK_ID=$TASK_ID dataset_key=$dataset_key backbone=$backbone variant=$variant seed=$seed config=$config dataset=$dataset"

export PATH="/home/erimoldi/.conda/envs/sparse-hate/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/erimoldi/.conda/envs/sparse-hate/lib/python3.11/site-packages/nvidia/cu13/lib:/home/erimoldi/.conda/envs/sparse-hate/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_ROOT"
"$PYTHON" src/training/train.py --config "$config" --output-dir "$OUT_ROOT"
