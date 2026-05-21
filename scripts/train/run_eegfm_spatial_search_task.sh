#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/erimoldi/openclaw_remote/projects/eeg-spatial-paper}"
MANIFEST="${MANIFEST:?MANIFEST is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
PYTHON="${PYTHON:-/home/erimoldi/.conda/envs/sparse-hate/bin/python}"
STATUS_DIR="${STATUS_DIR:-$PROJECT_ROOT/results/spatial_search/status}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

line_no=$((TASK_ID + 2))
line="$(sed -n "${line_no}p" "$MANIFEST")"
if [[ -z "$line" ]]; then
  echo "No manifest row for task ${TASK_ID} (line ${line_no})"
  exit 1
fi

IFS=$'\t' read -r run_id stage dataset dataset_slug backbone variant seed seeds hparams_id hparams_json config output_dir job_name status notes <<< "$line"
mkdir -p "$STATUS_DIR"

write_status() {
  local status_value="$1"
  local exit_code="${2:-0}"
  "$PYTHON" - "$STATUS_DIR" "$run_id" "$status_value" "$exit_code" "$TASK_ID" "$config" <<'PY'
import datetime
import json
import sys
from pathlib import Path

status_dir, run_id, status, exit_code, task_id, config = sys.argv[1:]
payload = {
    "run_id": run_id,
    "status": status,
    "exit_code": int(exit_code),
    "task_id": int(task_id),
    "config": config,
    "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
}
Path(status_dir).mkdir(parents=True, exist_ok=True)
(Path(status_dir) / f"{run_id}.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
}

on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    write_status failed "$rc" || true
  fi
}
trap on_exit EXIT

echo "TASK_ID=$TASK_ID run_id=$run_id stage=$stage dataset=$dataset backbone=$backbone variant=$variant seed=$seed"
echo "config=$config"

write_status running 0

export PATH="/home/erimoldi/.conda/envs/sparse-hate/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/erimoldi/.conda/envs/sparse-hate/lib/python3.11/site-packages/nvidia/cu13/lib:/home/erimoldi/.conda/envs/sparse-hate/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_ROOT"
"$PYTHON" src/training/train.py --config "$config" --output-dir "$OUT_ROOT"

write_status completed 0
