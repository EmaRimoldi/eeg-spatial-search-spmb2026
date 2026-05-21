#!/usr/bin/env bash
# Cancel eval jobs (column 4) from the latest cross-generalization manifest.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_ROOT/results/slurm_logs"
MANIFEST="${1:-}"

if [[ -z "$MANIFEST" ]]; then
  MANIFEST="$(ls -1t "$LOG_DIR"/cross_generalization_full_*.tsv 2>/dev/null | head -1 || true)"
fi

if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
  printf 'No cross-generalization manifest found in %s\n' "$LOG_DIR" >&2
  exit 1
fi

printf 'Using manifest: %s\n' "$MANIFEST"
count=0
while IFS=$'\t' read -r variant seed train_job_id eval_job_id; do
  if [[ "$variant" == "variant" || -z "$eval_job_id" ]]; then
    continue
  fi
  if squeue -h -j "$eval_job_id" >/dev/null 2>&1; then
    scancel "$eval_job_id"
    printf '[cancelled] %s seed=%s eval=%s\n' "$variant" "$seed" "$eval_job_id"
    count=$((count + 1))
  fi
done < "$MANIFEST"

printf 'Cancelled %d eval jobs from manifest.\n' "$count"
