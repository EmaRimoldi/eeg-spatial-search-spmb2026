#!/usr/bin/env bash
# Cancel queued/running jobs from the latest priority-core16 submission manifest.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_ROOT/results/slurm_logs"
MANIFEST="${1:-}"

if [[ -z "$MANIFEST" ]]; then
  MANIFEST="$(ls -1t "$LOG_DIR"/priority_core16_submissions_*.tsv 2>/dev/null | head -1 || true)"
fi

if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
  printf 'No priority_core16 manifest found in %s\n' "$LOG_DIR" >&2
  exit 1
fi

printf 'Using manifest: %s\n' "$MANIFEST"

count=0
while IFS=$'\t' read -r priority job_name variant policy seed job_id; do
  if [[ "$priority" == "priority" || -z "$job_id" ]]; then
    continue
  fi
  if squeue -h -j "$job_id" >/dev/null 2>&1; then
    scancel "$job_id"
    printf '[cancelled] %s (%s)\n' "$job_name" "$job_id"
    count=$((count + 1))
  fi
done < "$MANIFEST"

printf 'Cancelled %d active jobs from manifest.\n' "$count"
