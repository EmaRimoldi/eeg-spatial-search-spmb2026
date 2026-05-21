#!/usr/bin/env bash
# Widen already-submitted priority-core16 pending jobs to multiple partitions.
# Avoid ou_bcs_* because the project handoff reports NFS/home issues there.
# Default target set uses partitions compatible with the 6h job template.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_ROOT/results/slurm_logs"
TARGET_PARTITIONS="${TARGET_PARTITIONS:-mit_normal_gpu,mit_preemptable,pi_tpoggio}"
MANIFEST="${1:-}"

if [[ -z "$MANIFEST" ]]; then
  MANIFEST="$(ls -1t "$LOG_DIR"/priority_core16_submissions_*.tsv 2>/dev/null | head -1 || true)"
fi

if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
  printf 'No priority_core16 manifest found in %s\n' "$LOG_DIR" >&2
  exit 1
fi

printf 'Using manifest: %s\n' "$MANIFEST"
printf 'Target partitions: %s\n\n' "$TARGET_PARTITIONS"

updated=0
skipped=0
failed=0

while IFS=$'\t' read -r priority job_name variant policy seed job_id; do
  if [[ "$priority" == "priority" ]]; then
    continue
  fi

  if [[ -z "$job_id" ]]; then
    printf '[skip] %s has empty job id\n' "$job_name"
    skipped=$((skipped + 1))
    continue
  fi

  if ! squeue -h -j "$job_id" >/dev/null 2>&1; then
    printf '[skip] %s (%s) no longer in queue\n' "$job_name" "$job_id"
    skipped=$((skipped + 1))
    continue
  fi

  state="$(squeue -h -j "$job_id" -o '%T' | head -1 || true)"
  if [[ "$state" != "PENDING" ]]; then
    printf '[skip] %s (%s) state=%s\n' "$job_name" "$job_id" "$state"
    skipped=$((skipped + 1))
    continue
  fi

  if out="$(scontrol update JobId="$job_id" Partition="$TARGET_PARTITIONS" 2>&1)"; then
    part="$(scontrol show job "$job_id" | sed -n 's/.* Partition=\([^ ]*\).*/\1/p' | head -1)"
    reason="$(squeue -h -j "$job_id" -o '%R' | head -1 || true)"
    printf '[ok]   %s (%s) -> partition=%s reason=%s\n' "$job_name" "$job_id" "${part:-unknown}" "${reason:-unknown}"
    updated=$((updated + 1))
  else
    printf '[fail] %s (%s): %s\n' "$job_name" "$job_id" "$out" >&2
    failed=$((failed + 1))
  fi
done < "$MANIFEST"

printf '\nSummary: updated=%d skipped=%d failed=%d\n' "$updated" "$skipped" "$failed"
