#!/usr/bin/env bash
# Record exact commit hashes for all external repos.
# Run after cloning or updating external repos.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
EXTERNAL_DIR="$PROJECT_ROOT/external"
OUTPUT_FILE="$PROJECT_ROOT/docs/notes/external_repo_commits.txt"

echo "# External Repository Commit Hashes" > "$OUTPUT_FILE"
echo "# Recorded: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUTPUT_FILE"
echo "# Purpose: Reproducibility — exact versions used in this project" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

for repo in EEG-FM-Bench reve_eeg LaBraM CBraMod BIOT; do
    dir="$EXTERNAL_DIR/$repo"
    if [ -d "$dir/.git" ]; then
        hash=$(git -C "$dir" rev-parse HEAD)
        echo "$repo $hash" >> "$OUTPUT_FILE"
        echo "  $repo: $hash"
    else
        echo "  $repo: NOT FOUND (skipping)"
        echo "$repo NOT_CLONED" >> "$OUTPUT_FILE"
    fi
done

echo "" >> "$OUTPUT_FILE"
echo "# To restore exact state: git -C external/<repo> checkout <hash>" >> "$OUTPUT_FILE"

echo ""
echo "Commits recorded to: $OUTPUT_FILE"
