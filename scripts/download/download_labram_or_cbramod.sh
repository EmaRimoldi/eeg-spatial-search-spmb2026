#!/usr/bin/env bash
# Download LaBraM checkpoint.
# Note: LaBraM checkpoint may be included in the git repo via Git LFS.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LABRAM_DIR="$PROJECT_ROOT/external/LaBraM"
CHECKPOINT="$LABRAM_DIR/checkpoints/labram-base.pth"

echo "=============================="
echo "LaBraM Checkpoint Setup"
echo "=============================="

if [ -f "$CHECKPOINT" ]; then
    SIZE=$(stat -c%s "$CHECKPOINT" 2>/dev/null || stat -f%z "$CHECKPOINT" 2>/dev/null)
    if [ "$SIZE" -lt 10000 ]; then
        echo "labram-base.pth is a Git LFS pointer ($SIZE bytes)."
        echo "Attempting git lfs pull..."
        cd "$LABRAM_DIR"
        if git lfs pull; then
            echo "Git LFS pull successful."
        else
            echo "Git LFS pull failed. You may need to install git-lfs:"
            echo "  https://git-lfs.com/"
            echo ""
            echo "Alternatively, download manually from:"
            echo "  https://github.com/935963004/LaBraM/releases"
        fi
    else
        echo "labram-base.pth already downloaded ($SIZE bytes). OK."
    fi
else
    echo "Checkpoint not found at: $CHECKPOINT"
    echo "Run from the repo root:"
    echo "  cd external/LaBraM && git lfs pull"
fi

echo ""
echo "Done."
