#!/usr/bin/env bash
# Clone all external repositories required for the project.
# Run once during project setup.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
EXTERNAL_DIR="$PROJECT_ROOT/external"

echo "Cloning external repositories into $EXTERNAL_DIR"

clone_or_update() {
    local url="$1"
    local name="$2"
    local dir="$EXTERNAL_DIR/$name"

    if [ -d "$dir/.git" ]; then
        echo "  $name: already exists, skipping"
    else
        echo "  Cloning $name..."
        git clone "$url" "$dir"
    fi
}

clone_or_update "https://github.com/xw1216/EEG-FM-Bench.git" "EEG-FM-Bench"
clone_or_update "https://github.com/elouayas/reve_eeg.git" "reve_eeg"
clone_or_update "https://github.com/935963004/LaBraM.git" "LaBraM"
clone_or_update "https://github.com/wjq-learning/CBraMod.git" "CBraMod"
clone_or_update "https://github.com/ycq091044/BIOT.git" "BIOT"

echo ""
echo "Recording commit hashes..."
bash "$SCRIPT_DIR/freeze_commits.sh"

echo ""
echo "Done. External repos cloned."
echo "Next: bash env/install.sh"
