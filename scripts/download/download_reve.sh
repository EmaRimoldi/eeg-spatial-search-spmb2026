#!/usr/bin/env bash
# Download REVE checkpoint from HuggingFace.
#
# brain-bzh/reve-base is a GATED repository.
# Before running this script you must:
#   1. Visit https://huggingface.co/brain-bzh/reve-base and request access
#   2. Log in: huggingface-cli login   (paste your HF token when prompted)
#
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_DIR="$PROJECT_ROOT/checkpoints/reve"
POS_DIR="$CKPT_DIR/positions"
MANIFEST="$PROJECT_ROOT/data/metadata/checkpoint_manifest.csv"

PYTHON="${PYTHON:-python}"
# Use sparse-hate env if available
if [ -x "/home/erimoldi/.conda/envs/sparse-hate/bin/python" ]; then
    PYTHON="/home/erimoldi/.conda/envs/sparse-hate/bin/python"
fi

echo "=== REVE checkpoint download ==="
echo "Destination: $CKPT_DIR"

# Check HF authentication
if ! "$PYTHON" -c "from huggingface_hub import whoami; whoami()" 2>/dev/null; then
    echo ""
    echo "ERROR: Not logged in to HuggingFace."
    echo "Run: huggingface-cli login"
    echo "Then request access at: https://huggingface.co/brain-bzh/reve-base"
    exit 1
fi

echo "HuggingFace user: $("$PYTHON" -c "from huggingface_hub import whoami; print(whoami()['name'])")"

mkdir -p "$CKPT_DIR"
mkdir -p "$POS_DIR"
export CKPT_DIR POS_DIR

"$PYTHON" - <<'EOF'
import os, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

dest_dir = Path(os.environ.get("CKPT_DIR", "checkpoints/reve"))
dest_dir.mkdir(parents=True, exist_ok=True)
pos_dir = Path(os.environ.get("POS_DIR", dest_dir / "positions"))
pos_dir.mkdir(parents=True, exist_ok=True)

print("Downloading model.safetensors from brain-bzh/reve-base ...")
path = hf_hub_download(
    repo_id="brain-bzh/reve-base",
    filename="model.safetensors",
    local_dir=str(dest_dir),
)
size_mb = os.path.getsize(path) / 1e6
print(f"Downloaded to: {path}  ({size_mb:.0f} MB)")

print("Downloading position bank from brain-bzh/reve-positions ...")
pos_path = hf_hub_download(
    repo_id="brain-bzh/reve-positions",
    filename="model.safetensors",
    local_dir=str(pos_dir),
)
pos_size_kb = os.path.getsize(pos_path) / 1e3
print(f"Downloaded to: {pos_path}  ({pos_size_kb:.0f} kB)")
EOF

# Record provenance in checkpoint manifest
DOWNLOAD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
if [ ! -f "$MANIFEST" ]; then
    echo "model_name,source_url,local_path,sha256,license,date_downloaded,notes" > "$MANIFEST"
fi

SIZE=$(stat -c%s "$CKPT_DIR/model.safetensors" 2>/dev/null || echo "0")
echo "reve-base,https://huggingface.co/brain-bzh/reve-base,$CKPT_DIR/model.safetensors,,CC-BY-NC-4.0,$DOWNLOAD_DATE,size_bytes=$SIZE" >> "$MANIFEST"
POS_SIZE=$(stat -c%s "$POS_DIR/model.safetensors" 2>/dev/null || echo "0")
echo "reve-positions,https://huggingface.co/brain-bzh/reve-positions,$POS_DIR/model.safetensors,,CC-BY-NC-4.0,$DOWNLOAD_DATE,size_bytes=$POS_SIZE" >> "$MANIFEST"

echo ""
echo "=== Download complete ==="
echo "Manifest updated: $MANIFEST"
