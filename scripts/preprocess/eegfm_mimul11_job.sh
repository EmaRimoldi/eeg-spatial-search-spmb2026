#!/usr/bin/env bash
# SLURM job body for EEG-FM-Bench Mimul-11 preparation + preprocessing.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/erimoldi/openclaw_remote/projects/eeg-spatial-paper}"
PYTHON="${PYTHON:-/home/erimoldi/.conda/envs/sparse-hate/bin/python}"
MIMUL_URL="${MIMUL_URL:-https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100788/RawData.tar.gz}"

RAW_ROOT="$PROJECT_ROOT/data/eegfm/raw/Multimodal 11 intuitive movement"
ARCHIVE_PATH="$RAW_ROOT/RawData.tar.gz"
EXTRACTED_ROOT="$RAW_ROOT/RawData"

cd "$PROJECT_ROOT"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/external/EEG-FM-Bench:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/erimoldi/.conda/envs/sparse-hate/lib:${LD_LIBRARY_PATH:-}"

export EEGFM_CONF_ROOT="$PROJECT_ROOT/external/EEG-FM-Bench/assets/conf"
export EEGFM_RUN_ROOT="$PROJECT_ROOT/results/eegfm_bench"
export EEGFM_LOG_ROOT="$PROJECT_ROOT/results/eegfm_bench/log"
export EEGFM_DATABASE_RAW_ROOT="$PROJECT_ROOT/data/eegfm/raw"
export EEGFM_DATABASE_PROC_ROOT="$PROJECT_ROOT/data/eegfm/processed"
export EEGFM_DATABASE_CACHE_ROOT="$PROJECT_ROOT/data/eegfm/cache"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "$RAW_ROOT" "$EEGFM_RUN_ROOT" "$EEGFM_LOG_ROOT"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "MIMUL_URL=$MIMUL_URL"
echo "ARCHIVE_PATH=$ARCHIVE_PATH"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"

if [[ ! -s "$ARCHIVE_PATH" ]]; then
  echo "Downloading Mimul-11 RawData archive..."
else
  echo "Archive already present; resuming/validating download..."
fi
wget --continue --output-document="$ARCHIVE_PATH" "$MIMUL_URL"

if [[ ! -d "$EXTRACTED_ROOT" ]] || [[ -z "$(find "$EXTRACTED_ROOT" -type f -name '*.vhdr' -print -quit 2>/dev/null)" ]]; then
  echo "Extracting RawData archive..."
  tar -xzf "$ARCHIVE_PATH" -C "$RAW_ROOT"
else
  echo "Extracted RawData tree already present; skipping tar extraction."
fi

"$PYTHON" scripts/preprocess/fix_mimul11_brainvision_headers.py \
  --root "$EXTRACTED_ROOT"

"$PYTHON" external/EEG-FM-Bench/preproc.py \
  conf_file=configs/eegfm_bench/mimul11_preproc.yaml
