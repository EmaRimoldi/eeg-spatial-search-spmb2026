#!/usr/bin/env bash
# SLURM job body for EEG-FM-Bench PhysioMI preprocessing.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/erimoldi/openclaw_remote/projects/eeg-spatial-paper}"
PYTHON="${PYTHON:-/home/erimoldi/.conda/envs/sparse-hate/bin/python}"
DOWNLOAD_JOBS="${DOWNLOAD_JOBS:-4}"

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

mkdir -p "$EEGFM_RUN_ROOT" "$EEGFM_LOG_ROOT"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DOWNLOAD_JOBS=$DOWNLOAD_JOBS"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"

"$PYTHON" scripts/preprocess/download_physionetmi_to_eegfm.py \
  --subjects all \
  --jobs "$DOWNLOAD_JOBS"

"$PYTHON" external/EEG-FM-Bench/preproc.py \
  conf_file=configs/eegfm_bench/physiomi_preproc.yaml
