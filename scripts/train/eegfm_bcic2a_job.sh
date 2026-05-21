#!/usr/bin/env bash
# SLURM job body for EEG-FM-Bench BCIC-2a baseline reproduction runs.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/erimoldi/openclaw_remote/projects/eeg-spatial-paper}"
PYTHON="${PYTHON:-/home/erimoldi/.conda/envs/sparse-hate/bin/python}"

: "${CONF_FILE:?CONF_FILE must point to an EEG-FM-Bench YAML config}"
: "${MODEL_TYPE:?MODEL_TYPE must be set}"

cd "$PROJECT_ROOT"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/external/EEG-FM-Bench:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/erimoldi/.conda/envs/sparse-hate/lib:/home/erimoldi/.conda/envs/sparse-hate/lib/python3.11/site-packages/nvidia/cu13/lib:/home/erimoldi/.conda/envs/sparse-hate/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

export EEGFM_CONF_ROOT="$PROJECT_ROOT/external/EEG-FM-Bench/assets/conf"
export EEGFM_RUN_ROOT="$PROJECT_ROOT/results/eegfm_bench"
export EEGFM_LOG_ROOT="$PROJECT_ROOT/results/eegfm_bench/log"
export EEGFM_DATABASE_RAW_ROOT="$PROJECT_ROOT/data/eegfm/raw"
export EEGFM_DATABASE_PROC_ROOT="$PROJECT_ROOT/data/eegfm/processed"
export EEGFM_DATABASE_CACHE_ROOT="$PROJECT_ROOT/data/eegfm/cache"

export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 40000))}"

mkdir -p "$EEGFM_RUN_ROOT" "$EEGFM_LOG_ROOT"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "MODEL_TYPE=$MODEL_TYPE"
echo "CONF_FILE=$CONF_FILE"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

srun --ntasks=1 "$PYTHON" external/EEG-FM-Bench/baseline_main.py \
  "conf_file=$CONF_FILE" \
  "model_type=$MODEL_TYPE"
