#!/bin/bash
# SLURM job template for EEG spatial ablation experiments.
# Invoked by submission wrappers with TRAIN_ARGS in the environment.
#SBATCH --partition=pi_tpoggio
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/erimoldi/openclaw_remote/projects/eeg-spatial-paper}"

# Activate conda env by path (no conda init needed)
export PATH="/home/erimoldi/.conda/envs/sparse-hate/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/erimoldi/.conda/envs/sparse-hate/lib/python3.11/site-packages/nvidia/cu13/lib:/home/erimoldi/.conda/envs/sparse-hate/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

PYTHON="/home/erimoldi/.conda/envs/sparse-hate/bin/python"

cd "$PROJECT_ROOT"

echo "=== Job start ==="
echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-unset}"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "Python: $($PYTHON --version 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo ""

echo "Running: $PYTHON src/training/train.py ${TRAIN_ARGS}"
$PYTHON src/training/train.py ${TRAIN_ARGS}

echo ""
echo "=== Job complete ==="
