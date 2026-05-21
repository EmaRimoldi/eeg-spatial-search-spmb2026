#!/usr/bin/env bash
set -euo pipefail
while true; do
  out=$(sbatch --array=64-134%3 \
    --partition=mit_normal_gpu,mit_preemptable,ou_bcs_low,ou_bcs_normal,pi_tpoggio \
    --time=06:00:00 --cpus-per-task=8 --mem=32G --gres=gpu:1 \
    --job-name=spatial_eegfm_array \
    --output=/home/erimoldi/openclaw_remote/projects/eeg-spatial-paper/results/eegfm_spatial_coords/slurm_logs/spatial_eegfm_array_%A_%a.out \
    --error=/home/erimoldi/openclaw_remote/projects/eeg-spatial-paper/results/eegfm_spatial_coords/slurm_logs/spatial_eegfm_array_%A_%a.err \
    --export=ALL,PROJECT_ROOT=/home/erimoldi/openclaw_remote/projects/eeg-spatial-paper,MANIFEST=/home/erimoldi/openclaw_remote/projects/eeg-spatial-paper/results/eegfm_spatial_coords/manifest.tsv,OUT_ROOT=/home/erimoldi/openclaw_remote/projects/eeg-spatial-paper/results/eegfm_spatial_coords,PYTHON=/home/erimoldi/.conda/envs/sparse-hate/bin/python \
    /home/erimoldi/openclaw_remote/projects/eeg-spatial-paper/scripts/train/run_eegfm_spatial_manifest_task.sh 2>&1) && {
      echo "$out"
      exit 0
    }
  echo "[$(date -Is)] submit blocked, retrying in 300s: $out"
  sleep 300
done
