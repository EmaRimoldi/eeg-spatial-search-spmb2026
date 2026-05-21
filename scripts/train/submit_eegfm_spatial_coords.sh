#!/usr/bin/env bash
# Submit EEG-FM-Bench spatial-coordinate experiments that reuse the saved
# baseline preprocessing and seeds. Baselines are NOT recomputed here; only
# coords2d / coords3d / coords3d_distbias variants are launched.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="$PROJECT_ROOT/results/eegfm_spatial_coords"
GEN_DIR="$OUT_ROOT/generated_configs"
LOG_DIR="$OUT_ROOT/slurm_logs"
MANIFEST="$OUT_ROOT/manifest.tsv"
SUBMITTED="$OUT_ROOT/submitted_jobs.tsv"
PYTHON="/home/erimoldi/.conda/envs/sparse-hate/bin/python"
PARTITIONS="${PARTITIONS:-mit_normal_gpu,mit_preemptable,ou_bcs_low,ou_bcs_normal,pi_tpoggio}"

mkdir -p "$GEN_DIR" "$LOG_DIR"

"$PYTHON" - <<'PY'
from pathlib import Path
import yaml

PROJECT_ROOT = Path('/home/erimoldi/openclaw_remote/projects/eeg-spatial-paper')
OUT_ROOT = PROJECT_ROOT / 'results' / 'eegfm_spatial_coords'
GEN_DIR = OUT_ROOT / 'generated_configs'

datasets = {
    'bcic2a': {'dataset': 'bcic_2a', 'num_classes': 4, 'n_channels': 22, 'eeg_size': 800},
    'physiomi': {'dataset': 'motor_mv_img', 'num_classes': 4, 'n_channels': 64, 'eeg_size': 800},
    'workload': {'dataset': 'workload', 'num_classes': 2, 'n_channels': 19, 'eeg_size': 800},
}
backbones = ['biot', 'labram', 'cbramod']
variants = ['coords2d', 'coords3d', 'coords3d_distbias']
seeds = [42, 43, 44, 45, 46]

rows = []
for dataset_key, meta in datasets.items():
    for backbone in backbones:
        src = PROJECT_ROOT / 'configs' / 'eegfm_bench' / f'{dataset_key}_{backbone}_table2.yaml'
        base = yaml.safe_load(src.read_text())
        for variant in variants:
            for seed in seeds:
                cfg = {
                    'backbone': backbone,
                    'spatial_variant': variant,
                    'freeze_policy': 'full',
                    'dataset': meta['dataset'],
                    'num_classes': meta['num_classes'],
                    'seed': seed,
                    'epochs': int(base['training']['max_epochs']),
                    'batch_size': int(base['data']['batch_size']),
                    'num_workers': int(base['data'].get('num_workers', 4)),
                    'allow_synthetic_fallback': False,
                    'smoke_test': False,
                    'checkpoint': base['model']['pretrained_path'],
                    'optimizer': {
                        'name': 'adamw',
                        'lr': float(base['training']['max_lr']),
                        'weight_decay': float(base['training']['weight_decay']),
                        'backbone_lr': float(base['training']['max_lr']) * float(base['training'].get('encoder_lr_scale', 1.0)),
                        'head_lr': float(base['training']['max_lr']),
                        'spatial_lr': float(base['training']['max_lr']),
                    },
                    'scheduler': {
                        'name': 'cosine',
                        'warmup_epochs': int(base['training'].get('warmup_epochs', 5)),
                    },
                    'early_stopping_patience': 1000,
                }

                if backbone == 'biot':
                    cfg.update({
                        'embed_dim': int(base['model'].get('emb_size', 256)),
                        'heads': int(base['model'].get('heads', 8)),
                        'depth': int(base['model'].get('depth', 4)),
                        'max_channels': int(max(meta['n_channels'], 64)),
                        'n_fft': int(base['model'].get('n_fft', 200)),
                        'hop_length': int(base['model'].get('hop_length', 100)),
                    })
                elif backbone == 'labram':
                    cfg.update({
                        'eeg_size': int(meta['eeg_size']),
                        'patch_size': int(base['model'].get('patch_size', 200)),
                        'spatial_pos_mode': 'residual',
                    })
                elif backbone == 'cbramod':
                    cfg.update({
                        'n_channels': int(meta['n_channels']),
                        'eeg_size': int(meta['eeg_size']),
                        'classifier_head': base['model']['classifier_head'],
                    })

                dst = GEN_DIR / f'{dataset_key}_{backbone}_{variant}_seed{seed}.yaml'
                dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
                rows.append((dataset_key, backbone, variant, seed, str(dst), meta['dataset']))

manifest = OUT_ROOT / 'manifest.tsv'
with manifest.open('w') as fh:
    fh.write('dataset_key\tbackbone\tvariant\tseed\tconfig\tdataset\n')
    for row in rows:
        fh.write('\t'.join(map(str, row)) + '\n')

print(manifest)
print(len(rows))
PY

printf 'jobid\tdataset_key\tbackbone\tvariant\tseed\tconfig\tdataset\n' > "$SUBMITTED"

while IFS=$'\t' read -r dataset_key backbone variant seed config dataset; do
  [[ "$dataset_key" == "dataset_key" ]] && continue
  job="sp_${dataset_key}_${backbone}_${variant}_s${seed}"
  wrap="export PATH=/home/erimoldi/.conda/envs/sparse-hate/bin:\$PATH; export PYTHONNOUSERSITE=1; export PYTHONUNBUFFERED=1; export LD_LIBRARY_PATH=/home/erimoldi/.conda/envs/sparse-hate/lib/python3.11/site-packages/nvidia/cu13/lib:/home/erimoldi/.conda/envs/sparse-hate/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:\${LD_LIBRARY_PATH:-}; cd $PROJECT_ROOT; $PYTHON src/training/train.py --config $config --output-dir $OUT_ROOT"
  sub=$(sbatch \
    --partition="$PARTITIONS" \
    --time=06:00:00 \
    --cpus-per-task=8 \
    --mem=32G \
    --gres=gpu:1 \
    --job-name="$job" \
    --output="$LOG_DIR/${job}_%j.out" \
    --error="$LOG_DIR/${job}_%j.err" \
    --wrap="$wrap")
  jobid=$(awk '/Submitted batch job/ {print $4}' <<< "$sub")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$jobid" "$dataset_key" "$backbone" "$variant" "$seed" "$config" "$dataset" >> "$SUBMITTED"
  echo "$sub | $dataset_key $backbone $variant seed=$seed"
  sleep 0.1
done < "$MANIFEST"

echo "submitted_manifest=$MANIFEST"
echo "submitted_jobs=$SUBMITTED"
