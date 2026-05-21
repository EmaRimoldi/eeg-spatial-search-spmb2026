#!/usr/bin/env python3
"""Launch the BNCI2014_001 multi-backbone spatial pilot."""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "bnci2014_001_multibackbone_spatial_pilot.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    return parser.parse_args()


def flatten_model_kwargs(backbone: str, cfg: dict) -> dict:
    return dict(cfg.get("model_kwargs", {}).get(backbone, {}))


def write_run_config(
    base_cfg: dict,
    backbone: str,
    variant: str,
    seed: int,
    out_dir: Path,
    dry_run: bool = False,
) -> Path:
    train_cfg = base_cfg.get("training", {})
    model_kwargs = flatten_model_kwargs(backbone, base_cfg)
    run_cfg = {
        "backbone": backbone,
        "spatial_variant": variant,
        "freeze_policy": base_cfg.get("freeze_policy", "head_only"),
        "dataset": base_cfg.get("dataset", "BNCI2014_001"),
        "num_classes": int(base_cfg.get("num_classes", 4)),
        "seed": int(seed),
        "epochs": int(train_cfg.get("epochs", 50)),
        "batch_size": int(train_cfg.get("batch_size", 32)),
        "smoke_test": bool(train_cfg.get("smoke_test", False)),
        "allow_synthetic_fallback": bool(train_cfg.get("allow_synthetic_fallback", False)),
        "optimizer": {
            "name": "adamw",
            "lr": float(train_cfg.get("lr", 1.0e-4)),
            "weight_decay": float(train_cfg.get("weight_decay", 1.0e-4)),
        },
        "scheduler": {"name": "cosine", "warmup_epochs": 5},
        "early_stopping_patience": int(train_cfg.get("early_stopping_patience", 10)),
        **model_kwargs,
    }

    config_dir = out_dir / "generated_configs"
    path = config_dir / f"{backbone}_{variant}_seed{seed}.yaml"
    if dry_run:
        return path

    config_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(run_cfg, f, sort_keys=False)
    return path


def main() -> int:
    args = parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})
    output_dir = PROJECT_ROOT / train_cfg.get("output_dir", "results/pilot_bnci2014_001_multibackbone")
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    runs = list(itertools.product(cfg["backbones"], cfg["spatial_variants"], cfg["seeds"]))
    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    print(f"Pilot config: {args.config}")
    print(f"Runs: {len(runs)}")
    print(f"Output: {output_dir}")

    for idx, (backbone, variant, seed) in enumerate(runs, start=1):
        run_config = write_run_config(
            cfg,
            backbone,
            variant,
            int(seed),
            output_dir,
            dry_run=args.dry_run,
        )
        cmd = [
            args.python,
            "src/training/train.py",
            "--config",
            str(run_config),
            "--output-dir",
            str(output_dir),
        ]
        print(f"[{idx:03d}/{len(runs):03d}] {backbone} {variant} seed={seed}")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
