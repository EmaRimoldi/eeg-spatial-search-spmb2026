"""
Main training script for spatial ablation experiments.

Supports all spatial variants, freeze policies, and training regimes.
Every run saves a complete reproducibility record.

Usage:
    python src/training/train.py \
        --config configs/experiments/pilot.yaml \
        --backbone reve \
        --spatial-variant coords3d \
        --freeze-policy head_only \
        --dataset BNCI2014_001 \
        --num-classes 4 \
        --seed 42 \
        --output-dir results/logs/
"""

import argparse
import json
import os
import re
import sys
import time
import hashlib
import platform
import datetime
import pathlib
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_git_state() -> str:
    """Capture current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_system_info() -> dict:
    """Capture system information for reproducibility."""
    info = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def make_run_dir(
    base_dir: str,
    dataset: str,
    backbone: str,
    spatial_variant: str,
    freeze_policy: str,
    seed: int,
    transfer_regime: str = "same_layout",
    run_id: str | None = None,
) -> Path:
    """Create a run directory with descriptive naming convention."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
    if run_id:
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id)).strip("_")
        run_name = f"{safe_run_id}_{timestamp}"
    else:
        run_name = (
            f"{dataset}_{backbone}_{spatial_variant}_{freeze_policy}"
            f"_seed{seed}_{transfer_regime}_{timestamp}"
        )
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_record(
    run_dir: Path,
    config: dict,
    metrics: dict,
    channel_metadata_snapshot: Optional[str] = None,
):
    """Save all reproducibility artifacts for a run."""
    # Save resolved config
    with open(run_dir / "config_resolved.yaml", "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)

    # Save metrics
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save git state
    with open(run_dir / "git_state.txt", "w") as f:
        f.write(get_git_state())

    # Save system info
    with open(run_dir / "system_info.json", "w") as f:
        json.dump(get_system_info(), f, indent=2)

    # Save channel metadata snapshot
    if channel_metadata_snapshot:
        with open(run_dir / "channel_metadata_snapshot.csv", "w") as f:
            f.write(channel_metadata_snapshot)


class SpatialAblationTrainer:
    """
    Training engine for spatial ablation experiments.

    Handles:
    - Model instantiation from registry
    - Data loading (MOABB, TUH, or custom)
    - Training loop with early stopping
    - Evaluation
    - Reproducibility record-keeping
    """

    def __init__(self, config: dict, run_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] Run dir: {run_dir}")

    def build_model(self) -> nn.Module:
        """Instantiate model from registry using config."""
        from src.models.wrappers.registry import build_model
        cfg = self.config

        model_kwargs = dict(cfg.get("model_kwargs") or {})
        if isinstance(model_kwargs.get(cfg["backbone"]), dict):
            model_kwargs = dict(model_kwargs[cfg["backbone"]])
        wrapper_kwargs = dict(cfg.get("wrapper_kwargs") or {})

        extra_kwargs = {
            k: v for k, v in cfg.items()
            if k not in {
                "backbone",
                "spatial_variant",
                "num_classes",
                "freeze_policy",
                "checkpoint",
                "model_kwargs",
                "wrapper_kwargs",
                "search",
            }
        }
        extra_kwargs.update(model_kwargs)
        extra_kwargs.update(wrapper_kwargs)
        model = build_model(
            backbone=cfg["backbone"],
            spatial_variant=cfg["spatial_variant"],
            num_classes=cfg["num_classes"],
            freeze_policy=cfg.get("freeze_policy", "head_only"),
            checkpoint_path=cfg.get("checkpoint"),
            **extra_kwargs,
        )
        return model.to(self.device)

    def build_optimizer(self, model: nn.Module):
        """Build optimizer from config."""
        opt_cfg = self.config.get("optimizer", {})
        name = opt_cfg.get("name", "adamw")
        lr = opt_cfg.get("lr", 1e-4)
        wd = opt_cfg.get("weight_decay", 1e-4)

        backbone_lr = opt_cfg.get("backbone_lr")
        head_lr = opt_cfg.get("head_lr")
        spatial_lr = opt_cfg.get("spatial_lr")

        trainable_params = [p for p in model.parameters() if p.requires_grad]

        param_groups = None
        if any(v is not None for v in (backbone_lr, head_lr, spatial_lr)):
            seen: set[int] = set()

            def collect(params):
                group = []
                for p in params:
                    if not p.requires_grad:
                        continue
                    pid = id(p)
                    if pid in seen:
                        continue
                    seen.add(pid)
                    group.append(p)
                return group

            param_groups = []

            if hasattr(model, "backbone"):
                params = collect(model.backbone.parameters())
                if params:
                    param_groups.append({"params": params, "lr": backbone_lr or lr, "weight_decay": wd})

            if hasattr(model, "classifier_head"):
                params = collect(model.classifier_head.parameters())
                if params:
                    param_groups.append({"params": params, "lr": head_lr or lr, "weight_decay": wd})

            spatial_params = []
            for attr in (
                "spatial_adapter",
                "spatial_embedding",
                "spatial_norm",
                "native_pair_norm",
                "native_pair_q",
                "native_pair_k",
                "graph_layers",
            ):
                module = getattr(model, attr, None)
                if module is not None:
                    spatial_params.extend(collect(module.parameters()))

            for attr in ("spatial_gain", "dist_bias_gain", "spatial_token_gain", "spatial_pos_gain"):
                param = getattr(model, attr, None)
                if isinstance(param, nn.Parameter) and param.requires_grad:
                    pid = id(param)
                    if pid not in seen:
                        seen.add(pid)
                        spatial_params.append(param)

            for attr in ("graph_gains", "native_attn_gains"):
                param_list = getattr(model, attr, None)
                if param_list is None:
                    continue
                for param in param_list:
                    if isinstance(param, nn.Parameter) and param.requires_grad:
                        pid = id(param)
                        if pid not in seen:
                            seen.add(pid)
                            spatial_params.append(param)

            if spatial_params:
                param_groups.append({"params": spatial_params, "lr": spatial_lr or lr, "weight_decay": wd})

            leftover = collect(model.parameters())
            if leftover:
                param_groups.append({"params": leftover, "lr": lr, "weight_decay": wd})

        if name == "adamw":
            return optim.AdamW(param_groups or trainable_params, lr=lr, weight_decay=wd)
        elif name == "adam":
            return optim.Adam(param_groups or trainable_params, lr=lr)
        elif name == "sgd":
            return optim.SGD(param_groups or trainable_params, lr=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {name}")

    def build_scheduler(self, optimizer, n_epochs: int):
        """Build learning rate scheduler."""
        sched_cfg = self.config.get("scheduler", {})
        name = sched_cfg.get("name", "cosine")
        warmup = sched_cfg.get("warmup_epochs", 5)

        if name == "cosine":
            t_max = max(1, n_epochs - warmup)
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        elif name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        else:
            return None

    def load_data(self):
        """
        Load dataset specified in config.

        Returns (train_loader, val_loader, test_loader, channel_metadata).
        Falls back to synthetic data only when explicitly enabled.
        """
        dataset_name = self.config.get("dataset", "synthetic")
        batch_size = self.config.get("batch_size", 32)
        allow_synthetic_fallback = bool(self.config.get("allow_synthetic_fallback", False))

        # Try to load from EEG-FM-Bench processed splits
        if dataset_name in (
            "bcic_2a", "BCIC2a", "bcic2a", "BNCI2014_001",
            "motor_mv_img", "PhysioMI", "PhysionetMI", "physiomi",
            "workload", "Workload",
        ):
            try:
                return self._load_eegfm_data(dataset_name, batch_size)
            except Exception as e:
                print(f"WARNING: Could not load {dataset_name} from EEG-FM-Bench processed splits: {e}")
                if not allow_synthetic_fallback:
                    raise
                print("Falling back to synthetic data for testing")

        # Try to load from MOABB
        if dataset_name in ("BNCI2014_001", "BNCI2014_004", "PhysionetMI", "Cho2017", "Shin2017A"):
            try:
                return self._load_moabb_data(dataset_name, batch_size)
            except Exception as e:
                print(f"WARNING: Could not load {dataset_name} from MOABB: {e}")
                if not allow_synthetic_fallback:
                    raise
                print("Falling back to synthetic data for testing")

        # Fallback: synthetic data
        return self._load_synthetic_data(batch_size)

    def _load_moabb_data(self, dataset_name: str, batch_size: int):
        """Load data from MOABB."""
        from src.data.preprocessing import load_moabb_dataset
        return load_moabb_dataset(
            dataset_name, batch_size, self.config,
            label_fraction=self.config.get("label_fraction", 1.0),
        )

    def _load_eegfm_data(self, dataset_name: str, batch_size: int):
        """Load data from EEG-FM-Bench processed local splits."""
        from src.data.eegfm_loader import load_eegfm_dataset

        return load_eegfm_dataset(
            dataset_name,
            batch_size,
            num_workers=int(self.config.get("num_workers", 0)),
        )

    def _load_synthetic_data(self, batch_size: int):
        """Create synthetic data with real BNCI2014-001 channel geometry."""
        from torch.utils.data import TensorDataset, DataLoader

        n_classes = self.config.get("num_classes", 4)
        ch_names = [
            "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
            "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
            "CP3", "CP1", "CPz", "CP2", "CP4",
            "P1",  "Pz",  "P2",  "POz",
        ]
        n_channels = len(ch_names)
        T = 1000

        n_train, n_val, n_test = 200, 50, 50

        def make_ds(n):
            X = torch.randn(n, n_channels, T)
            y = torch.randint(0, n_classes, (n,))
            return TensorDataset(X, y)

        train_loader = DataLoader(make_ds(n_train), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(make_ds(n_val),   batch_size=batch_size)
        test_loader  = DataLoader(make_ds(n_test),  batch_size=batch_size)

        # Use real MNE coordinates if available, else fallback to lookup table
        try:
            from src.data.preprocessing import _get_mne_coords_3d, _project_to_2d
            coords_3d = _get_mne_coords_3d(ch_names)
            coords_2d = _project_to_2d(coords_3d)
            coords_3d = torch.tensor(coords_3d, dtype=torch.float32)
            coords_2d = torch.tensor(coords_2d, dtype=torch.float32)
        except Exception:
            coords_3d = coords_2d = None

        metadata = {
            "channel_names": ch_names,
            "coords_3d": coords_3d,
            "coords_2d": coords_2d,
            "reference_meta": ["average"] * n_channels,
        }

        print(f"[Trainer] Using synthetic data: {n_train} train, {n_val} val, {n_test} test")
        return train_loader, val_loader, test_loader, metadata

    def _apply_channel_dropout(self, x: torch.Tensor, p: float) -> torch.Tensor:
        """Zero-out each channel independently with probability p (per batch)."""
        if p <= 0.0:
            return x
        B, C, T = x.shape
        mask = (torch.rand(B, C, 1, device=x.device) > p).float()
        return x * mask

    def train_epoch(
        self,
        model: nn.Module,
        loader,
        optimizer,
        metadata: dict,
        criterion: nn.Module,
    ) -> dict:
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        ch_dropout_p = self.config.get("channel_dropout", 0.0)

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            if ch_dropout_p > 0.0:
                batch_x = self._apply_channel_dropout(batch_x, ch_dropout_p)

            optimizer.zero_grad()
            logits = model(batch_x, metadata)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += len(batch_y)

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
        }

    @torch.no_grad()
    def eval_epoch(
        self,
        model: nn.Module,
        loader,
        metadata: dict,
        criterion: nn.Module,
    ) -> dict:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            logits = model(batch_x, metadata)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())

        acc = correct / max(total, 1)

        # Balanced accuracy
        from collections import Counter
        label_counts = Counter(all_labels)
        per_class_acc = {}
        for cls in label_counts:
            cls_mask = [l == cls for l in all_labels]
            cls_preds = [p for p, m in zip(all_preds, cls_mask) if m]
            cls_labels = [l for l, m in zip(all_labels, cls_mask) if m]
            per_class_acc[cls] = sum(p == l for p, l in zip(cls_preds, cls_labels)) / len(cls_labels)
        balanced_acc = sum(per_class_acc.values()) / len(per_class_acc) if per_class_acc else 0.0

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
        }

    def run(self) -> dict:
        """Full training run. Returns final metrics dict."""
        torch.manual_seed(self.config.get("seed", 42))

        # Build model
        model = self.build_model()
        optimizer = self.build_optimizer(model)

        n_epochs = self.config.get("epochs", 50)
        scheduler = self.build_scheduler(optimizer, n_epochs)
        criterion = nn.CrossEntropyLoss()

        # Load data
        train_loader, val_loader, test_loader, metadata = self.load_data()

        # Move metadata tensors to device
        for k, v in metadata.items():
            if isinstance(v, torch.Tensor):
                metadata[k] = v.to(self.device)

        # Training loop with early stopping
        patience = self.config.get("early_stopping_patience", 10)
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        history = []

        print(f"[Trainer] Starting training: {n_epochs} epochs, patience={patience}")

        for epoch in range(n_epochs):
            train_metrics = self.train_epoch(model, train_loader, optimizer, metadata, criterion)
            val_metrics = self.eval_epoch(model, val_loader, metadata, criterion)

            if scheduler is not None:
                scheduler.step()

            history.append({
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
            })

            val_acc = val_metrics["balanced_accuracy"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                # Save best model to /tmp (node-local, no quota) or run_dir
                _ckpt = (
                    self.run_dir / "best_model.pt"
                    if self.config.get("save_checkpoint", False)
                    else pathlib.Path(f"/tmp/best_model_{os.getpid()}.pt")
                )
                torch.save(model.state_dict(), _ckpt)
                self._tmp_ckpt = _ckpt
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch+1:3d}/{n_epochs}: "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"val_acc={val_metrics['accuracy']:.4f} "
                    f"val_bacc={val_metrics['balanced_accuracy']:.4f}"
                )

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best: epoch {best_epoch})")
                break

        # Load best model weights for test eval
        _ckpt = getattr(self, "_tmp_ckpt", self.run_dir / "best_model.pt")
        if _ckpt.exists():
            model.load_state_dict(torch.load(_ckpt, map_location=self.device))
            # Remove /tmp checkpoint after loading (run_dir checkpoint kept if requested)
            if not self.config.get("save_checkpoint", False) and _ckpt.parent == pathlib.Path("/tmp"):
                _ckpt.unlink(missing_ok=True)

        test_metrics = self.eval_epoch(model, test_loader, metadata, criterion)

        final_metrics = {
            "test": test_metrics,
            "best_val": {"balanced_accuracy": best_val_acc, "epoch": best_epoch},
            "config": self.config,
            "history": history,
        }

        return final_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Spatial ablation training")
    parser.add_argument("--backbone", type=str, default="reve")
    parser.add_argument("--spatial-variant", type=str, default="coords3d")
    parser.add_argument("--freeze-policy", type=str, default="head_only")
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="results/logs")
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Persist best_model.pt inside the run directory (needed for transfer eval).",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Allow lightweight stub backbones and synthetic fallback for smoke tests only.",
    )
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Use synthetic data if the requested real dataset cannot be loaded.",
    )
    # Few-shot: fraction of training labels to use
    parser.add_argument("--label-fraction", type=float, default=1.0,
                        help="Fraction of training labels (0..1]. 1.0 = full dataset.")
    # Channel dropout: fraction of channels to randomly zero at training time
    parser.add_argument("--channel-dropout", type=float, default=0.0,
                        help="Prob of zeroing each channel per batch (0=disabled).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build config from args (or load from YAML if specified)
    config = {
        "backbone": args.backbone,
        "spatial_variant": args.spatial_variant,
        "freeze_policy": args.freeze_policy,
        "dataset": args.dataset,
        "num_classes": args.num_classes,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "label_fraction": args.label_fraction,
        "channel_dropout": args.channel_dropout,
        "save_checkpoint": args.save_checkpoint,
        "smoke_test": args.smoke_test,
        "allow_synthetic_fallback": args.allow_synthetic_fallback or args.smoke_test,
        "optimizer": {"name": "adamw", "lr": args.lr, "weight_decay": 1e-4},
        "scheduler": {"name": "cosine", "warmup_epochs": 5},
        "early_stopping_patience": 10,
    }

    if args.config:
        import yaml
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)
        config.update(yaml_config)

    search_cfg = config.get("search") if isinstance(config.get("search"), dict) else {}

    # Create run directory
    run_dir = make_run_dir(
        base_dir=args.output_dir,
        dataset=config["dataset"],
        backbone=config["backbone"],
        spatial_variant=config["spatial_variant"],
        freeze_policy=config["freeze_policy"],
        seed=config["seed"],
        run_id=search_cfg.get("run_id"),
    )

    print(f"Run directory: {run_dir}")

    # Train
    trainer = SpatialAblationTrainer(config, run_dir)
    metrics = trainer.run()

    # Save all reproducibility artifacts
    save_run_record(run_dir, config, metrics)

    print(f"\nFinal test metrics:")
    for k, v in metrics["test"].items():
        print(f"  {k}: {v:.4f}")
    print(f"\nRun saved to: {run_dir}")
    return metrics


if __name__ == "__main__":
    main()
