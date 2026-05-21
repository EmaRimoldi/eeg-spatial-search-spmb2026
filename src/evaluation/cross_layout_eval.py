"""
Cross-layout evaluation.

Loads a model trained on one EEG montage (e.g. BNCI2014_001, 22ch)
and evaluates on a different montage (e.g. PhysionetMI, 64ch).

This is the central generalization experiment:
- Spatial variants that encode 3D coordinates should generalize because
  electrode coordinates are informative even for unseen channel layouts.
- Spatial variants with no coordinates or channel-ID lookup will have
  no or degraded transfer signal.

Usage:
    python src/evaluation/cross_layout_eval.py \
        --model-dir results/core_ablation/<run>/ \
        --eval-dataset PhysionetMI \
        --output-dir results/cross_layout/
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def shared_label_space(train_dataset: str, eval_dataset: str):
    """Return a shared semantic label space between train and eval datasets."""
    from src.data.preprocessing import DATASET_CONFIGS

    train_cfg = DATASET_CONFIGS[train_dataset]
    eval_cfg = DATASET_CONFIGS[eval_dataset]
    train_order = list(train_cfg.get("label_order", train_cfg["event_id"].keys()))
    eval_order = list(eval_cfg.get("label_order", eval_cfg["event_id"].keys()))

    shared = [name for name in train_order if name in eval_order]
    if not shared:
        raise ValueError(f"No shared labels between {train_dataset} and {eval_dataset}")

    train_indices = [train_order.index(name) for name in shared]
    eval_indices = [eval_order.index(name) for name in shared]
    return shared, train_indices, eval_indices


def parse_args():
    from src.data.preprocessing import DATASET_CONFIGS

    p = argparse.ArgumentParser(description="Cross-layout evaluation")
    p.add_argument("--model-dir", required=True, help="Path to a completed run directory")
    p.add_argument("--eval-dataset", default="PhysionetMI",
                   choices=sorted(DATASET_CONFIGS))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output-dir", default="results/cross_layout")
    p.add_argument("--num-classes", type=int, default=4)
    return p.parse_args()


def load_checkpoint(run_dir: Path, device: torch.device):
    """Load model from a completed run directory."""
    # Read config
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config at {config_path}")

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Instantiate model
    from src.models.wrappers.registry import build_model
    extra_kwargs = {
        k: v for k, v in config.items()
        if k not in {"backbone", "spatial_variant", "num_classes", "freeze_policy", "checkpoint"}
    }
    model = build_model(
        backbone=config.get("backbone", "reve"),
        spatial_variant=config.get("spatial_variant", "coords3d"),
        num_classes=config.get("num_classes", 4),
        freeze_policy=config.get("freeze_policy", "head_only"),
        checkpoint_path=config.get("checkpoint"),
        **extra_kwargs,
    )

    # Load weights
    ckpt_path = run_dir / "best_model.pt"
    if ckpt_path.exists():
        state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        print(f"Loaded weights from {ckpt_path}")
    else:
        print(f"WARNING: No checkpoint at {ckpt_path}; using model as-is")

    model.to(device)
    model.eval()
    return model, config


def evaluate(model, loader, metadata, device, train_label_indices, eval_label_indices) -> dict:
    """Compute accuracy and balanced accuracy on a dataset."""
    from sklearn.metrics import balanced_accuracy_score, accuracy_score
    import numpy as np

    all_preds, all_labels = [], []
    eval_remap = {idx: i for i, idx in enumerate(eval_label_indices)}
    dropped = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x, metadata)
            logits = logits[:, train_label_indices]
            labels = y.numpy()

            mask = np.isin(labels, eval_label_indices)
            dropped += int((~mask).sum())
            if not np.any(mask):
                continue

            labels = np.array([eval_remap[int(v)] for v in labels[mask]], dtype=np.int64)
            preds = logits[mask].argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if len(all_labels) == 0:
        raise ValueError("No eval samples remained after shared-label filtering")

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(all_labels, all_preds)),
        "n_samples": len(all_labels),
        "n_classes": len(train_label_indices),
        "n_dropped": dropped,
    }


def main():
    args = parse_args()
    run_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model from {run_dir} ...")
    model, config = load_checkpoint(run_dir, device)

    print(f"Loading eval dataset: {args.eval_dataset} ...")
    from src.data.preprocessing import load_moabb_dataset
    _, _, test_loader, metadata = load_moabb_dataset(
        args.eval_dataset,
        batch_size=args.batch_size,
        config={"num_classes": args.num_classes},
        split_names=("test",),
    )

    # Move metadata tensors to device
    for k, v in metadata.items():
        if isinstance(v, torch.Tensor):
            metadata[k] = v.to(device)

    train_dataset = config.get("dataset", "unknown")
    shared_labels, train_label_indices, eval_label_indices = shared_label_space(
        train_dataset,
        args.eval_dataset,
    )
    print(f"Shared label space: {shared_labels}")

    print(f"Evaluating {config.get('spatial_variant')} on {args.eval_dataset} ...")
    metrics = evaluate(model, test_loader, metadata, device, train_label_indices, eval_label_indices)

    result = {
        "train_dataset": train_dataset,
        "eval_dataset": args.eval_dataset,
        "backbone": config.get("backbone", "reve"),
        "spatial_variant": config.get("spatial_variant", "unknown"),
        "freeze_policy": config.get("freeze_policy", "unknown"),
        "seed": config.get("seed", 0),
        "run_dir": str(run_dir),
        "shared_labels": shared_labels,
        **metrics,
    }

    # Save results
    run_name = run_dir.name
    out_path = output_dir / f"{run_name}_cross_{args.eval_dataset}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Cross-Layout Results ===")
    print(f"Train:   {result['train_dataset']}")
    print(f"Test:    {result['eval_dataset']}")
    print(f"Variant: {result['spatial_variant']}")
    print(f"BAccuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"N samples: {metrics['n_samples']}")
    print(f"Saved to: {out_path}")

    return result


if __name__ == "__main__":
    main()
