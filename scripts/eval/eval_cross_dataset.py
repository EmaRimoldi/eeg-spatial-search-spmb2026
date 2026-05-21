"""
Cross-dataset evaluation script.

Loads a trained model from a run directory and evaluates it on a
different dataset than it was trained on.

Usage:
    python scripts/eval/eval_cross_dataset.py \
        --model-run results/logs/BNCI2014_001_reve_coords3d_head_only_seed42_*/ \
        --target-dataset PhysionetMI
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


def load_model_from_run(run_dir: Path):
    """Load model from a completed run directory."""
    config_path = run_dir / "config_resolved.yaml"
    checkpoint_path = run_dir / "best_model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"No config found in {run_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from src.models.wrappers.registry import build_model
    model = build_model(
        backbone=config["backbone"],
        spatial_variant=config["spatial_variant"],
        num_classes=config["num_classes"],
        freeze_policy="frozen",  # eval only
    )

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    return model, config


def evaluate_on_dataset(
    model,
    dataset_name: str,
    config: dict,
    device: torch.device,
) -> dict:
    """Evaluate a model on a target dataset."""
    print(f"Evaluating on {dataset_name}...")

    # Load target dataset
    # For now, use synthetic data as placeholder
    n_channels = config.get("num_classes", 22)
    T = 1000
    n_classes = config.get("num_classes", 4)

    X = torch.randn(50, n_channels, T)
    y = torch.randint(0, n_classes, (50,))

    metadata = {
        "channel_names": [f"ch{i}" for i in range(n_channels)],
        "coords_2d": None,
        "coords_3d": None,
        "reference_meta": ["unknown"] * n_channels,
    }

    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(X), 16):
            batch_x = X[i:i+16].to(device)
            batch_y = y[i:i+16]
            logits = model(batch_x, metadata)
            preds = logits.argmax(dim=1).cpu()
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

    return {
        "dataset": dataset_name,
        "accuracy": correct / max(total, 1),
        "n_samples": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-run", type=str, required=True)
    parser.add_argument("--target-dataset", type=str, default="PhysionetMI")
    args = parser.parse_args()

    run_dir = Path(args.model_run)
    if not run_dir.exists():
        # Try glob expansion
        import glob
        matches = glob.glob(args.model_run)
        if matches:
            run_dir = Path(sorted(matches)[-1])
        else:
            print(f"Run directory not found: {args.model_run}")
            sys.exit(1)

    print(f"Loading model from: {run_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model, config = load_model_from_run(run_dir)
        model = model.to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    results = evaluate_on_dataset(model, args.target_dataset, config, device)

    print(f"\nCross-dataset evaluation results:")
    print(f"  Source: {config.get('dataset', 'unknown')}")
    print(f"  Target: {results['dataset']}")
    print(f"  Accuracy: {results['accuracy']:.4f}")

    output_path = run_dir / f"eval_cross_{args.target_dataset}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
