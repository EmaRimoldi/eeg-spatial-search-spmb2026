"""
Preprocess MOABB datasets for use in spatial ablation experiments.

Downloads datasets via MOABB, applies standard preprocessing,
and saves processed tensors to data/processed/.

Requires: moabb, mne, braindecode

Usage:
    python scripts/preprocess/preprocess_moabb.py --dataset BNCI2014_001
    python scripts/preprocess/preprocess_moabb.py --dataset PhysionetMI
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def preprocess_bnci2014_001(output_dir: Path):
    """
    Preprocess BNCI2014_001 (BCI Competition IV Dataset 2a).

    22 channels, 4-class motor imagery, 250 Hz.
    9 subjects.
    """
    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
        import numpy as np
    except ImportError:
        print("ERROR: moabb not installed. Run: pip install moabb")
        return

    print("Loading BNCI2014_001...")
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4)

    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, metadata = paradigm.get_data(
        dataset=dataset,
        subjects=dataset.subject_list,
    )

    # Save as numpy arrays
    np.save(output_dir / "bnci2014_001_X.npy", X)
    np.save(output_dir / "bnci2014_001_y.npy", y)
    metadata.to_csv(output_dir / "bnci2014_001_metadata.csv", index=False)

    print(f"Saved: X={X.shape}, y={y.shape}")
    print(f"Output dir: {output_dir}")

    # Save split definitions
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    subjects = metadata["subject"].unique()
    n = len(subjects)
    train_subs = subjects[:int(0.7 * n)].tolist()
    val_subs = subjects[int(0.7 * n):int(0.85 * n)].tolist()
    test_subs = subjects[int(0.85 * n):].tolist()

    import json
    split = {"train": train_subs, "val": val_subs, "test": test_subs}
    with open(SPLITS_DIR / "bnci2014_001_subject_split.json", "w") as f:
        json.dump(split, f, indent=2)
    print(f"Saved split: train={len(train_subs)}, val={len(val_subs)}, test={len(test_subs)}")


def preprocess_physionetmi(output_dir: Path):
    """
    Preprocess PhysionetMI dataset.

    64 channels, 2/4-class motor imagery, 160 Hz.
    109 subjects (use subset).
    """
    try:
        from moabb.datasets import PhysionetMI
        from moabb.paradigms import MotorImagery
        import numpy as np
    except ImportError:
        print("ERROR: moabb not installed. Run: pip install moabb")
        return

    print("Loading PhysionetMI (first 20 subjects)...")
    dataset = PhysionetMI()
    paradigm = MotorImagery(n_classes=4)

    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, metadata = paradigm.get_data(
        dataset=dataset,
        subjects=dataset.subject_list[:20],  # First 20 subjects for efficiency
    )

    np.save(output_dir / "physionetmi_X.npy", X)
    np.save(output_dir / "physionetmi_y.npy", y)
    metadata.to_csv(output_dir / "physionetmi_metadata.csv", index=False)

    print(f"Saved: X={X.shape}, y={y.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["BNCI2014_001", "PhysionetMI", "all"])
    parser.add_argument("--output-dir", type=str, default=str(PROCESSED_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.dataset in ("BNCI2014_001", "all"):
        preprocess_bnci2014_001(output_dir)

    if args.dataset in ("PhysionetMI", "all"):
        preprocess_physionetmi(output_dir)

    print("\nPreprocessing complete.")
    print("Next: python src/data/build_metadata_moabb.py")


if __name__ == "__main__":
    main()
