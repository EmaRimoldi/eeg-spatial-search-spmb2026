"""Smoke test: verify all core dependencies are importable."""
import sys


def test_torch():
    import torch
    assert torch.__version__


def test_numpy():
    import numpy as np
    assert np.__version__


def test_scipy():
    import scipy
    assert scipy.__version__


def test_pandas():
    import pandas as pd
    assert pd.__version__


def test_sklearn():
    import sklearn
    assert sklearn.__version__


def test_mne():
    import mne
    assert mne.__version__


def test_braindecode():
    import braindecode
    assert braindecode.__version__


def test_moabb():
    import moabb
    assert moabb.__version__


def test_einops():
    import einops
    assert einops.__version__


def test_hydra():
    import hydra
    import omegaconf
    assert hydra.__version__
    assert omegaconf.__version__


def test_matplotlib():
    import matplotlib
    import seaborn
    assert matplotlib.__version__
    assert seaborn.__version__


def test_transformers():
    import transformers
    assert transformers.__version__


def test_timm():
    import timm
    assert timm.__version__


def test_project_src():
    """Verify that the project's src package is importable."""
    import os
    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    assert os.path.isdir(src_path), "src/ directory not found"
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import src  # noqa: F401


if __name__ == "__main__":
    print("=" * 60)
    print("EEG Spatial Paper — Import Smoke Test")
    print("=" * 60)

    tests = [
        ("torch", test_torch),
        ("numpy", test_numpy),
        ("scipy", test_scipy),
        ("pandas", test_pandas),
        ("scikit-learn", test_sklearn),
        ("mne", test_mne),
        ("braindecode", test_braindecode),
        ("moabb", test_moabb),
        ("einops", test_einops),
        ("hydra", test_hydra),
        ("matplotlib/seaborn", test_matplotlib),
        ("transformers", test_transformers),
        ("timm", test_timm),
        ("project src", test_project_src),
    ]

    results = {}
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            results[name] = True
            print(f"  OK")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All imports OK")
        sys.exit(0)
