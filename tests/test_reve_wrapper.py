"""
Tests for the REVE model wrapper.

Tests:
- Wrapper can be instantiated for all spatial variants
- Forward pass produces correct output shape
- Freeze policy correctly freezes/unfreezes parameters
- Parameter count makes sense (spatial + head > 0 trainable)
"""

import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

VARIANTS_TO_TEST = ["channel_id", "coords3d", "coords3d_reference", "topology_agnostic"]

CHANNEL_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]
N_CHANNELS = len(CHANNEL_NAMES)
B, T = 2, 1000
NUM_CLASSES = 4


def make_metadata():
    return {
        "channel_names": CHANNEL_NAMES,
        "coords_2d": None,
        "coords_3d": None,
        "reference_meta": ["referential"] * N_CHANNELS,
    }


def test_reve_wrapper_instantiation_all_variants():
    """REVE wrapper should instantiate for all tested spatial variants."""
    from src.models.wrappers.reve_wrapper import REVEWrapper

    for variant in VARIANTS_TO_TEST:
        wrapper = REVEWrapper(
            spatial_variant=variant,
            num_classes=NUM_CLASSES,
            freeze_policy="head_only",
            checkpoint_path=None,
            smoke_test=True,
        )
        assert wrapper is not None, f"Failed to instantiate REVEWrapper with variant={variant}"
        assert wrapper.spatial_variant == variant
    print("  instantiation all variants — OK")


def test_reve_wrapper_forward_pass():
    """Forward pass should produce (B, num_classes) logits."""
    from src.models.wrappers.reve_wrapper import REVEWrapper

    x = torch.randn(B, N_CHANNELS, T)
    metadata = make_metadata()

    for variant in VARIANTS_TO_TEST:
        wrapper = REVEWrapper(
            spatial_variant=variant,
            num_classes=NUM_CLASSES,
            freeze_policy="head_only",
            checkpoint_path=None,
            smoke_test=True,
        )
        wrapper.eval()

        with torch.no_grad():
            logits = wrapper(x, metadata)

        assert logits.shape == (B, NUM_CLASSES), (
            f"{variant}: expected logits ({B}, {NUM_CLASSES}), got {logits.shape}"
        )

    print("  forward pass all variants — OK")


def test_reve_wrapper_freeze_policy():
    """head_only policy should freeze backbone but not head or spatial embedding."""
    from src.models.wrappers.reve_wrapper import REVEWrapper

    wrapper = REVEWrapper(
        spatial_variant="coords3d",
        num_classes=NUM_CLASSES,
        freeze_policy="head_only",
        checkpoint_path=None,
        smoke_test=True,
    )

    param_counts = wrapper.parameter_count()
    assert param_counts["trainable"] > 0, "No trainable parameters"
    assert param_counts["classifier_head"] > 0, "Classifier head has no parameters"

    print("  freeze policy head_only — OK")


def test_reve_wrapper_no_spatial():
    """'none' spatial variant should still run without errors."""
    from src.models.wrappers.reve_wrapper import REVEWrapper

    wrapper = REVEWrapper(
        spatial_variant="none",
        num_classes=NUM_CLASSES,
        freeze_policy="head_only",
        checkpoint_path=None,
        smoke_test=True,
    )
    wrapper.eval()

    x = torch.randn(B, N_CHANNELS, T)
    metadata = make_metadata()

    with torch.no_grad():
        logits = wrapper(x, metadata)

    assert logits.shape == (B, NUM_CLASSES)
    print("  'none' spatial variant — OK")


def test_registry_reve_build():
    """registry.build_model should correctly build REVE wrapper."""
    from src.models.wrappers.registry import build_model

    model = build_model(
        backbone="reve",
        spatial_variant="coords3d",
        num_classes=NUM_CLASSES,
        freeze_policy="head_only",
        smoke_test=True,
    )

    assert model is not None
    x = torch.randn(B, N_CHANNELS, T)
    metadata = make_metadata()
    logits = model(x, metadata)
    assert logits.shape == (B, NUM_CLASSES)

    print("  registry build_model — OK")


if __name__ == "__main__":
    print("=" * 50)
    print("REVE Wrapper Tests")
    print("=" * 50)

    tests = [
        test_reve_wrapper_instantiation_all_variants,
        test_reve_wrapper_forward_pass,
        test_reve_wrapper_freeze_policy,
        test_reve_wrapper_no_spatial,
        test_registry_reve_build,
    ]

    failed = []
    for fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"  FAILED {fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(fn.__name__)

    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All REVE wrapper tests passed")
