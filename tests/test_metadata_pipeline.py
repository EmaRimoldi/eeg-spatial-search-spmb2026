"""
Unit tests for the metadata pipeline.

Tests:
- Channel name normalization (case, synonyms, bipolar)
- Coordinate lookup (known/unknown channels)
- Metadata CSV validation
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def test_canonical_mapping_basic():
    """Basic normalization cases."""
    from src.data.channel_name_mapping import normalize_channel_name

    test_cases = [
        ("Fp1", "Fp1", 1.0),
        ("FP1", "Fp1", 1.0),  # case normalization
        ("fp1", "Fp1", 1.0),  # lowercase
        ("T3", "T7", 1.0),    # old alias
        ("T4", "T8", 1.0),    # old alias
        ("T5", "P7", 1.0),    # old alias
        ("T6", "P8", 1.0),    # old alias
    ]

    for raw, expected, min_conf in test_cases:
        result = normalize_channel_name(raw)
        assert result.canonical_name == expected, (
            f"normalize({raw!r}): expected {expected!r}, got {result.canonical_name!r}"
        )
        assert result.confidence >= min_conf, (
            f"normalize({raw!r}): confidence {result.confidence} < {min_conf}"
        )

    print("  basic mapping — OK")


def test_canonical_mapping_tuab_style():
    """TUAB-style 'EEG Fp1-REF' channel names."""
    from src.data.channel_name_mapping import normalize_with_dataset_override

    test_cases = [
        ("EEG Fp1-REF", "tuab", "Fp1"),
        ("EEG T3-REF", "tuab", "T7"),   # T3 → T7
        ("EEG T5-REF", "tuab", "P7"),   # T5 → P7
        ("EEG Cz-REF", "tuab", "Cz"),
    ]

    for raw, dataset, expected in test_cases:
        result = normalize_with_dataset_override(raw, dataset)
        assert result.canonical_name == expected, (
            f"normalize({raw!r}, {dataset}): expected {expected!r}, got {result.canonical_name!r}"
        )

    print("  TUAB-style mapping — OK")


def test_unknown_channel_not_mapped():
    """Truly unknown channels should return None canonical name."""
    from src.data.channel_name_mapping import normalize_channel_name

    result = normalize_channel_name("GIBBERISH_CHANNEL_XYZ")
    assert result.canonical_name is None, "Unknown channel should not be mapped"
    assert result.confidence == 0.0
    assert not result.is_known

    print("  unknown channel handling — OK")


def test_coordinate_lookup_known():
    """Known channels should have coordinates."""
    from src.data.coordinate_lookup import lookup_coordinates, get_coords_3d, get_coords_2d

    for ch in ["Fp1", "Cz", "O1", "T7", "Fz"]:
        c = lookup_coordinates(ch)
        assert c is not None, f"{ch} should have coordinates"
        assert c.canonical_name == ch
        assert -1.5 <= c.x_3d <= 1.5
        assert -1.5 <= c.y_3d <= 1.5
        assert -1.5 <= c.z_3d <= 1.5

        xyz = get_coords_3d(ch)
        assert xyz is not None
        assert len(xyz) == 3

        xy = get_coords_2d(ch)
        assert xy is not None
        assert len(xy) == 2

    print("  coordinate lookup known channels — OK")


def test_coordinate_lookup_unknown():
    """Unknown channels should return None gracefully."""
    from src.data.coordinate_lookup import lookup_coordinates

    result = lookup_coordinates("TOTALLY_UNKNOWN_CHANNEL")
    assert result is None, "Unknown channel should return None"

    print("  coordinate lookup unknown — OK")


def test_coordinate_range_validity():
    """All stored coordinates should be within unit sphere (±1.2 tolerance)."""
    from src.data.coordinate_lookup import _COORD_TABLE

    for name, c in _COORD_TABLE.items():
        for dim, val in [("x", c.x_3d), ("y", c.y_3d), ("z", c.z_3d)]:
            assert abs(val) <= 1.2, (
                f"{name} {dim}_3d={val:.3f} is outside ±1.2 range"
            )

    print(f"  coordinate range validity — OK ({len(_COORD_TABLE)} channels)")


def test_metadata_csv_exists():
    """Check that metadata CSVs have been generated."""
    import os
    metadata_dir = os.path.join(PROJECT_ROOT, "data", "metadata")
    expected_files = [
        "canonical_1020_coordinates.csv",
        "moabb_channel_metadata.csv",
        "tuab_channel_metadata.csv",
    ]

    for fname in expected_files:
        path = os.path.join(metadata_dir, fname)
        assert os.path.exists(path), (
            f"Missing metadata file: {fname}. "
            f"Run: python src/data/build_metadata_moabb.py"
        )
        # Check it has content
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) > 1, f"{fname} appears to be empty"

    print(f"  metadata CSVs present — OK ({len(expected_files)} files)")


def test_metadata_validation_passes():
    """Validate all generated metadata CSVs."""
    from src.data.validate_channel_metadata import validate_all
    import os

    metadata_dir = os.path.join(PROJECT_ROOT, "data", "metadata")
    success = validate_all(metadata_dir)
    assert success, "Channel metadata validation failed — check warnings above"

    print("  metadata validation — OK")


def test_prepare_metadata_batch():
    """Test the prepare_metadata_batch utility."""
    import torch
    from src.models.spatial_embeddings.utils import prepare_metadata_batch

    ch_names = ["Fp1", "Fz", "Cz", "Pz", "O1"]
    B = 3
    meta = prepare_metadata_batch(ch_names, B, reference_type="referential", device="cpu")

    assert "channel_names" in meta
    assert "coords_2d" in meta
    assert "coords_3d" in meta
    assert "reference_meta" in meta

    assert meta["coords_2d"].shape == (len(ch_names), 2)
    assert meta["coords_3d"].shape == (len(ch_names), 3)
    assert len(meta["reference_meta"]) == len(ch_names)

    print("  prepare_metadata_batch — OK")


if __name__ == "__main__":
    print("=" * 50)
    print("Metadata Pipeline Tests")
    print("=" * 50)

    tests = [
        test_canonical_mapping_basic,
        test_canonical_mapping_tuab_style,
        test_unknown_channel_not_mapped,
        test_coordinate_lookup_known,
        test_coordinate_lookup_unknown,
        test_coordinate_range_validity,
        test_metadata_csv_exists,
        test_metadata_validation_passes,
        test_prepare_metadata_batch,
    ]

    failed = []
    for fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"  FAILED {fn.__name__}: {e}")
            failed.append(fn.__name__)

    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All metadata tests passed")
