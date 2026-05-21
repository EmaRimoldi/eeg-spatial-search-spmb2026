"""
Unit tests for spatial embedding modules.

Tests:
- Shape consistency: all variants produce (B, C, embed_dim) or None
- Deterministic behavior in eval mode (same input → same output)
- Graceful handling of missing coordinates (no crash, produces valid output)
- Correct handling of the 'none' variant (returns None)
- Factory function (build_spatial_embedding)
"""

import sys
import os
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch

EMBED_DIM = 64
BATCH_SIZE = 2
N_CHANNELS = 13

CHANNEL_NAMES = [
    "Fp1", "F3", "Fz", "F4", "Fp2",
    "C3", "Cz", "C4",
    "P3", "Pz", "P4",
    "O1", "O2",
]

ALL_VARIANTS = [
    "none",
    "channel_id",
    "coords2d",
    "coords3d",
    "coords3d_distbias",
    "coords3d_reference",
    "coords3d_rbf",
    "coords3d_geodesic_rbf",
    "topology_agnostic",
]


@pytest.fixture
def sample_coords_2d():
    return torch.randn(BATCH_SIZE, N_CHANNELS, 2)


@pytest.fixture
def sample_coords_3d():
    return torch.randn(BATCH_SIZE, N_CHANNELS, 3)


@pytest.fixture
def sample_reference_meta():
    return ["referential"] * N_CHANNELS


def build_embedding(variant: str) -> object:
    from src.models.spatial_embeddings.utils import build_spatial_embedding
    return build_spatial_embedding(variant, embed_dim=EMBED_DIM)


class TestShapeConsistency:
    """All non-None variants must produce (B, C, embed_dim)."""

    @pytest.mark.parametrize("variant", [v for v in ALL_VARIANTS if v != "none"])
    def test_output_shape(self, variant, sample_coords_2d, sample_coords_3d, sample_reference_meta):
        mod = build_embedding(variant)
        mod.eval()

        out = mod(
            channel_names=CHANNEL_NAMES,
            coords_2d=sample_coords_2d,
            coords_3d=sample_coords_3d,
            reference_meta=sample_reference_meta,
            batch_size=BATCH_SIZE,
            device=torch.device("cpu"),
        )

        assert out is not None, f"{variant} returned None unexpectedly"
        assert out.shape == (BATCH_SIZE, N_CHANNELS, EMBED_DIM), (
            f"{variant}: expected ({BATCH_SIZE}, {N_CHANNELS}, {EMBED_DIM}), got {out.shape}"
        )

    def test_none_variant_returns_none(self):
        mod = build_embedding("none")
        mod.eval()
        out = mod(
            channel_names=CHANNEL_NAMES,
            coords_2d=None,
            coords_3d=None,
            reference_meta=None,
            batch_size=BATCH_SIZE,
            device=torch.device("cpu"),
        )
        assert out is None, f"'none' variant should return None, got {type(out)}"


class TestDeterminism:
    """In eval mode, same input must produce same output."""

    @pytest.mark.parametrize("variant", [v for v in ALL_VARIANTS if v != "none"])
    def test_deterministic_eval(self, variant, sample_coords_2d, sample_coords_3d, sample_reference_meta):
        mod = build_embedding(variant)
        mod.eval()

        kwargs = dict(
            channel_names=CHANNEL_NAMES,
            coords_2d=sample_coords_2d,
            coords_3d=sample_coords_3d,
            reference_meta=sample_reference_meta,
            batch_size=BATCH_SIZE,
            device=torch.device("cpu"),
        )

        with torch.no_grad():
            out1 = mod(**kwargs)
            out2 = mod(**kwargs)

        assert torch.allclose(out1, out2), f"{variant}: non-deterministic output in eval mode"


class TestMissingCoordinates:
    """Variants should handle missing coords without crashing."""

    @pytest.mark.parametrize(
        "variant",
        [
            "coords2d",
            "coords3d",
            "coords3d_distbias",
            "coords3d_reference",
            "coords3d_rbf",
            "coords3d_geodesic_rbf",
        ],
    )
    def test_none_coords_fallback(self, variant):
        """When coords are None, module should fall back to coordinate lookup or zeros."""
        mod = build_embedding(variant)
        mod.eval()

        out = mod(
            channel_names=CHANNEL_NAMES,
            coords_2d=None,
            coords_3d=None,
            reference_meta=None,
            batch_size=BATCH_SIZE,
            device=torch.device("cpu"),
        )

        assert out is not None, f"{variant}: returned None when coords are missing"
        assert out.shape == (BATCH_SIZE, N_CHANNELS, EMBED_DIM), f"{variant}: wrong shape with missing coords"
        assert not torch.isnan(out).any(), f"{variant}: NaN in output with missing coords"

    def test_unknown_channels_handled(self):
        """Unknown channel names should not cause crashes."""
        from src.models.spatial_embeddings.coords3d import Coords3DEmbedding
        mod = Coords3DEmbedding(embed_dim=EMBED_DIM)
        mod.eval()

        unknown_names = ["Unknown1", "Unknown2", "Unknown3"]
        out = mod(
            channel_names=unknown_names,
            coords_2d=None,
            coords_3d=None,
            reference_meta=None,
            batch_size=BATCH_SIZE,
            device=torch.device("cpu"),
        )

        assert out is not None
        assert out.shape == (BATCH_SIZE, 3, EMBED_DIM)


class TestChannelIDVariant:
    """Specific tests for channel_id variant."""

    def test_known_channels_in_vocab(self):
        from src.models.spatial_embeddings.channel_id import ChannelIDEmbedding
        mod = ChannelIDEmbedding(embed_dim=EMBED_DIM)

        known = mod.known_channels()
        for ch in ["Fp1", "Cz", "O1", "T7"]:
            assert ch in known, f"{ch} should be in default vocab"

    def test_unknown_channel_uses_unk(self):
        from src.models.spatial_embeddings.channel_id import ChannelIDEmbedding
        mod = ChannelIDEmbedding(embed_dim=EMBED_DIM)
        mod.eval()

        # Mix known and unknown
        names = ["Fp1", "TOTALLY_UNKNOWN_CHANNEL", "Cz"]
        out = mod(
            channel_names=names,
            coords_2d=None, coords_3d=None, reference_meta=None,
            batch_size=1, device=torch.device("cpu"),
        )
        assert out.shape == (1, 3, EMBED_DIM)


class TestDistBiasVariant:
    """Test coords3d_distbias attention bias."""

    def test_attention_bias_shape(self):
        from src.models.spatial_embeddings.coords3d_distbias import Coords3DDistBiasEmbedding
        mod = Coords3DDistBiasEmbedding(embed_dim=EMBED_DIM)
        mod.eval()

        coords_3d = torch.randn(BATCH_SIZE, N_CHANNELS, 3)
        out = mod(
            channel_names=CHANNEL_NAMES,
            coords_2d=None,
            coords_3d=coords_3d,
            reference_meta=None,
            batch_size=BATCH_SIZE,
            device=torch.device("cpu"),
        )

        assert out.shape == (BATCH_SIZE, N_CHANNELS, EMBED_DIM)
        bias = mod.get_attention_bias()
        assert bias is not None
        assert bias.shape == (BATCH_SIZE, N_CHANNELS, N_CHANNELS)


class TestReferenceVariant:
    """Test coords3d_reference reference encoding."""

    def test_different_refs_produce_different_embeddings(self):
        from src.models.spatial_embeddings.coords3d_reference import Coords3DReferenceEmbedding
        mod = Coords3DReferenceEmbedding(embed_dim=EMBED_DIM)
        mod.eval()

        coords_3d = torch.zeros(1, 3, 3)  # same positions

        ref_a = ["referential", "referential", "referential"]
        ref_b = ["average", "average", "average"]

        out_a = mod(["Fp1", "Cz", "O1"], None, coords_3d, ref_a, 1, "cpu")
        out_b = mod(["Fp1", "Cz", "O1"], None, coords_3d, ref_b, 1, "cpu")

        assert not torch.allclose(out_a, out_b), (
            "Different reference types should produce different embeddings"
        )


class TestRBFVariants:
    """Test anchor-RBF coordinate variants."""

    @pytest.mark.parametrize("variant", ["coords3d_rbf", "coords3d_geodesic_rbf"])
    def test_rbf_kwargs_change_parameterization(self, variant):
        from src.models.spatial_embeddings.utils import build_spatial_embedding

        mod = build_spatial_embedding(
            variant,
            embed_dim=EMBED_DIM,
            n_rbf=4,
            anchor_mode="axes_diagonal",
            rbf_sigma=0.4,
        )
        mod.eval()

        out = mod(
            channel_names=CHANNEL_NAMES,
            coords_2d=None,
            coords_3d=torch.randn(BATCH_SIZE, N_CHANNELS, 3),
            reference_meta=None,
            batch_size=BATCH_SIZE,
            device=torch.device("cpu"),
        )

        assert out.shape == (BATCH_SIZE, N_CHANNELS, EMBED_DIM)
        assert torch.isfinite(out).all()


class TestFactory:
    """Test the build_spatial_embedding factory function."""

    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_build_all_variants(self, variant):
        from src.models.spatial_embeddings.utils import build_spatial_embedding
        mod = build_spatial_embedding(variant, embed_dim=EMBED_DIM)
        assert mod is not None
        assert mod.name == variant
        assert mod.embed_dim == EMBED_DIM

    def test_unknown_variant_raises(self):
        from src.models.spatial_embeddings.utils import build_spatial_embedding
        with pytest.raises(ValueError, match="Unknown spatial variant"):
            build_spatial_embedding("nonexistent_variant", embed_dim=EMBED_DIM)


if __name__ == "__main__":
    # Run without pytest for quick manual testing
    print("Running spatial embedding tests manually...")

    for variant in ALL_VARIANTS:
        try:
            from src.models.spatial_embeddings.utils import build_spatial_embedding
            mod = build_spatial_embedding(variant, embed_dim=EMBED_DIM)
            mod.eval()

            coords_3d = torch.randn(BATCH_SIZE, N_CHANNELS, 3)
            out = mod(
                channel_names=CHANNEL_NAMES,
                coords_2d=None,
                coords_3d=coords_3d,
                reference_meta=["referential"] * N_CHANNELS,
                batch_size=BATCH_SIZE,
                device=torch.device("cpu"),
            )
            shape = str(out.shape) if out is not None else "None"
            print(f"  {variant:25} -> {shape} ✓")
        except Exception as e:
            print(f"  {variant:25} -> FAILED: {e}")
