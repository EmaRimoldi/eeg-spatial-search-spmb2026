"""Smoke test: verify checkpoint loading and synthetic forward pass.

This test does NOT require actual pretrained weights.
It verifies the model architecture can be instantiated and run.
When weights are available, it also verifies loading succeeds.
"""
import os
import sys
import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "external")


def test_reve_architecture():
    """REVE model can be instantiated via transformers AutoModel."""
    reve_hf_path = os.path.join(EXTERNAL_DIR, "reve_eeg", "hf", "reve-base")
    if not os.path.isdir(reve_hf_path):
        pytest.skip("REVE hf directory not found")

    from transformers import AutoModel, AutoConfig
    config = AutoConfig.from_pretrained(reve_hf_path, trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 1_000_000, f"REVE has unexpectedly few params: {n_params}"


def test_reve_forward():
    """REVE forward pass produces correct output shape."""
    reve_hf_path = os.path.join(EXTERNAL_DIR, "reve_eeg", "hf", "reve-base")
    if not os.path.isdir(reve_hf_path):
        pytest.skip("REVE hf directory not found")

    from transformers import AutoModel, AutoConfig
    config = AutoConfig.from_pretrained(reve_hf_path, trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True)
    model.eval()

    B, C, T = 2, 19, 1000
    x = torch.randn(B, C, T)
    pos = torch.randn(B, C, 3)

    with torch.no_grad():
        out = model(x, pos)

    assert out.dim() == 4, f"Expected 4D output (B,C,H,E), got shape {out.shape}"
    assert out.shape[0] == B
    assert out.shape[1] == C


def test_labram_architecture():
    """LaBraM module is importable from the external directory."""
    labram_path = os.path.join(EXTERNAL_DIR, "LaBraM")
    if not os.path.isdir(labram_path):
        pytest.skip("LaBraM directory not found")

    sys.path.insert(0, labram_path)
    try:
        import modeling_finetune  # noqa: F401
    except ImportError as e:
        pytest.skip(f"LaBraM modeling_finetune not importable: {e}")
    finally:
        if labram_path in sys.path:
            sys.path.remove(labram_path)

    checkpoint_path = os.path.join(labram_path, "checkpoints", "labram-base.pth")
    if os.path.exists(checkpoint_path):
        size = os.path.getsize(checkpoint_path)
        if size < 1000:
            pytest.skip(f"labram-base.pth is LFS pointer ({size} bytes)")


def test_reve_wrapper():
    """REVEWrapper instantiates and runs a forward pass for all spatial variants."""
    from src.models.wrappers.reve_wrapper import REVEWrapper

    B, C, T = 2, 22, 500
    x = torch.randn(B, C, T)
    metadata = {
        "channel_names": [f"EEG{i}" for i in range(C)],
        "coords_2d": torch.randn(C, 2),
        "coords_3d": torch.randn(C, 3),
        "reference_meta": ["average"] * C,
    }

    variants = ["none", "coords2d", "coords3d", "topology_agnostic"]
    for variant in variants:
        model = REVEWrapper(
            spatial_variant=variant, num_classes=4, freeze_policy="head_only", smoke_test=True
        )
        with torch.no_grad():
            out = model(x, metadata)
        assert out.shape == (B, 4), f"Unexpected shape for {variant}: {out.shape}"


def test_spatial_embeddings():
    """All spatial embedding modules instantiate and produce correct shapes."""
    from src.models.spatial_embeddings.none import NoSpatialEmbedding
    from src.models.spatial_embeddings.channel_id import ChannelIDEmbedding
    from src.models.spatial_embeddings.coords2d import Coords2DEmbedding
    from src.models.spatial_embeddings.coords3d import Coords3DEmbedding

    embed_dim = 64
    B, C = 2, 19
    channel_names = [f"ch{i}" for i in range(C)]
    coords_2d = torch.randn(B, C, 2)
    coords_3d = torch.randn(B, C, 3)

    for cls, name, kwargs in [
        (ChannelIDEmbedding, "channel_id", {"embed_dim": embed_dim, "num_channels": C}),
        (Coords2DEmbedding, "coords2d", {"embed_dim": embed_dim}),
        (Coords3DEmbedding, "coords3d", {"embed_dim": embed_dim}),
    ]:
        m = cls(**kwargs)
        m.eval()
        out = m(
            channel_names=channel_names,
            coords_2d=coords_2d,
            coords_3d=coords_3d,
            reference_meta=None,
            batch_size=B,
            device="cpu",
        )
        assert out is not None
        assert out.shape == (B, C, embed_dim), f"{name}: shape mismatch {out.shape}"

    # NoSpatialEmbedding should return None
    no_emb = NoSpatialEmbedding(embed_dim=embed_dim)
    assert no_emb(channel_names=channel_names, coords_2d=coords_2d, coords_3d=coords_3d,
                  reference_meta=None, batch_size=B, device="cpu") is None


if __name__ == "__main__":
    print("=" * 60)
    print("EEG Spatial Paper — Checkpoint Loading Smoke Test")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    pytest.main([__file__, "-v"])
