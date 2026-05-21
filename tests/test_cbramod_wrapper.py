"""Smoke tests for the CBraMod wrapper scaffold."""

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

CHANNEL_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]
VARIANTS_TO_TEST = [
    "none",
    "channel_id",
    "coords2d",
    "coords3d",
    "coords3d_reference",
    "coords3d_rbf",
    "coords3d_geodesic_rbf",
    "topology_agnostic",
]
BATCH_SIZE = 1
N_CHANNELS = len(CHANNEL_NAMES)
TIMEPOINTS = 400
NUM_CLASSES = 4


def make_metadata(batch_size: int = BATCH_SIZE) -> dict:
    coords_2d = torch.randn(N_CHANNELS, 2)
    coords_3d = torch.randn(N_CHANNELS, 3)
    channel_mask = torch.ones(batch_size, N_CHANNELS, dtype=torch.bool)
    channel_mask[:, -2:] = False  # exercise lightweight masking logic
    return {
        "channel_names": CHANNEL_NAMES,
        "coords_2d": coords_2d,
        "coords_3d": coords_3d,
        "reference_meta": ["average"] * N_CHANNELS,
        "channel_mask": channel_mask,
    }


def test_registry_lists_cbramod():
    from src.models.wrappers.registry import list_backbones

    backbones = list_backbones()
    assert "cbramod" in backbones



def test_cbramod_wrapper_forward_dummy_metadata():
    from src.models.wrappers.registry import build_model

    x = torch.randn(BATCH_SIZE, N_CHANNELS, TIMEPOINTS)
    metadata = make_metadata()

    for variant in VARIANTS_TO_TEST:
        model = build_model(
            backbone="cbramod",
            spatial_variant=variant,
            num_classes=NUM_CLASSES,
            freeze_policy="head_only",
            smoke_test=True,
        )
        model.eval()

        with torch.no_grad():
            logits = model(x, metadata)

        assert logits.shape == (BATCH_SIZE, NUM_CLASSES), (
            f"{variant}: expected {(BATCH_SIZE, NUM_CLASSES)}, got {tuple(logits.shape)}"
        )
        assert torch.isfinite(logits).all(), f"{variant}: logits contain non-finite values"



def test_cbramod_wrapper_parameter_count():
    from src.models.wrappers.cbramod_wrapper import CBraModWrapper

    model = CBraModWrapper(
        spatial_variant="coords3d_reference",
        num_classes=NUM_CLASSES,
        freeze_policy="head_only",
        smoke_test=True,
    )
    counts = model.parameter_count()

    assert counts["total"] > 0
    assert counts["trainable"] > 0
    assert counts["classifier_head"] > 0
    assert counts["spatial"] >= 0


def test_cbramod_wrapper_forward_with_graph_layers():
    from src.models.wrappers.registry import build_model

    x = torch.randn(BATCH_SIZE, N_CHANNELS, TIMEPOINTS)
    metadata = make_metadata()

    model = build_model(
        backbone="cbramod",
        spatial_variant="coords2d",
        num_classes=NUM_CLASSES,
        freeze_policy="full",
        graph_depth=2,
        smoke_test=True,
    )
    model.eval()

    with torch.no_grad():
        logits = model(x, metadata)

    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
    assert torch.isfinite(logits).all()
    counts = model.parameter_count()
    assert counts["graph"] > 0


def test_cbramod_graph_neighbors_are_sparse_and_normalized():
    from src.models.wrappers.cbramod_wrapper import CBraModWrapper

    model = CBraModWrapper(
        spatial_variant="coords3d",
        num_classes=NUM_CLASSES,
        freeze_policy="head_only",
        graph_depth=1,
        graph_k_neighbors=2,
        graph_sigma_scale=0.75,
        smoke_test=True,
    )
    metadata = make_metadata()
    adjacency = model._build_graph_adjacency(
        metadata=metadata,
        channel_mask=metadata["channel_mask"],
        device=torch.device("cpu"),
    )

    assert adjacency.shape == (BATCH_SIZE, N_CHANNELS, N_CHANNELS)
    assert torch.isfinite(adjacency).all()
    valid_rows = metadata["channel_mask"]
    assert torch.allclose(adjacency.sum(dim=-1)[valid_rows], torch.ones(valid_rows.sum()), atol=1e-5)
    assert (adjacency > 0).sum(dim=-1)[valid_rows].max() < N_CHANNELS


def test_cbramod_native_attention_bias_is_masked_and_nontrivial():
    from src.models.wrappers.cbramod_wrapper import CBraModWrapper

    model = CBraModWrapper(
        spatial_variant="coords3d_reference",
        num_classes=NUM_CLASSES,
        freeze_policy="head_only",
        smoke_test=True,
    )
    metadata = make_metadata()
    channel_mask = metadata["channel_mask"]

    raw_spatial, _ = model._build_spatial_embedding(
        metadata=metadata,
        batch_size=BATCH_SIZE,
        n_channels=N_CHANNELS,
        device=torch.device("cpu"),
    )
    assert raw_spatial is not None

    prepared = model._prepare_spatial_injection(raw_spatial, channel_mask)
    assert prepared is not None
    assert prepared.shape == raw_spatial.shape
    assert torch.isfinite(prepared).all()
    assert prepared.abs().sum() > 0

    # Masked channels should stay exactly zero after preprocessing.
    assert torch.allclose(prepared[:, -2:, :], torch.zeros_like(prepared[:, -2:, :]))

    valid = channel_mask.unsqueeze(-1).expand_as(prepared)
    valid_prepared = prepared[valid].view(BATCH_SIZE, -1, model.backbone_dim)
    channel_mean = valid_prepared.mean(dim=1)
    assert torch.allclose(channel_mean, torch.zeros_like(channel_mean), atol=1e-5, rtol=1e-4)

    native_bias = model._build_native_attention_bias(prepared, None, channel_mask)
    assert native_bias is not None
    assert native_bias.shape == (BATCH_SIZE, N_CHANNELS, N_CHANNELS)
    assert torch.isfinite(native_bias).all()
    assert native_bias[:, :, :-2].abs().sum() > 0
    assert torch.all(native_bias[:, :, -2:] < -1e3)

    expanded = model._expand_spatial_attn_bias(native_bias, patch_count=2, num_heads=4)
    assert expanded.shape == (BATCH_SIZE * 2 * 4, N_CHANNELS, N_CHANNELS)


def test_cbramod_none_variant_skips_native_attention_bias():
    from src.models.wrappers.cbramod_wrapper import CBraModWrapper

    model = CBraModWrapper(
        spatial_variant="none",
        num_classes=NUM_CLASSES,
        freeze_policy="head_only",
        smoke_test=True,
    )
    metadata = make_metadata()
    channel_mask = metadata["channel_mask"]

    native_bias = model._build_native_attention_bias(None, None, channel_mask)
    assert native_bias is None


def test_cbramod_flatten_heads_are_configurable():
    from src.models.wrappers.cbramod_wrapper import CBraModWrapper

    x = torch.randn(BATCH_SIZE, N_CHANNELS, TIMEPOINTS)
    metadata = make_metadata()

    for head_type in ("flatten_linear", "flatten_mlp"):
        model = CBraModWrapper(
            spatial_variant="coords3d",
            num_classes=NUM_CLASSES,
            freeze_policy="head_only",
            smoke_test=True,
            classifier_head={
                "head_type": head_type,
                "n_channels": N_CHANNELS,
                "n_patches": TIMEPOINTS // 200,
            },
        )
        model.eval()
        with torch.no_grad():
            logits = model(x, metadata)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
