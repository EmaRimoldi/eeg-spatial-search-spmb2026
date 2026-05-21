"""Wrapper construction and spatial-variant plumbing across FM backbones."""

import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

CHANNEL_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]
BATCH_SIZE = 2
N_CHANNELS = len(CHANNEL_NAMES)
TIMEPOINTS = 400
NUM_CLASSES = 4


def make_metadata():
    return {
        "channel_names": CHANNEL_NAMES,
        "coords_2d": torch.randn(N_CHANNELS, 2),
        "coords_3d": torch.randn(N_CHANNELS, 3),
        "reference_meta": ["average"] * N_CHANNELS,
        "channel_mask": torch.ones(BATCH_SIZE, N_CHANNELS, dtype=torch.bool),
    }


@pytest.mark.parametrize("backbone", ["reve", "biot", "labram", "cbramod"])
@pytest.mark.parametrize("variant", ["none", "channel_id", "coords2d", "coords3d"])
def test_fm_wrappers_construct_and_forward_spatial_variants(backbone, variant):
    from src.models.wrappers.registry import build_model

    x = torch.randn(BATCH_SIZE, N_CHANNELS, TIMEPOINTS)
    model = build_model(
        backbone=backbone,
        spatial_variant=variant,
        num_classes=NUM_CLASSES,
        freeze_policy="head_only",
        smoke_test=True,
    )
    model.eval()
    with torch.no_grad():
        logits = model(x, make_metadata())
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
    assert torch.isfinite(logits).all()


def test_reve_coords3d_enters_native_position_tensor():
    from src.models.wrappers.reve_wrapper import REVEWrapper

    metadata = make_metadata()
    model = REVEWrapper(spatial_variant="coords3d", num_classes=NUM_CLASSES, smoke_test=True)
    pos = model._build_pos_tensor(metadata, BATCH_SIZE, N_CHANNELS, torch.device("cpu"))
    expected = metadata["coords_3d"].unsqueeze(0).expand(BATCH_SIZE, -1, -1)
    assert torch.allclose(pos, expected)


def test_biot_coords3d_builds_channel_token_residual():
    from src.models.wrappers.biot_wrapper import BIOTWrapper

    metadata = make_metadata()
    x = torch.randn(BATCH_SIZE, N_CHANNELS, TIMEPOINTS)
    model = BIOTWrapper(
        spatial_variant="coords3d",
        num_classes=NUM_CLASSES,
        smoke_test=True,
        input_channels=N_CHANNELS,
        max_channels=18,
    )
    mapped_x, input_mask, mapped_mask, projection = model._project_input(x, metadata)
    state = model._build_spatial_state(
        metadata=metadata,
        batch_size=BATCH_SIZE,
        device=torch.device("cpu"),
        input_channel_mask=input_mask,
        projection=projection,
        mapped_channel_mask=mapped_mask,
    )
    assert mapped_x.shape == (BATCH_SIZE, model.max_channels, TIMEPOINTS)
    assert state is not None
    assert state.shape == (BATCH_SIZE, model.max_channels, model.embed_dim)
    assert state.abs().sum() > 0


def test_labram_coords3d_builds_selected_channel_position_state():
    from src.models.wrappers.labram_wrapper import LaBraMWrapper

    metadata = make_metadata()
    x = torch.randn(BATCH_SIZE, N_CHANNELS, TIMEPOINTS)
    model = LaBraMWrapper(spatial_variant="coords3d", num_classes=NUM_CLASSES, smoke_test=True)
    _, channel_mask, supported_indices, input_indices, selected_names = model._prepare_patches(x, metadata)
    state = model._build_labram_spatial_state(
        metadata=metadata,
        input_indices=input_indices,
        selected_names=selected_names,
        batch_size=BATCH_SIZE,
        device=torch.device("cpu"),
        channel_mask=channel_mask,
    )
    assert supported_indices
    assert state is not None
    assert state.shape == (BATCH_SIZE, len(selected_names), model.embed_dim)
    assert state[:, channel_mask[0], :].abs().sum() > 0


def test_cbramod_coords3d_builds_native_attention_bias():
    from src.models.wrappers.cbramod_wrapper import CBraModWrapper

    metadata = make_metadata()
    model = CBraModWrapper(spatial_variant="coords3d", num_classes=NUM_CLASSES, smoke_test=True)
    spatial, explicit_bias = model._build_spatial_embedding(
        metadata=metadata,
        batch_size=BATCH_SIZE,
        n_channels=N_CHANNELS,
        device=torch.device("cpu"),
    )
    prepared = model._prepare_spatial_injection(spatial, metadata["channel_mask"])
    native_bias = model._build_native_attention_bias(prepared, explicit_bias, metadata["channel_mask"])
    assert native_bias is not None
    assert native_bias.shape == (BATCH_SIZE, N_CHANNELS, N_CHANNELS)
    assert torch.isfinite(native_bias).all()


def test_trainer_flattens_nested_wrapper_kwargs(tmp_path):
    from src.training.train import SpatialAblationTrainer

    trainer = SpatialAblationTrainer(
        {
            "backbone": "cbramod",
            "spatial_variant": "coords3d_rbf",
            "num_classes": NUM_CLASSES,
            "freeze_policy": "head_only",
            "smoke_test": True,
            "wrapper_kwargs": {
                "graph_depth": 1,
                "graph_k_neighbors": 4,
                "native_pair_dim": 32,
                "spatial_embedding_kwargs": {
                    "n_rbf": 4,
                    "anchor_mode": "axes_diagonal",
                },
            },
        },
        tmp_path,
    )
    model = trainer.build_model()

    assert model.graph_depth == 1
    assert model.graph_k_neighbors == 4
    assert model.native_pair_dim == 32
    assert model.spatial_adapter.spatial_embedding_kwargs["n_rbf"] == 4


def test_real_run_missing_backbone_fails_without_smoke(monkeypatch, tmp_path):
    from src.models.wrappers import biot_wrapper, cbramod_wrapper, reve_wrapper

    monkeypatch.setattr(biot_wrapper, "_BIOT_MODEL", tmp_path / "missing_biot.py")
    with pytest.raises(RuntimeError, match="smoke_test=True"):
        biot_wrapper.BIOTWrapper(spatial_variant="coords3d", num_classes=NUM_CLASSES)

    monkeypatch.setattr(cbramod_wrapper, "_EEG_FM_BENCH_CBRAMOD_MODEL", tmp_path / "missing_cbramod.py")
    with pytest.raises(RuntimeError, match="smoke_test=True"):
        cbramod_wrapper.CBraModWrapper(spatial_variant="coords3d", num_classes=NUM_CLASSES)

    monkeypatch.setattr(reve_wrapper, "_REVE_HF_PATH", tmp_path / "missing_reve")
    with pytest.raises(RuntimeError, match="smoke_test=True"):
        reve_wrapper.REVEWrapper(spatial_variant="coords3d", num_classes=NUM_CLASSES)
