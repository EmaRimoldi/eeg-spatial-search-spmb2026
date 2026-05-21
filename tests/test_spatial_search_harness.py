"""Tests for the staged EEG-FM-Bench spatial-search harness."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = PROJECT_ROOT / "scripts" / "train" / "eegfm_spatial_search.py"


def load_harness_module():
    spec = importlib.util.spec_from_file_location("eegfm_spatial_search", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stage1_manifest_space_is_coordinate_only(tmp_path):
    harness = load_harness_module()
    manifest = tmp_path / "manifest.tsv"
    rows = harness.build_stage_rows(
        stage="stage1_screen",
        datasets=["bcic_2a"],
        backbones=["biot", "labram", "cbramod"],
        seeds=[42],
        manifest_path=manifest,
        run_root=tmp_path / "runs",
        config_root=tmp_path / "generated_configs",
    )

    assert len(rows) == 26
    assert {row["variant"] for row in rows}.isdisjoint({"none", "channel_id"})
    assert {"coords3d", "coords3d_distbias", "coords3d_rbf", "coords3d_geodesic_rbf"}.issubset(
        {row["variant"] for row in rows}
    )
    assert {row["backbone"] for row in rows} == {"biot", "labram", "cbramod"}
    assert all(row["stage"] == "stage1_screen" for row in rows)
    assert all(row["dataset"] == "bcic_2a" for row in rows)
    assert all(row["seed"] == row["seeds"] == "42" for row in rows)


def test_stage1_config_uses_wrapper_kwargs_and_search_record(tmp_path):
    harness = load_harness_module()
    manifest = tmp_path / "manifest.tsv"
    rows = harness.build_stage_rows(
        stage="stage1_screen",
        datasets=["bcic_2a"],
        backbones=["cbramod"],
        seeds=[42],
        manifest_path=manifest,
        run_root=tmp_path / "runs",
        config_root=tmp_path / "generated_configs",
    )
    target = next(row for row in rows if row["hparams_id"] == "coords3d_distbias_pair64_scale0p05_n8")
    harness.write_configs([target], manifest)

    cfg = yaml.safe_load(Path(target["config"]).read_text())
    assert cfg["dataset"] == "bcic_2a"
    assert cfg["spatial_variant"] == "coords3d_distbias"
    assert cfg["search"]["run_id"] == target["run_id"]
    assert cfg["search"]["baseline_fixed_externally"] is True
    assert cfg["wrapper_kwargs"]["native_pair_dim"] == 64
    assert cfg["wrapper_kwargs"]["dist_bias_scale"] == 0.05
    assert cfg["wrapper_kwargs"]["spatial_embedding_kwargs"]["n_rbf"] == 8
    assert json.loads(target["hparams_json"])["dist_bias_scale"] == 0.05


def test_stage1_includes_graph_neighborhood_knobs(tmp_path):
    harness = load_harness_module()
    rows = harness.build_stage_rows(
        stage="stage1_screen",
        datasets=["workload"],
        backbones=["cbramod"],
        seeds=[42],
        manifest_path=tmp_path / "manifest.tsv",
        run_root=tmp_path / "runs",
        config_root=tmp_path / "generated_configs",
    )
    graph_rows = [row for row in rows if "graph1_k4" in row["hparams_id"]]

    assert graph_rows
    for row in graph_rows:
        hparams = json.loads(row["hparams_json"])
        assert hparams["graph_depth"] == 1
        assert hparams["graph_k_neighbors"] == 4


def test_biot_stage1_config_keeps_checkpoint_channel_count(tmp_path):
    harness = load_harness_module()
    manifest = tmp_path / "manifest.tsv"
    rows = harness.build_stage_rows(
        stage="stage1_screen",
        datasets=["motor_mv_img"],
        backbones=["biot"],
        seeds=[42],
        manifest_path=manifest,
        run_root=tmp_path / "runs",
        config_root=tmp_path / "generated_configs",
    )
    target = next(row for row in rows if row["variant"] == "coords3d")
    harness.write_configs([target], manifest)

    cfg = yaml.safe_load(Path(target["config"]).read_text())
    assert cfg["wrapper_kwargs"]["max_channels"] == 18
    assert cfg["wrapper_kwargs"]["input_channels"] == harness.DATASETS["motor_mv_img"]["n_channels"]
    assert cfg["wrapper_kwargs"]["dropout"] == 0.3
