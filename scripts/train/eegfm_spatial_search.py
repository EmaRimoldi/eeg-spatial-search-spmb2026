#!/usr/bin/env python3
"""Build and launch staged EEG-FM-Bench spatial-coordinate searches."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "spatial_search"
DEFAULT_MANIFEST = RESULTS_ROOT / "manifest.tsv"
DEFAULT_RUN_ROOT = RESULTS_ROOT / "runs"
DEFAULT_CONFIG_ROOT = RESULTS_ROOT / "generated_configs"
DEFAULT_LOG_ROOT = RESULTS_ROOT / "slurm_logs"
DEFAULT_STATUS_ROOT = RESULTS_ROOT / "status"
DEFAULT_RUNNER = PROJECT_ROOT / "scripts" / "train" / "run_eegfm_spatial_search_task.sh"
DEFAULT_PYTHON = "/home/erimoldi/.conda/envs/sparse-hate/bin/python"
DEFAULT_PARTITIONS = "mit_normal_gpu,mit_preemptable,ou_bcs_low,ou_bcs_normal,pi_tpoggio"

OPERATIONAL_SEEDS = [42, 43, 44, 45, 46]
PROMOTE_NEW_SEEDS = [43, 44, 45, 46]

DATASETS = {
    "bcic_2a": {
        "slug": "bcic2a",
        "source_prefix": "bcic2a",
        "num_classes": 4,
        "n_channels": 22,
        "eeg_size": 800,
    },
    "motor_mv_img": {
        "slug": "motor_mv_img",
        "source_prefix": "physiomi",
        "num_classes": 4,
        "n_channels": 64,
        "eeg_size": 800,
    },
    "workload": {
        "slug": "workload",
        "source_prefix": "workload",
        "num_classes": 2,
        "n_channels": 19,
        "eeg_size": 800,
    },
}

FIELDNAMES = [
    "run_id",
    "stage",
    "dataset",
    "dataset_slug",
    "backbone",
    "variant",
    "seed",
    "seeds",
    "hparams_id",
    "hparams_json",
    "config",
    "output_dir",
    "job_name",
    "status",
    "notes",
]


@dataclass(frozen=True)
class SearchSpec:
    variant: str
    hparams_id: str
    wrapper_kwargs: dict
    notes: str = ""


STAGE_SHORT = {
    "stage1_screen": "s1",
    "stage2_promote": "s2",
}

STAGE1_PLAN: dict[str, list[SearchSpec]] = {
    "biot": [
        SearchSpec("coords3d", "coords3d_gain0p10", {"spatial_init_gain": 0.10}, "BIOT 3D coordinate token residual"),
        SearchSpec(
            "coords3d_rbf",
            "coords3d_rbf_axes_n6_gain0p10",
            {"spatial_init_gain": 0.10, "spatial_embedding_kwargs": {"n_rbf": 6, "anchor_mode": "axes"}},
            "BIOT Euclidean anchor-RBF coordinate token residual",
        ),
        SearchSpec(
            "coords3d_geodesic_rbf",
            "coords3d_geodesic_rbf_axes_n6_gain0p10",
            {"spatial_init_gain": 0.10, "spatial_embedding_kwargs": {"n_rbf": 6, "anchor_mode": "axes"}},
            "BIOT spherical geodesic anchor-RBF token residual",
        ),
        SearchSpec("coords3d_reference", "coords3d_reference_gain0p10", {"spatial_init_gain": 0.10}, "BIOT 3D coordinates plus reference metadata"),
    ],
    "labram": [
        SearchSpec("coords3d", "coords3d_residual", {"spatial_pos_mode": "residual"}, "3D coordinate residual on native positions"),
        SearchSpec("coords3d", "coords3d_replace", {"spatial_pos_mode": "replace"}, "3D coordinate replacement of native positions"),
        SearchSpec(
            "coords3d_rbf",
            "coords3d_rbf_axes_n6_residual",
            {"spatial_pos_mode": "residual", "spatial_embedding_kwargs": {"n_rbf": 6, "anchor_mode": "axes"}},
            "Euclidean anchor-RBF residual on native positions",
        ),
        SearchSpec(
            "coords3d_rbf",
            "coords3d_rbf_axes_n6_replace",
            {"spatial_pos_mode": "replace", "spatial_embedding_kwargs": {"n_rbf": 6, "anchor_mode": "axes"}},
            "Euclidean anchor-RBF replacement of native positions",
        ),
        SearchSpec(
            "coords3d_geodesic_rbf",
            "coords3d_geodesic_rbf_axes_n6_residual",
            {"spatial_pos_mode": "residual", "spatial_embedding_kwargs": {"n_rbf": 6, "anchor_mode": "axes"}},
            "Spherical geodesic anchor-RBF residual positions",
        ),
        SearchSpec(
            "coords3d_geodesic_rbf",
            "coords3d_geodesic_rbf_axes_n6_replace",
            {"spatial_pos_mode": "replace", "spatial_embedding_kwargs": {"n_rbf": 6, "anchor_mode": "axes"}},
            "Spherical geodesic anchor-RBF replacement positions",
        ),
        SearchSpec("coords3d_reference", "coords3d_reference_residual", {"spatial_pos_mode": "residual"}, "3D reference-aware residual positions"),
        SearchSpec("coords3d_reference", "coords3d_reference_replace", {"spatial_pos_mode": "replace"}, "3D reference-aware replacement positions"),
    ],
    "cbramod": [
        SearchSpec("coords3d", "coords3d_pair32", {"native_pair_dim": 32, "graph_depth": 0}, "3D native bias with small pair projection"),
        SearchSpec("coords3d", "coords3d_pair64", {"native_pair_dim": 64, "graph_depth": 0}, "3D native bias default pair projection"),
        SearchSpec("coords3d_reference", "coords3d_reference_pair64", {"native_pair_dim": 64, "graph_depth": 0}, "reference-aware 3D native bias"),
        SearchSpec("coords3d_reference", "coords3d_reference_pair128", {"native_pair_dim": 128, "graph_depth": 0}, "reference-aware 3D larger pair projection"),
        SearchSpec(
            "coords3d_distbias",
            "coords3d_distbias_pair64_scale0p02_n8",
            {"native_pair_dim": 64, "graph_depth": 0, "dist_bias_scale": 0.02, "spatial_embedding_kwargs": {"n_rbf": 8}},
            "3D native bias plus weak explicit RBF distance bias",
        ),
        SearchSpec(
            "coords3d_distbias",
            "coords3d_distbias_pair64_scale0p05_n8",
            {"native_pair_dim": 64, "graph_depth": 0, "dist_bias_scale": 0.05, "spatial_embedding_kwargs": {"n_rbf": 8}},
            "3D native bias plus default explicit RBF distance bias",
        ),
        SearchSpec(
            "coords3d_distbias",
            "coords3d_distbias_pair128_scale0p05_n12",
            {"native_pair_dim": 128, "graph_depth": 0, "dist_bias_scale": 0.05, "spatial_embedding_kwargs": {"n_rbf": 12}},
            "3D distance bias with larger pair projection and richer RBF basis",
        ),
        SearchSpec(
            "coords3d_rbf",
            "coords3d_rbf_axes_n6_pair64",
            {"native_pair_dim": 64, "graph_depth": 0, "spatial_embedding_kwargs": {"n_rbf": 6, "anchor_mode": "axes"}},
            "Euclidean anchor-RBF native channel-attention bias",
        ),
        SearchSpec(
            "coords3d_rbf",
            "coords3d_rbf_axesdiag_n4_pair64",
            {"native_pair_dim": 64, "graph_depth": 0, "spatial_embedding_kwargs": {"n_rbf": 4, "anchor_mode": "axes_diagonal"}},
            "Euclidean anchor-RBF native bias with diagonal anchors",
        ),
        SearchSpec(
            "coords3d_rbf",
            "coords3d_rbf_axes_n8_pair128",
            {"native_pair_dim": 128, "graph_depth": 0, "spatial_embedding_kwargs": {"n_rbf": 8, "anchor_mode": "axes"}},
            "Euclidean anchor-RBF larger pair projection",
        ),
        SearchSpec(
            "coords3d_geodesic_rbf",
            "coords3d_geodesic_rbf_axes_n6_pair64",
            {"native_pair_dim": 64, "graph_depth": 0, "spatial_embedding_kwargs": {"n_rbf": 6, "anchor_mode": "axes"}},
            "Spherical geodesic anchor-RBF native bias",
        ),
        SearchSpec(
            "coords3d_geodesic_rbf",
            "coords3d_geodesic_rbf_axesdiag_n4_pair64",
            {"native_pair_dim": 64, "graph_depth": 0, "spatial_embedding_kwargs": {"n_rbf": 4, "anchor_mode": "axes_diagonal"}},
            "Spherical geodesic anchor-RBF native bias with diagonal anchors",
        ),
        SearchSpec(
            "coords3d",
            "coords3d_pair64_graph1_k4",
            {"native_pair_dim": 64, "graph_depth": 1, "graph_k_neighbors": 4, "graph_sigma_scale": 1.0},
            "3D native bias plus one k-neighborhood graph layer",
        ),
        SearchSpec(
            "coords3d_geodesic_rbf",
            "coords3d_geodesic_rbf_pair64_graph1_k4",
            {
                "native_pair_dim": 64,
                "graph_depth": 1,
                "graph_k_neighbors": 4,
                "graph_sigma_scale": 1.0,
                "spatial_embedding_kwargs": {"n_rbf": 6, "anchor_mode": "axes"},
            },
            "Geodesic RBF native bias plus one k-neighborhood graph layer",
        ),
    ],
}


def parse_csv_list(raw: str | None, allowed: Iterable[str], label: str) -> list[str]:
    allowed_set = set(allowed)
    if raw is None or raw == "all":
        return list(allowed)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(values) - allowed_set)
    if unknown:
        raise SystemExit(f"Unknown {label}: {unknown}. Allowed: {sorted(allowed_set)}")
    return values


def parse_seed_list(raw: str | None, default: list[int]) -> list[int]:
    if raw is None:
        return list(default)
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    invalid = [seed for seed in seeds if seed not in OPERATIONAL_SEEDS]
    if invalid:
        raise SystemExit(f"Seeds must come from {OPERATIONAL_SEEDS}; got {invalid}")
    return seeds


def hparams_json(wrapper_kwargs: dict) -> str:
    return json.dumps(wrapper_kwargs, sort_keys=True, separators=(",", ":"))


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def base_config_path(dataset: str, backbone: str) -> Path:
    prefix = DATASETS[dataset]["source_prefix"]
    return PROJECT_ROOT / "configs" / "eegfm_bench" / f"{prefix}_{backbone}_table2.yaml"


def load_base_config(dataset: str, backbone: str) -> dict:
    path = base_config_path(dataset, backbone)
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline config: {path}")
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def fixed_wrapper_kwargs(dataset: str, backbone: str, base: dict) -> dict:
    meta = DATASETS[dataset]
    model_cfg = base.get("model", {})
    head_cfg = model_cfg.get("classifier_head", {})
    if backbone == "biot":
        return {
            "embed_dim": int(model_cfg.get("emb_size", 256)),
            "heads": int(model_cfg.get("heads", 8)),
            "depth": int(model_cfg.get("depth", 4)),
            "max_channels": int(model_cfg.get("max_channels", 18)),
            "input_channels": int(meta["n_channels"]),
            "use_channel_conv": bool(model_cfg.get("use_channel_conv", True)),
            "router_init_mode": "montage_prior",
            "classifier_hidden_dims": list(head_cfg.get("hidden_dims", [128])),
            "dropout": float(head_cfg.get("dropout", 0.3)),
            "n_fft": int(model_cfg.get("n_fft", 200)),
            "hop_length": int(model_cfg.get("hop_length", 100)),
        }
    if backbone == "labram":
        return {
            "eeg_size": int(meta["eeg_size"]),
            "patch_size": int(model_cfg.get("patch_size", 200)),
            "classifier_hidden_dims": list(head_cfg.get("hidden_dims", [128])),
            "dropout": float(head_cfg.get("dropout", 0.3)),
            "input_scale": 0.01,
        }
    if backbone == "cbramod":
        return {
            "n_channels": int(meta["n_channels"]),
            "eeg_size": int(meta["eeg_size"]),
            "classifier_head": model_cfg.get("classifier_head", {}),
        }
    raise ValueError(f"Unsupported backbone for spatial search: {backbone}")


def make_train_config(
    *,
    stage: str,
    dataset: str,
    backbone: str,
    spec: SearchSpec,
    seed: int,
    run_id: str,
    manifest_path: Path,
    promoted_from_run_id: str | None = None,
) -> dict:
    if spec.variant in {"none", "channel_id"}:
        raise ValueError("Spatial search must not generate none/channel_id baseline rows")

    meta = DATASETS[dataset]
    base = load_base_config(dataset, backbone)
    training = base.get("training", {})
    data_cfg = base.get("data", {})
    model_cfg = base.get("model", {})
    max_lr = float(training["max_lr"])
    wrapper_kwargs = fixed_wrapper_kwargs(dataset, backbone, base)
    wrapper_kwargs.update(spec.wrapper_kwargs)

    search_record = {
        "run_id": run_id,
        "stage": stage,
        "dataset": dataset,
        "dataset_slug": meta["slug"],
        "backbone": backbone,
        "variant": spec.variant,
        "seed": int(seed),
        "hparams_id": spec.hparams_id,
        "hparams": spec.wrapper_kwargs,
        "manifest": str(manifest_path),
        "baseline_fixed_externally": True,
    }
    if promoted_from_run_id:
        search_record["promoted_from_run_id"] = promoted_from_run_id

    return {
        "backbone": backbone,
        "spatial_variant": spec.variant,
        "freeze_policy": "full",
        "dataset": dataset,
        "num_classes": int(meta["num_classes"]),
        "seed": int(seed),
        "epochs": int(training["max_epochs"]),
        "batch_size": int(data_cfg.get("batch_size", 32)),
        "num_workers": int(data_cfg.get("num_workers", 4)),
        "allow_synthetic_fallback": False,
        "smoke_test": False,
        "checkpoint": model_cfg.get("pretrained_path"),
        "optimizer": {
            "name": "adamw",
            "lr": max_lr,
            "weight_decay": float(training.get("weight_decay", 1.0e-4)),
            "backbone_lr": max_lr * float(training.get("encoder_lr_scale", 1.0)),
            "head_lr": max_lr,
            "spatial_lr": max_lr,
        },
        "scheduler": {
            "name": "cosine",
            "warmup_epochs": int(training.get("warmup_epochs", 5)),
        },
        "early_stopping_patience": 1000,
        "wrapper_kwargs": wrapper_kwargs,
        "search": search_record,
    }


def make_run_id(stage: str, dataset: str, backbone: str, hparams_id: str, seed: int) -> str:
    return f"{stage}_{DATASETS[dataset]['slug']}_{backbone}_{hparams_id}_s{seed}"


def make_job_name(stage: str, dataset: str, backbone: str, hparams_id: str, seed: int) -> str:
    short_stage = STAGE_SHORT.get(stage, stage[:2])
    safe_hparams = hparams_id.replace("coords3d_", "c3_").replace("coords2d_", "c2_")
    return f"ss_{short_stage}_{DATASETS[dataset]['slug']}_{backbone}_{safe_hparams}_s{seed}"[:120]


def stage_specs(stage: str) -> dict[str, list[SearchSpec]]:
    if stage == "stage1_screen":
        return STAGE1_PLAN
    raise ValueError(f"Stage {stage!r} is generated through promote, not a static plan")


def build_stage_rows(
    *,
    stage: str,
    datasets: list[str],
    backbones: list[str],
    seeds: list[int],
    manifest_path: Path,
    run_root: Path,
    config_root: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    specs_by_backbone = stage_specs(stage)
    for dataset in datasets:
        for backbone in backbones:
            for spec in specs_by_backbone[backbone]:
                for seed in seeds:
                    run_id = make_run_id(stage, dataset, backbone, spec.hparams_id, seed)
                    config_path = config_root / stage / DATASETS[dataset]["slug"] / backbone / f"{run_id}.yaml"
                    rows.append(
                        {
                            "run_id": run_id,
                            "stage": stage,
                            "dataset": dataset,
                            "dataset_slug": DATASETS[dataset]["slug"],
                            "backbone": backbone,
                            "variant": spec.variant,
                            "seed": str(seed),
                            "seeds": str(seed),
                            "hparams_id": spec.hparams_id,
                            "hparams_json": hparams_json(spec.wrapper_kwargs),
                            "config": str(config_path),
                            "output_dir": str(run_root),
                            "job_name": make_job_name(stage, dataset, backbone, spec.hparams_id, seed),
                            "status": "planned",
                            "notes": spec.notes,
                        }
                    )
    return rows


def cbramod_head_override(head_type: str | None) -> dict | None:
    if head_type in (None, "base"):
        return None
    if head_type == "avg_pool":
        return {"head_type": "avg_pool", "hidden_dims": [128], "dropout": 0.3}
    if head_type == "flatten_mlp":
        return {"head_type": "flatten_mlp", "dropout": 0.1}
    raise ValueError(f"Unsupported CBraMod head override: {head_type}")


def apply_cbramod_head_override(rows: list[dict[str, str]], head_type: str | None) -> None:
    override = cbramod_head_override(head_type)
    if override is None:
        return
    suffix = f"head_{head_type.replace('_', '')}"
    for row in rows:
        if row["backbone"] != "cbramod":
            continue
        hparams = json.loads(row["hparams_json"])
        hparams["classifier_head"] = override
        if not row["hparams_id"].endswith(f"_{suffix}"):
            row["hparams_id"] = f"{row['hparams_id']}_{suffix}"
        row["hparams_json"] = hparams_json(hparams)
        row["run_id"] = make_run_id(row["stage"], row["dataset"], row["backbone"], row["hparams_id"], int(row["seed"]))
        row["config"] = str(Path(row["config"]).with_name(f"{row['run_id']}.yaml"))
        row["job_name"] = make_job_name(row["stage"], row["dataset"], row["backbone"], row["hparams_id"], int(row["seed"]))
        row["notes"] = f"{row.get('notes', '')}; cbramod_head={head_type}".strip("; ")


def write_configs(rows: list[dict[str, str]], manifest_path: Path, promoted_from: dict[str, str] | None = None) -> None:
    promoted_from = promoted_from or {}
    for row in rows:
        spec = SearchSpec(
            variant=row["variant"],
            hparams_id=row["hparams_id"],
            wrapper_kwargs=json.loads(row["hparams_json"]),
            notes=row.get("notes", ""),
        )
        config = make_train_config(
            stage=row["stage"],
            dataset=row["dataset"],
            backbone=row["backbone"],
            spec=spec,
            seed=int(row["seed"]),
            run_id=row["run_id"],
            manifest_path=manifest_path,
            promoted_from_run_id=promoted_from.get(row["run_id"]),
        )
        path = Path(row["config"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)


def write_readme(path: Path, manifest_path: Path, rows: list[dict[str, str]]) -> None:
    try:
        manifest_display = str(manifest_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        manifest_display = str(manifest_path)
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["stage"], row["backbone"])
        counts[key] = counts.get(key, 0) + 1
    count_lines = "\n".join(
        f"- {stage} / {backbone}: {count}"
        for (stage, backbone), count in sorted(counts.items())
    )
    text = f"""# EEG-FM-Bench Spatial Search

This directory records staged coordinate-only searches on the fixed EEG-FM-Bench
processed datasets: `bcic_2a`, `motor_mv_img`, and `workload`.

Baseline `none` and `channel_id` runs are not generated here; those baselines
are fixed externally. Final promoted comparisons use operational seeds
`42,43,44,45,46`.

## Stages

- `stage1_screen`: broad seed-42 screen over coordinate-based variants only.
- `stage2_promote`: generated from the stage-1 leaderboard. By default it adds
  seeds `43,44,45,46`, treating the matching seed-42 stage-1 run as the first
  final-comparison seed.

## Current Manifest

- Manifest: `{manifest_display}`
- Planned rows: {len(rows)}
{count_lines}

## Commands

Generate the stage-1 manifest and configs:

```bash
python scripts/train/eegfm_spatial_search.py generate --stage stage1_screen --force
```

Inspect a submission without launching jobs:

```bash
python scripts/train/eegfm_spatial_search.py submit --stage stage1_screen --dry-run --limit 3
```

Aggregate status and leaderboard:

```bash
python scripts/train/summarize_eegfm_spatial_search.py --results-root results/spatial_search
```

Generate stage-2 promotion rows after stage-1 metrics exist:

```bash
python scripts/train/eegfm_spatial_search.py promote --top-k 1
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def compact_array(indices: list[int]) -> str:
    if not indices:
        raise ValueError("No indices selected")
    indices = sorted(set(indices))
    ranges: list[str] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = idx
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def cmd_generate(args: argparse.Namespace) -> int:
    stage = args.stage
    datasets = parse_csv_list(args.datasets, DATASETS.keys(), "dataset")
    backbones = parse_csv_list(args.backbones, STAGE1_PLAN.keys(), "backbone")
    seeds = parse_seed_list(args.seeds, [42])
    manifest_path = args.manifest
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"{manifest_path} exists; pass --force to overwrite it")

    rows = build_stage_rows(
        stage=stage,
        datasets=datasets,
        backbones=backbones,
        seeds=seeds,
        manifest_path=manifest_path,
        run_root=args.run_root,
        config_root=args.config_root,
    )
    apply_cbramod_head_override(rows, args.cbramod_head)
    write_configs(rows, manifest_path)
    write_tsv(manifest_path, rows)
    if args.write_readme:
        write_readme(args.results_root / "README.md", manifest_path, rows)
    print(f"wrote_manifest={manifest_path}")
    print(f"planned_runs={len(rows)}")
    return 0


def load_leaderboard(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Leaderboard not found: {path}")
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def promote_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    leaderboard = load_leaderboard(args.leaderboard)
    source_rows = read_tsv(args.manifest)
    source_by_key = {
        (row["dataset"], row["backbone"], row["variant"], row["hparams_id"], row["hparams_json"]): row
        for row in source_rows
    }

    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in leaderboard:
        if row.get("stage") not in {"stage1_screen", "mixed"}:
            continue
        if int(row.get("n_completed", "0") or 0) < args.min_completed:
            continue
        bacc = row.get("mean_balanced_accuracy")
        if not bacc:
            continue
        groups.setdefault((row["dataset"], row["backbone"]), []).append(row)

    selected: list[dict[str, str]] = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda row: float(row["mean_balanced_accuracy"]), reverse=True)
        selected.extend(rows[: args.top_k])

    existing_ids = {row["run_id"] for row in source_rows}
    new_rows: list[dict[str, str]] = []
    promoted_from: dict[str, str] = {}
    seeds = parse_seed_list(args.seeds, PROMOTE_NEW_SEEDS)
    for row in selected:
        key = (row["dataset"], row["backbone"], row["variant"], row["hparams_id"], row["hparams_json"])
        source = source_by_key.get(key)
        if source is None:
            continue
        for seed in seeds:
            run_id = make_run_id("stage2_promote", row["dataset"], row["backbone"], row["hparams_id"], seed)
            if run_id in existing_ids:
                continue
            config_path = args.config_root / "stage2_promote" / source["dataset_slug"] / row["backbone"] / f"{run_id}.yaml"
            new_row = copy.deepcopy(source)
            new_row.update(
                {
                    "run_id": run_id,
                    "stage": "stage2_promote",
                    "seed": str(seed),
                    "seeds": str(seed),
                    "config": str(config_path),
                    "output_dir": str(args.run_root),
                    "job_name": make_job_name("stage2_promote", row["dataset"], row["backbone"], row["hparams_id"], seed),
                    "status": "planned",
                    "notes": f"promoted_from={source['run_id']}; stage1_mean_bacc={row['mean_balanced_accuracy']}",
                }
            )
            new_rows.append(new_row)
            promoted_from[run_id] = source["run_id"]
            existing_ids.add(run_id)
    args._promoted_from = promoted_from
    return new_rows


def cmd_promote(args: argparse.Namespace) -> int:
    existing = read_tsv(args.manifest)
    new_rows = promote_rows(args)
    if not new_rows:
        print("No promotion rows generated.")
        return 0
    write_configs(new_rows, args.manifest, promoted_from=getattr(args, "_promoted_from", {}))
    combined = existing + new_rows
    write_tsv(args.manifest, combined)
    write_readme(args.results_root / "README.md", args.manifest, combined)
    print(f"appended_stage2_rows={len(new_rows)}")
    print(f"manifest={args.manifest}")
    return 0


def row_selected(row: dict[str, str], args: argparse.Namespace) -> bool:
    if args.stage and row["stage"] != args.stage:
        return False
    if args.datasets and row["dataset"] not in set(parse_csv_list(args.datasets, DATASETS.keys(), "dataset")):
        return False
    if args.backbones and row["backbone"] not in set(parse_csv_list(args.backbones, STAGE1_PLAN.keys(), "backbone")):
        return False
    if args.status and row.get("status") != args.status:
        return False
    return True


def cmd_submit(args: argparse.Namespace) -> int:
    rows = read_tsv(args.manifest)
    selected = [(idx, row) for idx, row in enumerate(rows) if row_selected(row, args)]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("No manifest rows selected for submission")

    indices = [idx for idx, _ in selected]
    array = compact_array(indices)
    log_root = args.log_root
    log_root.mkdir(parents=True, exist_ok=True)
    export = (
        "ALL,"
        f"PROJECT_ROOT={PROJECT_ROOT},"
        f"MANIFEST={args.manifest},"
        f"OUT_ROOT={args.run_root},"
        f"PYTHON={args.python},"
        f"STATUS_DIR={args.status_root}"
    )
    cmd = [
        "sbatch",
        f"--array={array}%{args.max_parallel}",
        f"--partition={args.partitions}",
        f"--time={args.time}",
        f"--cpus-per-task={args.cpus}",
        f"--mem={args.mem}",
        "--gres=gpu:1",
        "--job-name=spatial_search",
        f"--output={log_root}/spatial_search_%A_%a.out",
        f"--error={log_root}/spatial_search_%A_%a.err",
        f"--export={export}",
        str(args.runner),
    ]
    print(f"selected_runs={len(selected)}")
    print(" ".join(cmd))
    if args.dry_run:
        for idx, row in selected[:10]:
            print(f"task={idx} run_id={row['run_id']} config={row['config']}")
        if len(selected) > 10:
            print(f"... {len(selected) - 10} more")
        return 0
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    common.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    common.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    common.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)

    gen = sub.add_parser("generate", parents=[common], help="Generate a static stage manifest and per-run configs")
    gen.add_argument("--stage", choices=["stage1_screen"], default="stage1_screen")
    gen.add_argument("--datasets", default="all", help="Comma list of processed dataset names, or all")
    gen.add_argument("--backbones", default="all", help="Comma list of backbones, or all")
    gen.add_argument("--seeds", default=None, help="Comma list of operational seeds; default is stage seed 42")
    gen.add_argument(
        "--cbramod-head",
        choices=["base", "avg_pool", "flatten_mlp"],
        default="base",
        help="Optional CBraMod adaptation-head override recorded in hparams_json.",
    )
    gen.add_argument("--force", action="store_true")
    gen.add_argument("--write-readme", action=argparse.BooleanOptionalAction, default=True)
    gen.set_defaults(func=cmd_generate)

    promote = sub.add_parser("promote", parents=[common], help="Append stage2_promote rows from the leaderboard")
    promote.add_argument("--leaderboard", type=Path, default=RESULTS_ROOT / "leaderboard.tsv")
    promote.add_argument("--top-k", type=int, default=1)
    promote.add_argument("--min-completed", type=int, default=1)
    promote.add_argument("--seeds", default=",".join(map(str, PROMOTE_NEW_SEEDS)))
    promote.set_defaults(func=cmd_promote)

    submit = sub.add_parser("submit", parents=[common], help="Submit selected manifest rows as a Slurm array")
    submit.add_argument("--stage", choices=["stage1_screen", "stage2_promote"], default=None)
    submit.add_argument("--datasets", default=None)
    submit.add_argument("--backbones", default=None)
    submit.add_argument("--status", default="planned")
    submit.add_argument("--limit", type=int, default=None)
    submit.add_argument("--max-parallel", type=int, default=3)
    submit.add_argument("--partitions", default=DEFAULT_PARTITIONS)
    submit.add_argument("--time", default="06:00:00")
    submit.add_argument("--cpus", default="8")
    submit.add_argument("--mem", default="32G")
    submit.add_argument("--python", default=DEFAULT_PYTHON)
    submit.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    submit.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    submit.add_argument("--status-root", type=Path, default=DEFAULT_STATUS_ROOT)
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(func=cmd_submit)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.results_root.mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
