#!/usr/bin/env python3
"""Summarize staged EEG-FM-Bench spatial-search status and leaderboard."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "spatial_search"
DEFAULT_MANIFEST = DEFAULT_RESULTS_ROOT / "manifest.tsv"

STATUS_FIELDS = [
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
    "status",
    "balanced_accuracy",
    "accuracy",
    "best_val_balanced_accuracy",
    "best_epoch",
    "run_dir",
    "config",
    "job_name",
    "notes",
]

LEADERBOARD_FIELDS = [
    "rank_in_dataset_backbone",
    "is_best_for_dataset_backbone",
    "dataset",
    "dataset_slug",
    "backbone",
    "variant",
    "hparams_id",
    "hparams_json",
    "stage",
    "n_completed",
    "n_planned",
    "seeds_completed",
    "mean_balanced_accuracy",
    "std_balanced_accuracy",
    "mean_accuracy",
    "std_accuracy",
    "best_run_id",
    "best_seed",
    "best_balanced_accuracy",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r") as handle:
            return json.load(handle)
    except Exception:
        return None


def load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r") as handle:
            return yaml.safe_load(handle)
    except Exception:
        return None


def index_completed_runs(run_roots: set[Path]) -> dict[str, dict[str, Any]]:
    by_run_id: dict[str, dict[str, Any]] = {}
    for run_root in sorted(run_roots):
        if not run_root.exists():
            continue
        for config_path in run_root.glob("*/config_resolved.yaml"):
            config = load_yaml(config_path) or {}
            search = config.get("search") or {}
            run_id = search.get("run_id")
            if not run_id:
                continue
            run_dir = config_path.parent
            metrics = load_json(run_dir / "metrics.json")
            if metrics is None:
                continue
            by_run_id[str(run_id)] = {
                "run_dir": str(run_dir),
                "config_resolved": str(config_path),
                "config": config,
                "metrics": metrics,
            }
    return by_run_id


def metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get("test", {}).get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_status_rows(
    manifest_rows: list[dict[str, str]],
    completed: dict[str, dict[str, Any]],
    status_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        row = dict(manifest_row)
        run_id = row["run_id"]
        status_payload = load_json(status_root / f"{run_id}.json") if status_root.exists() else None
        row["status"] = row.get("status") or "planned"
        if status_payload and status_payload.get("status"):
            row["status"] = status_payload["status"]

        record = completed.get(run_id)
        if record is not None:
            metrics = record["metrics"]
            row["status"] = "completed"
            row["balanced_accuracy"] = metric_value(metrics, "balanced_accuracy")
            row["accuracy"] = metric_value(metrics, "accuracy")
            row["best_val_balanced_accuracy"] = metrics.get("best_val", {}).get("balanced_accuracy")
            row["best_epoch"] = metrics.get("best_val", {}).get("epoch")
            row["run_dir"] = record["run_dir"]
        else:
            row.setdefault("balanced_accuracy", "")
            row.setdefault("accuracy", "")
            row.setdefault("best_val_balanced_accuracy", "")
            row.setdefault("best_epoch", "")
            row.setdefault("run_dir", "")
        rows.append(row)
    return rows


def manifest_row_from_completed(record: dict[str, Any]) -> dict[str, str]:
    config = record.get("config") or {}
    search = config.get("search") or {}
    hparams = search.get("hparams") or {}
    seed = str(search.get("seed", config.get("seed", "")))
    promoted_from = search.get("promoted_from_run_id")
    notes = "completed_not_in_manifest"
    if promoted_from:
        notes += f"; promoted_from={promoted_from}"
    return {
        "run_id": str(search.get("run_id", "")),
        "stage": str(search.get("stage", "")),
        "dataset": str(search.get("dataset", config.get("dataset", ""))),
        "dataset_slug": str(search.get("dataset_slug", search.get("dataset", config.get("dataset", "")))),
        "backbone": str(search.get("backbone", config.get("backbone", ""))),
        "variant": str(search.get("variant", config.get("spatial_variant", ""))),
        "seed": seed,
        "seeds": seed,
        "hparams_id": str(search.get("hparams_id", "")),
        "hparams_json": json.dumps(hparams, sort_keys=True, separators=(",", ":")),
        "status": "completed",
        "config": str(record.get("config_resolved", "")),
        "job_name": str(search.get("run_id", "")),
        "notes": notes,
    }


def append_completed_not_in_manifest(
    manifest_rows: list[dict[str, str]],
    completed: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    known_run_ids = {row.get("run_id") for row in manifest_rows}
    extra_rows: list[dict[str, str]] = []
    for run_id, record in sorted(completed.items()):
        if run_id in known_run_ids:
            continue
        row = manifest_row_from_completed(record)
        if row.get("run_id"):
            extra_rows.append(row)
    return manifest_rows + extra_rows


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_leaderboard(status_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in status_rows:
        key = (
            row["dataset"],
            row["backbone"],
            row["variant"],
            row["hparams_id"],
            row["hparams_json"],
        )
        groups[key].append(row)

    leaderboard: list[dict[str, Any]] = []
    for (dataset, backbone, variant, hparams_id, hparams_json), rows in sorted(groups.items()):
        completed = [row for row in rows if parse_float(row.get("balanced_accuracy")) is not None]
        baccs = [parse_float(row.get("balanced_accuracy")) for row in completed]
        accs = [parse_float(row.get("accuracy")) for row in completed]
        bacc_values = [value for value in baccs if value is not None]
        acc_values = [value for value in accs if value is not None]
        stages = sorted({row["stage"] for row in rows})
        best_row = None
        if completed:
            best_row = max(completed, key=lambda row: parse_float(row.get("balanced_accuracy")) or float("-inf"))

        leaderboard.append(
            {
                "dataset": dataset,
                "dataset_slug": rows[0]["dataset_slug"],
                "backbone": backbone,
                "variant": variant,
                "hparams_id": hparams_id,
                "hparams_json": hparams_json,
                "stage": "mixed" if len(stages) > 1 else stages[0],
                "n_completed": len(completed),
                "n_planned": len(rows),
                "seeds_completed": ",".join(str(row["seed"]) for row in sorted(completed, key=lambda item: int(item["seed"]))),
                "mean_balanced_accuracy": statistics.mean(bacc_values) if bacc_values else "",
                "std_balanced_accuracy": statistics.stdev(bacc_values) if len(bacc_values) > 1 else (0.0 if bacc_values else ""),
                "mean_accuracy": statistics.mean(acc_values) if acc_values else "",
                "std_accuracy": statistics.stdev(acc_values) if len(acc_values) > 1 else (0.0 if acc_values else ""),
                "best_run_id": "" if best_row is None else best_row["run_id"],
                "best_seed": "" if best_row is None else best_row["seed"],
                "best_balanced_accuracy": "" if best_row is None else best_row["balanced_accuracy"],
            }
        )

    by_dataset_backbone: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in leaderboard:
        by_dataset_backbone[(row["dataset"], row["backbone"])].append(row)

    ranked: list[dict[str, Any]] = []
    for _, rows in by_dataset_backbone.items():
        rows.sort(
            key=lambda row: (
                parse_float(row.get("mean_balanced_accuracy")) is None,
                -(parse_float(row.get("mean_balanced_accuracy")) or float("-inf")),
                row["variant"],
                row["hparams_id"],
            )
        )
        for rank, row in enumerate(rows, start=1):
            row["rank_in_dataset_backbone"] = rank if row.get("mean_balanced_accuracy") != "" else ""
            row["is_best_for_dataset_backbone"] = "yes" if rank == 1 and row.get("mean_balanced_accuracy") != "" else ""
            ranked.append(row)

    ranked.sort(
        key=lambda row: (
            row["dataset"],
            row["backbone"],
            parse_float(row.get("mean_balanced_accuracy")) is None,
            -(parse_float(row.get("mean_balanced_accuracy")) or float("-inf")),
            row["variant"],
            row["hparams_id"],
        )
    )
    return ranked


def fmt_pct(value: Any) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return ""
    return f"{100.0 * parsed:.2f}"


def print_leaderboard(rows: list[dict[str, Any]], limit: int) -> None:
    completed = [row for row in rows if parse_float(row.get("mean_balanced_accuracy")) is not None]
    if not completed:
        print("No completed spatial-search runs found yet.")
        return
    print("| Dataset | Backbone | Rank | Variant | HParams | N | Mean B-Acc | Best |")
    print("| --- | --- | ---: | --- | --- | ---: | ---: | --- |")
    for row in completed[:limit]:
        best = "*" if row.get("is_best_for_dataset_backbone") == "yes" else ""
        print(
            "| {dataset} | {backbone} | {rank} | {variant} | {hparams} | {n} | {mean} | {best} |".format(
                dataset=row["dataset"],
                backbone=row["backbone"],
                rank=row["rank_in_dataset_backbone"],
                variant=row["variant"],
                hparams=row["hparams_id"],
                n=row["n_completed"],
                mean=fmt_pct(row["mean_balanced_accuracy"]),
                best=best,
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--status-root", type=Path, default=None)
    parser.add_argument("--status-out", type=Path, default=None)
    parser.add_argument("--leaderboard-out", type=Path, default=None)
    parser.add_argument(
        "--include-completed-not-in-manifest",
        action="store_true",
        help="Also summarize completed run directories whose search.run_id is absent from the manifest.",
    )
    parser.add_argument("--print-limit", type=int, default=30)
    args = parser.parse_args(argv)

    results_root = args.results_root
    manifest = args.manifest
    status_root = args.status_root or results_root / "status"
    status_out = args.status_out or results_root / "run_status.tsv"
    leaderboard_out = args.leaderboard_out or results_root / "leaderboard.tsv"

    manifest_rows = read_tsv(manifest)
    run_roots = {Path(row["output_dir"]) for row in manifest_rows if row.get("output_dir")}
    if args.run_root is not None:
        run_roots.add(args.run_root)

    completed = index_completed_runs(run_roots)
    if args.include_completed_not_in_manifest:
        manifest_rows = append_completed_not_in_manifest(manifest_rows, completed)
    status_rows = build_status_rows(manifest_rows, completed, status_root)
    leaderboard = build_leaderboard(status_rows)

    write_tsv(status_out, status_rows, STATUS_FIELDS)
    write_tsv(leaderboard_out, leaderboard, LEADERBOARD_FIELDS)

    print(f"wrote_status={status_out}")
    print(f"wrote_leaderboard={leaderboard_out}")
    print(f"manifest_rows={len(manifest_rows)} completed_runs={sum(1 for row in status_rows if row['status'] == 'completed')}")
    print_leaderboard(leaderboard, args.print_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
