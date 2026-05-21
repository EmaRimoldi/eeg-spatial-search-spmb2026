#!/usr/bin/env python3
"""Summarize EEG-FM-Bench BCIC-2a Table-2-style trainer logs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "results" / "eegfm_bench" / "log" / "baseline"
MODELS = ("biot", "labram", "cbramod")
TARGET_BACC = {
    "biot": 28.11,
    "labram": 29.03,
    "cbramod": 33.71,
}

LINE_RE = re.compile(r"\bbcic_2a/(?P<split>eval|test)\s+(?P<body>.*)$")
KV_RE = re.compile(r"(?P<key>[A-Za-z_]+):\s*(?P<value>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.IGNORECASE)


@dataclass(frozen=True)
class SummaryRow:
    model: str
    run: str
    epoch: int | None
    split: str
    bacc: float | None
    target_bacc: float
    delta: float | None
    acc: float | None
    f1: float | None
    cohen_kappa: float | None
    log_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    parser.add_argument("--all-runs", action="store_true", help="Emit every parsed run instead of the latest run per model.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output file. Defaults to stdout.")
    return parser.parse_args()


def model_from_path(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    for model in MODELS:
        if model in parts:
            return model

    lower_name = path.name.lower()
    for model in MODELS:
        if model in lower_name:
            return model
    return None


def parse_log(path: Path) -> SummaryRow | None:
    model = model_from_path(path)
    if model is None:
        return None

    latest: dict[str, float] | None = None
    latest_epoch: int | None = None
    latest_split = "test"

    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None

    for line in lines:
        match = LINE_RE.search(line)
        if not match or match.group("split") != "test":
            continue

        values = {m.group("key"): float(m.group("value")) for m in KV_RE.finditer(match.group("body"))}
        if "balanced_acc" not in values:
            continue

        epoch_value = values.get("epoch")
        epoch = int(epoch_value) if epoch_value is not None else None
        latest = values
        latest_epoch = epoch
        latest_split = match.group("split")

    if latest is None:
        return None

    bacc = latest.get("balanced_acc")
    bacc_percent = bacc * 100.0 if bacc is not None else None
    target = TARGET_BACC[model]
    delta = bacc_percent - target if bacc_percent is not None else None

    return SummaryRow(
        model=model,
        run=path.parent.name,
        epoch=latest_epoch,
        split=latest_split,
        bacc=bacc_percent,
        target_bacc=target,
        delta=delta,
        acc=latest.get("acc"),
        f1=latest.get("f1"),
        cohen_kappa=latest.get("cohen_kappa"),
        log_path=path,
    )


def collect_rows(log_root: Path, all_runs: bool) -> list[SummaryRow]:
    parsed: list[SummaryRow] = []
    if log_root.exists():
        for path in sorted(log_root.glob("**/*_trainer.log")):
            row = parse_log(path)
            if row is not None:
                parsed.append(row)

    if all_runs:
        return sorted(parsed, key=lambda row: (row.model, row.run))

    latest_by_model: dict[str, SummaryRow] = {}
    for row in parsed:
        current = latest_by_model.get(row.model)
        if current is None:
            latest_by_model[row.model] = row
            continue
        current_mtime = current.log_path.stat().st_mtime if current.log_path else -1
        row_mtime = row.log_path.stat().st_mtime if row.log_path else -1
        if (row_mtime, row.epoch or -1) >= (current_mtime, current.epoch or -1):
            latest_by_model[row.model] = row

    rows: list[SummaryRow] = []
    for model in MODELS:
        if model in latest_by_model:
            rows.append(latest_by_model[model])
        else:
            rows.append(
                SummaryRow(
                    model=model,
                    run="not_found",
                    epoch=None,
                    split="test",
                    bacc=None,
                    target_bacc=TARGET_BACC[model],
                    delta=None,
                    acc=None,
                    f1=None,
                    cohen_kappa=None,
                    log_path=None,
                )
            )
    return rows


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def render_markdown(rows: list[SummaryRow]) -> str:
    header = [
        "| Model | Run | Epoch | Test B-Acc | Target B-Acc | Delta | Test Acc | F1 | Kappa | Log |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    body = []
    for row in rows:
        log = str(row.log_path.relative_to(PROJECT_ROOT)) if row.log_path else ""
        body.append(
            "| {model} | {run} | {epoch} | {bacc} | {target} | {delta} | {acc} | {f1} | {kappa} | {log} |".format(
                model=row.model,
                run=row.run,
                epoch=fmt(row.epoch),
                bacc=fmt(row.bacc),
                target=fmt(row.target_bacc),
                delta=fmt(row.delta),
                acc=fmt(row.acc, 3),
                f1=fmt(row.f1, 3),
                kappa=fmt(row.cohen_kappa, 3),
                log=log,
            )
        )
    return "\n".join(header + body) + "\n"


def render_csv(rows: list[SummaryRow]) -> str:
    import io

    out = io.StringIO()
    writer = csv.DictWriter(
        out,
        fieldnames=[
            "model",
            "run",
            "epoch",
            "test_bacc_percent",
            "target_bacc_percent",
            "delta_percent",
            "test_acc_fraction",
            "test_f1_fraction",
            "test_cohen_kappa",
            "log_path",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "model": row.model,
                "run": row.run,
                "epoch": "" if row.epoch is None else row.epoch,
                "test_bacc_percent": fmt(row.bacc),
                "target_bacc_percent": fmt(row.target_bacc),
                "delta_percent": fmt(row.delta),
                "test_acc_fraction": fmt(row.acc, 6),
                "test_f1_fraction": fmt(row.f1, 6),
                "test_cohen_kappa": fmt(row.cohen_kappa, 6),
                "log_path": "" if row.log_path is None else str(row.log_path),
            }
        )
    return out.getvalue()


def main() -> int:
    args = parse_args()
    rows = collect_rows(args.log_root, args.all_runs)
    rendered = render_markdown(rows) if args.format == "markdown" else render_csv(rows)

    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
