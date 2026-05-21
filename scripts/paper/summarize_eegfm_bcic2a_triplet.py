#!/usr/bin/env python3
"""Summarize EEG-FM-Bench BCIC-2a triplet runs from trainer logs."""

import csv
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "results" / "eegfm_bench" / "log" / "baseline"
MODELS = ("biot", "cbramod", "labram")
PAPER_TARGET_BACC = {
    "biot": 28.11,
    "cbramod": 33.71,
    "labram": 29.03,
}
FIELDS = [
    "model",
    "log_path",
    "final_epoch",
    "final_test_bacc",
    "best_test_epoch",
    "best_test_bacc",
    "best_eval_epoch",
    "test_at_best_eval_bacc",
    "test_at_best_eval_acc",
    "test_at_best_eval_kappa",
    "test_at_best_eval_f1",
    "paper_target_bacc",
    "delta_vs_paper_bacc",
]
METRIC_RE = re.compile(
    r"bcic_2a/(?P<split>eval|test) epoch: (?P<epoch>\d+), "
    r"loss: (?P<loss>[0-9.]+), "
    r"acc: (?P<acc>[0-9.]+), "
    r"balanced_acc: (?P<bacc>[0-9.]+), "
    r"cohen_kappa: (?P<kappa>-?[0-9.]+), "
    r"f1: (?P<f1>[0-9.]+)"
)


def parse_log(path):
    eval_rows = {}
    test_rows = {}
    for line in path.read_text().splitlines():
        match = METRIC_RE.search(line)
        if not match:
            continue
        gd = match.groupdict()
        row = {
            "epoch": int(gd["epoch"]),
            "loss": float(gd["loss"]),
            "acc": float(gd["acc"]),
            "balanced_acc": float(gd["bacc"]),
            "cohen_kappa": float(gd["kappa"]),
            "f1": float(gd["f1"]),
        }
        if gd["split"] == "eval":
            eval_rows[row["epoch"]] = row
        else:
            test_rows[row["epoch"]] = row
    return eval_rows, test_rows


def latest_log_for_model(model):
    model_dir = RUN_ROOT / model
    if not model_dir.exists():
        return None
    candidates = sorted(model_dir.glob("torchrun_*/*.log"))
    return candidates[-1] if candidates else None


def best_epoch(rows):
    return max(rows, key=lambda e: (rows[e]["balanced_acc"], rows[e]["acc"], -rows[e]["loss"]))


def summarize_model(model):
    log_path = latest_log_for_model(model)
    if log_path is None:
        return None

    eval_rows, test_rows = parse_log(log_path)
    if not eval_rows or not test_rows:
        return None

    final_epoch = max(test_rows)
    best_test_epoch = best_epoch(test_rows)
    best_eval_epoch = best_epoch(eval_rows)

    final_test = test_rows[final_epoch]
    best_test = test_rows[best_test_epoch]
    paired_test = test_rows[best_eval_epoch]
    target = PAPER_TARGET_BACC[model]

    return {
        "model": model,
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "final_epoch": final_epoch,
        "final_test_bacc": final_test["balanced_acc"] * 100.0,
        "best_test_epoch": best_test_epoch,
        "best_test_bacc": best_test["balanced_acc"] * 100.0,
        "best_eval_epoch": best_eval_epoch,
        "test_at_best_eval_bacc": paired_test["balanced_acc"] * 100.0,
        "test_at_best_eval_acc": paired_test["acc"] * 100.0,
        "test_at_best_eval_kappa": paired_test["cohen_kappa"] * 100.0,
        "test_at_best_eval_f1": paired_test["f1"] * 100.0,
        "paper_target_bacc": target,
        "delta_vs_paper_bacc": (paired_test["balanced_acc"] * 100.0) - target,
    }


def iter_summaries():
    for model in MODELS:
        row = summarize_model(model)
        if row is not None:
            yield row


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows):
    lines = [
        "| model | test@best-eval B-Acc | paper target | delta | best-test B-Acc | final B-Acc | log |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {tb:.2f} | {target:.2f} | {delta:+.2f} | {bb:.2f} | {fb:.2f} | `{log}` |".format(
                model=row["model"],
                tb=row["test_at_best_eval_bacc"],
                target=row["paper_target_bacc"],
                delta=row["delta_vs_paper_bacc"],
                bb=row["best_test_bacc"],
                fb=row["final_test_bacc"],
                log=row["log_path"],
            )
        )
    return "\n".join(lines)


def main():
    rows = list(iter_summaries())
    if not rows:
        print("No usable triplet logs found under results/eegfm_bench/log/baseline")
        return 1

    output_dir = PROJECT_ROOT / "results" / "eegfm_bench" / "summaries"
    csv_path = output_dir / "bcic2a_triplet_summary.csv"
    md_path = output_dir / "bcic2a_triplet_summary.md"

    write_csv(rows, csv_path)
    md = markdown_table(rows)
    md_path.write_text(md + "\n")

    print(md)
    print("\nWrote: {}".format(os.path.relpath(str(csv_path), str(PROJECT_ROOT))))
    print("Wrote: {}".format(os.path.relpath(str(md_path), str(PROJECT_ROOT))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
