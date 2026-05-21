#!/usr/bin/env python3
"""Summarize 5-seed EEG-FM-Bench BCIC-2a Table-2-style runs."""

import csv
import glob
import os
import re
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "results" / "eegfm_bench" / "slurm_logs"
MODELS = ("biot", "cbramod", "labram")
SEEDS = (42, 43, 44, 45, 46)
PAPER_TARGET_BACC = {
    "biot": 28.11,
    "cbramod": 33.71,
    "labram": 29.03,
}
METRIC_RE = re.compile(
    r"bcic_2a/(?P<split>eval|test) epoch: (?P<epoch>\d+), "
    r"loss: (?P<loss>[0-9.]+), "
    r"acc: (?P<acc>[0-9.]+), "
    r"balanced_acc: (?P<bacc>[0-9.]+), "
    r"cohen_kappa: (?P<kappa>-?[0-9.]+), "
    r"f1: (?P<f1>[0-9.]+)"
)
RUN_RE = re.compile(r"efm_(?P<model>[^_]+)_bcic2a_t2_s(?P<seed>\d+)_(?P<jobid>\d+)\.out$")
PER_RUN_FIELDS = [
    "model",
    "seed",
    "jobid",
    "slurm_out",
    "final_epoch",
    "final_test_bacc",
    "best_test_epoch",
    "best_test_bacc",
    "best_eval_epoch",
    "test_at_best_eval_bacc",
    "test_at_best_eval_acc",
    "test_at_best_eval_kappa",
    "test_at_best_eval_f1",
]
AGG_FIELDS = [
    "model",
    "n_completed",
    "mean_test_at_best_eval_bacc",
    "std_test_at_best_eval_bacc",
    "mean_best_test_bacc",
    "std_best_test_bacc",
    "paper_target_bacc",
    "delta_mean_vs_paper_bacc",
]


def parse_metrics(path):
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


def best_epoch(rows):
    return max(rows, key=lambda e: (rows[e]["balanced_acc"], rows[e]["acc"], -rows[e]["loss"]))


def latest_out_for(model, seed):
    pattern = str(LOG_DIR / "{}_{}_{}".format("efm", model, "bcic2a_t2_s{}_*".format(seed))) + ".out"
    candidates = [Path(p) for p in glob.glob(pattern)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return candidates[-1]


def summarize_run(model, seed):
    path = latest_out_for(model, seed)
    if path is None:
        return None

    match = RUN_RE.search(path.name)
    jobid = match.group("jobid") if match else ""

    eval_rows, test_rows = parse_metrics(path)
    if not eval_rows or not test_rows:
        return {
            "model": model,
            "seed": seed,
            "jobid": jobid,
            "slurm_out": os.path.relpath(str(path), str(PROJECT_ROOT)),
            "final_epoch": "",
            "final_test_bacc": "",
            "best_test_epoch": "",
            "best_test_bacc": "",
            "best_eval_epoch": "",
            "test_at_best_eval_bacc": "",
            "test_at_best_eval_acc": "",
            "test_at_best_eval_kappa": "",
            "test_at_best_eval_f1": "",
        }

    final_epoch = max(test_rows)
    best_test_epoch = best_epoch(test_rows)
    best_eval_epoch = best_epoch(eval_rows)
    final_test = test_rows[final_epoch]
    best_test = test_rows[best_test_epoch]
    paired_test = test_rows[best_eval_epoch]

    return {
        "model": model,
        "seed": seed,
        "jobid": jobid,
        "slurm_out": os.path.relpath(str(path), str(PROJECT_ROOT)),
        "final_epoch": final_epoch,
        "final_test_bacc": final_test["balanced_acc"] * 100.0,
        "best_test_epoch": best_test_epoch,
        "best_test_bacc": best_test["balanced_acc"] * 100.0,
        "best_eval_epoch": best_eval_epoch,
        "test_at_best_eval_bacc": paired_test["balanced_acc"] * 100.0,
        "test_at_best_eval_acc": paired_test["acc"] * 100.0,
        "test_at_best_eval_kappa": paired_test["cohen_kappa"] * 100.0,
        "test_at_best_eval_f1": paired_test["f1"] * 100.0,
    }


def summarize_all_runs():
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            row = summarize_run(model, seed)
            if row is not None:
                rows.append(row)
    return rows


def aggregate(rows):
    out = []
    for model in MODELS:
        model_rows = [r for r in rows if r["model"] == model and r["test_at_best_eval_bacc"] != ""]
        baccs = [r["test_at_best_eval_bacc"] for r in model_rows]
        bests = [r["best_test_bacc"] for r in model_rows]
        if not baccs:
            out.append({
                "model": model,
                "n_completed": 0,
                "mean_test_at_best_eval_bacc": "",
                "std_test_at_best_eval_bacc": "",
                "mean_best_test_bacc": "",
                "std_best_test_bacc": "",
                "paper_target_bacc": PAPER_TARGET_BACC[model],
                "delta_mean_vs_paper_bacc": "",
            })
            continue
        mean_bacc = statistics.mean(baccs)
        std_bacc = statistics.stdev(baccs) if len(baccs) >= 2 else 0.0
        mean_best = statistics.mean(bests)
        std_best = statistics.stdev(bests) if len(bests) >= 2 else 0.0
        out.append({
            "model": model,
            "n_completed": len(baccs),
            "mean_test_at_best_eval_bacc": mean_bacc,
            "std_test_at_best_eval_bacc": std_bacc,
            "mean_best_test_bacc": mean_best,
            "std_best_test_bacc": std_best,
            "paper_target_bacc": PAPER_TARGET_BACC[model],
            "delta_mean_vs_paper_bacc": mean_bacc - PAPER_TARGET_BACC[model],
        })
    return out


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_num(x):
    if x == "":
        return ""
    return "{:.2f}".format(x)


def markdown_aggregate(rows):
    lines = [
        "| model | n | mean test@best-eval B-Acc | std | paper target | delta | mean best-test B-Acc | std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {n} | {mean} | {std} | {target:.2f} | {delta} | {mean_best} | {std_best} |".format(
                model=row["model"],
                n=row["n_completed"],
                mean=format_num(row["mean_test_at_best_eval_bacc"]),
                std=format_num(row["std_test_at_best_eval_bacc"]),
                target=row["paper_target_bacc"],
                delta=("{:+.2f}".format(row["delta_mean_vs_paper_bacc"]) if row["delta_mean_vs_paper_bacc"] != "" else ""),
                mean_best=format_num(row["mean_best_test_bacc"]),
                std_best=format_num(row["std_best_test_bacc"]),
            )
        )
    return "\n".join(lines)


def main():
    per_run = summarize_all_runs()
    if not per_run:
        print("No seed-sweep runs found under results/eegfm_bench/slurm_logs")
        return 1

    aggregated = aggregate(per_run)
    out_dir = PROJECT_ROOT / "results" / "eegfm_bench" / "summaries"
    per_run_csv = out_dir / "bcic2a_5seed_per_run.csv"
    agg_csv = out_dir / "bcic2a_5seed_aggregate.csv"
    agg_md = out_dir / "bcic2a_5seed_aggregate.md"

    write_csv(per_run_csv, PER_RUN_FIELDS, per_run)
    write_csv(agg_csv, AGG_FIELDS, aggregated)
    md = markdown_aggregate(aggregated)
    agg_md.write_text(md + "\n")

    print(md)
    print("\nWrote: {}".format(os.path.relpath(str(per_run_csv), str(PROJECT_ROOT))))
    print("Wrote: {}".format(os.path.relpath(str(agg_csv), str(PROJECT_ROOT))))
    print("Wrote: {}".format(os.path.relpath(str(agg_md), str(PROJECT_ROOT))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
