#!/usr/bin/env python3
"""Summarize EEG-FM-Bench Table-2-style sweep logs from Slurm outputs."""

import argparse
import csv
import glob
import math
import os
import re
import statistics
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_SLURM_LOG_DIR = os.path.join(PROJECT_ROOT, "results", "eegfm_bench", "slurm_logs")
DATASETS = {
    "bcic2a": {
        "dataset_key": "bcic_2a",
        "manifest": os.path.join(PROJECT_ROOT, "configs", "eegfm_bench", "generated", "bcic2a_table2_5seed", "manifest.tsv"),
        "targets": {"biot": 28.11, "labram": 29.03, "cbramod": 33.71},
    },
    "physiomi": {
        "dataset_key": "motor_mv_img",
        "manifest": os.path.join(PROJECT_ROOT, "configs", "eegfm_bench", "generated", "physiomi_table2_5seed", "manifest.tsv"),
        "targets": {"biot": 27.38, "labram": 57.27, "cbramod": 56.74},
    },
    "workload": {
        "dataset_key": "workload",
        "manifest": os.path.join(PROJECT_ROOT, "configs", "eegfm_bench", "generated", "workload_table2_5seed", "manifest.tsv"),
        "targets": {"biot": 63.98, "labram": 55.82, "cbramod": 71.94},
    },
    "mimul11": {
        "dataset_key": "mimul_11",
        "manifest": os.path.join(PROJECT_ROOT, "configs", "eegfm_bench", "generated", "mimul11_table2_5seed", "manifest.tsv"),
        "targets": {"biot": 50.44, "labram": 50.39, "cbramod": 45.00},
    },
}
MODELS = ("biot", "cbramod", "labram")
LINE_RE_TEMPLATE = r"\b%s/(?P<split>eval|test) epoch: (?P<epoch>\d+), (?P<body>.*)$"
KV_RE = re.compile(r"(?P<key>[A-Za-z_]+):\s*(?P<value>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--slurm-log-dir", default=DEFAULT_SLURM_LOG_DIR)
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    parser.add_argument("--per-seed", action="store_true", help="Include one row per seed instead of only model aggregates.")
    return parser.parse_args()


def load_manifest(path):
    rows = []
    with open(path, "r") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def pick_log(log_dir, job_name):
    pattern = os.path.join(log_dir, "%s_*.out" % job_name)
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(key=lambda path: os.path.getmtime(path))
    return matches[-1]


def parse_metrics(path, dataset_key):
    line_re = re.compile(LINE_RE_TEMPLATE % re.escape(dataset_key))
    eval_by_epoch = {}
    test_by_epoch = {}

    with open(path, "r", errors="replace") as handle:
        for line in handle:
            match = line_re.search(line)
            if not match:
                continue
            epoch = int(match.group("epoch"))
            values = {}
            for kv in KV_RE.finditer(match.group("body")):
                values[kv.group("key")] = float(kv.group("value"))
            if "balanced_acc" not in values:
                continue
            split = match.group("split")
            if split == "eval":
                eval_by_epoch[epoch] = values
            elif split == "test":
                test_by_epoch[epoch] = values

    if not test_by_epoch:
        return None

    final_epoch = max(test_by_epoch)
    final_test = test_by_epoch[final_epoch].get("balanced_acc")
    best_test = max(row.get("balanced_acc") for row in test_by_epoch.values())

    best_eval_epoch = None
    best_eval = None
    test_at_best_eval = None
    if eval_by_epoch:
        best_eval_epoch = max(sorted(eval_by_epoch), key=lambda epoch: eval_by_epoch[epoch].get("balanced_acc", float("-inf")))
        best_eval = eval_by_epoch[best_eval_epoch].get("balanced_acc")
        if best_eval_epoch in test_by_epoch:
            test_at_best_eval = test_by_epoch[best_eval_epoch].get("balanced_acc")

    return {
        "final_epoch": final_epoch,
        "final_test": final_test,
        "best_test": best_test,
        "best_eval_epoch": best_eval_epoch,
        "best_eval": best_eval,
        "test_at_best_eval": test_at_best_eval,
    }


def fmt_pct(value):
    if value is None:
        return ""
    return "%.2f" % (100.0 * value)


def fmt_num(value):
    if value is None:
        return ""
    return "%.2f" % value


def mean_std_percent(values):
    if not values:
        return (None, None)
    pct = [100.0 * value for value in values]
    mean = statistics.mean(pct)
    std = statistics.stdev(pct) if len(pct) > 1 else 0.0
    return (mean, std)


def collect(dataset, manifest_path, slurm_log_dir):
    dataset_key = DATASETS[dataset]["dataset_key"]
    rows = load_manifest(manifest_path)
    parsed = []
    for row in rows:
        log_path = pick_log(slurm_log_dir, row["job_name"])
        result = {
            "dataset": row["dataset"],
            "model": row["model"],
            "seed": row["seed"],
            "job_name": row["job_name"],
            "log_path": log_path,
            "status": "missing_log" if log_path is None else "ok",
            "final_epoch": None,
            "final_test": None,
            "best_test": None,
            "best_eval_epoch": None,
            "best_eval": None,
            "test_at_best_eval": None,
        }
        if log_path is not None:
            metrics = parse_metrics(log_path, dataset_key)
            if metrics is None:
                result["status"] = "no_metrics"
            else:
                result.update(metrics)
        parsed.append(result)
    return parsed


def summarize_models(dataset, per_seed_rows):
    out = []
    targets = DATASETS[dataset]["targets"]
    for model in MODELS:
        rows = [row for row in per_seed_rows if row["model"] == model and row["test_at_best_eval"] is not None]
        test_at_best_eval = [row["test_at_best_eval"] for row in rows]
        final_test = [row["final_test"] for row in rows if row["final_test"] is not None]
        best_test = [row["best_test"] for row in rows if row["best_test"] is not None]

        main_mean, main_std = mean_std_percent(test_at_best_eval)
        final_mean, final_std = mean_std_percent(final_test)
        best_mean, best_std = mean_std_percent(best_test)
        target = targets.get(model)
        delta = None if main_mean is None or target is None else main_mean - target

        out.append({
            "model": model,
            "n": len(rows),
            "test_at_best_eval_mean": main_mean,
            "test_at_best_eval_std": main_std,
            "paper_target": target,
            "delta": delta,
            "final_test_mean": final_mean,
            "final_test_std": final_std,
            "best_test_mean": best_mean,
            "best_test_std": best_std,
        })
    return out


def render_markdown_aggregate(rows):
    header = [
        "| Model | N | Test@best-eval B-Acc | Paper B-Acc | Delta | Final-test B-Acc | Best-test B-Acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    body = []
    for row in rows:
        main = "" if row["test_at_best_eval_mean"] is None else "%.2f ± %.2f" % (row["test_at_best_eval_mean"], row["test_at_best_eval_std"])
        final = "" if row["final_test_mean"] is None else "%.2f ± %.2f" % (row["final_test_mean"], row["final_test_std"])
        best = "" if row["best_test_mean"] is None else "%.2f ± %.2f" % (row["best_test_mean"], row["best_test_std"])
        body.append(
            "| {model} | {n} | {main} | {target} | {delta} | {final} | {best} |".format(
                model=row["model"],
                n=row["n"],
                main=main,
                target=fmt_num(row["paper_target"]),
                delta=fmt_num(row["delta"]),
                final=final,
                best=best,
            )
        )
    return "\n".join(header + body) + "\n"


def render_markdown_per_seed(rows):
    header = [
        "| Model | Seed | Status | Best eval epoch | Eval B-Acc | Test@best-eval | Final-test | Best-test | Log |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    body = []
    for row in rows:
        log_path = "" if row["log_path"] is None else os.path.relpath(row["log_path"], PROJECT_ROOT)
        body.append(
            "| {model} | {seed} | {status} | {best_eval_epoch} | {best_eval} | {test_at_best_eval} | {final_test} | {best_test} | {log_path} |".format(
                model=row["model"],
                seed=row["seed"],
                status=row["status"],
                best_eval_epoch="" if row["best_eval_epoch"] is None else row["best_eval_epoch"],
                best_eval=fmt_pct(row["best_eval"]),
                test_at_best_eval=fmt_pct(row["test_at_best_eval"]),
                final_test=fmt_pct(row["final_test"]),
                best_test=fmt_pct(row["best_test"]),
                log_path=log_path,
            )
        )
    return "\n".join(header + body) + "\n"


def render_csv(dataset, aggregate_rows, per_seed_rows, include_per_seed):
    writer = csv.writer(sys.stdout)
    writer.writerow(["section", "dataset", "model", "seed", "n", "test_at_best_eval_mean", "test_at_best_eval_std", "paper_target", "delta", "final_test_mean", "final_test_std", "best_test_mean", "best_test_std", "status", "best_eval_epoch", "best_eval", "test_at_best_eval", "final_test", "best_test", "log_path"])
    for row in aggregate_rows:
        writer.writerow([
            "aggregate", dataset, row["model"], "", row["n"],
            fmt_num(row["test_at_best_eval_mean"]), fmt_num(row["test_at_best_eval_std"]), fmt_num(row["paper_target"]), fmt_num(row["delta"]),
            fmt_num(row["final_test_mean"]), fmt_num(row["final_test_std"]), fmt_num(row["best_test_mean"]), fmt_num(row["best_test_std"]),
            "", "", "", "", "", "", "",
        ])
    if include_per_seed:
        for row in per_seed_rows:
            writer.writerow([
                "per_seed", dataset, row["model"], row["seed"], "",
                "", "", "", "", "", "", "", "",
                row["status"], row["best_eval_epoch"], fmt_pct(row["best_eval"]), fmt_pct(row["test_at_best_eval"]), fmt_pct(row["final_test"]), fmt_pct(row["best_test"]),
                "" if row["log_path"] is None else row["log_path"],
            ])


def main():
    args = parse_args()
    manifest_path = args.manifest or DATASETS[args.dataset]["manifest"]
    per_seed_rows = collect(args.dataset, manifest_path, args.slurm_log_dir)
    aggregate_rows = summarize_models(args.dataset, per_seed_rows)

    if args.format == "csv":
        render_csv(args.dataset, aggregate_rows, per_seed_rows, args.per_seed)
        return 0

    sys.stdout.write("# %s\n\n" % args.dataset)
    sys.stdout.write(render_markdown_aggregate(aggregate_rows))
    if args.per_seed:
        sys.stdout.write("\n")
        sys.stdout.write(render_markdown_per_seed(per_seed_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
