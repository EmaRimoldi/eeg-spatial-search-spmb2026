"""Audit EEG-FM-Bench preprocessing counts for a processed dataset.

The script reads the EEG-FM-Bench summary files produced by ``preproc.py``:

* ``*_info.csv``: one row per accepted raw recording after metadata/montage checks
* ``*_fs_<fs>_cache_files.csv``: one row per persisted parquet cache file
* optional Hugging Face ``dataset_info.json``: final Arrow split counts

It is intended to make count mismatches traceable without touching the data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_json_labels(series: pd.Series) -> pd.Series:
    return series.apply(lambda value: len(json.loads(value)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--config-name", default="finetune")
    parser.add_argument("--fs", type=int, default=200)
    parser.add_argument("--dataset-info", type=Path, default=None)
    args = parser.parse_args()

    info_csv = args.summary_dir / f"{args.dataset_name}_{args.config_name}_info.csv"
    cache_csv = args.summary_dir / (
        f"{args.dataset_name}_{args.config_name}_fs_{args.fs}_cache_files.csv"
    )

    if not info_csv.exists():
        raise FileNotFoundError(info_csv)
    if not cache_csv.exists():
        raise FileNotFoundError(cache_csv)

    info = pd.read_csv(info_csv)
    cache = pd.read_csv(cache_csv)
    info["n_labels"] = _load_json_labels(info["label"])

    print("Raw recordings accepted by EEG-FM-Bench:")
    print(info.groupby("split").size().rename("files").to_string())
    print()

    print("Labels/events accepted before window persistence:")
    print(info.groupby("split")["n_labels"].sum().rename("labels").to_string())
    print()

    if {"subject", "split"}.issubset(info.columns):
        print("Files by subject and split:")
        print(info.groupby(["split", "subject"]).size().rename("files").to_string())
        print()

    print("Persisted cache examples:")
    print(cache.groupby("split")["cnt"].sum().rename("examples").to_string())
    print()

    if args.dataset_info and args.dataset_info.exists():
        dataset_info = json.loads(args.dataset_info.read_text())
        print("Final Arrow dataset_info counts:")
        for split, meta in dataset_info.get("splits", {}).items():
            print(f"{split}: {meta.get('num_examples')}")


if __name__ == "__main__":
    main()
