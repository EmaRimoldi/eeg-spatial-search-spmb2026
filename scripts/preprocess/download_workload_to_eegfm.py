#!/usr/bin/env python3
"""Download and normalize the PhysioNet Workload dataset for EEG-FM-Bench.

Expected EEG-FM-Bench layout:

  <raw-root>/Workload EEGMAT/
    subject-info.csv
    data/
      Subject00_1.edf
      ...

PhysioNet provides a zip archive. This script downloads/resumes that archive,
extracts it under a private subdirectory, and then normalizes the layout above
using symlinks for the EDF files.
"""

import argparse
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple


ARCHIVE_URL = "https://physionet.org/content/eegmat/get-zip/1.0.0/"


def download_archive(url: str, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("wget"):
        subprocess.run(
            [
                "wget",
                "--continue",
                "--output-document",
                str(archive_path),
                url,
            ],
            check=True,
        )
        return

    if shutil.which("curl"):
        subprocess.run(
            [
                "curl",
                "-L",
                "-C",
                "-",
                "-o",
                str(archive_path),
                url,
            ],
            check=True,
        )
        return

    with urllib.request.urlopen(url, timeout=60) as response, archive_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_archive(archive_path: Path, extract_root: Path) -> None:
    existing_edfs = list(extract_root.rglob("Subject*_*.edf")) if extract_root.exists() else []
    if existing_edfs:
        return

    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_root)


def normalize_layout(output_root: Path, extract_root: Path) -> Tuple[int, Path]:
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    subject_info_candidates = sorted(extract_root.rglob("subject-info.csv"))
    if not subject_info_candidates:
        raise FileNotFoundError("subject-info.csv not found after extracting Workload archive")
    subject_info_src = subject_info_candidates[0]
    subject_info_dst = output_root / "subject-info.csv"
    if not subject_info_dst.exists() or subject_info_dst.stat().st_size != subject_info_src.stat().st_size:
        shutil.copy2(subject_info_src, subject_info_dst)

    edf_files = sorted(extract_root.rglob("Subject*_*.edf"))
    if not edf_files:
        raise FileNotFoundError("No Subject*_*.edf files found after extracting Workload archive")

    for edf_path in edf_files:
        link_path = data_dir / edf_path.name
        if link_path.is_symlink() or link_path.exists():
            if link_path.is_symlink() and link_path.resolve() == edf_path.resolve():
                continue
            link_path.unlink()
        os.symlink(edf_path, link_path)

    return len(edf_files), subject_info_dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/eegfm/raw/Workload EEGMAT"),
    )
    parser.add_argument("--archive-url", default=ARCHIVE_URL)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    archive_path = output_root / "eegmat.zip"
    extract_root = output_root / "_download" / "eegmat"

    download_archive(args.archive_url, archive_path)
    extract_archive(archive_path, extract_root)
    edf_count, subject_info = normalize_layout(output_root, extract_root)

    print(f"archive={archive_path}")
    print(f"extract_root={extract_root}")
    print(f"subject_info={subject_info}")
    print(f"edf_files={edf_count}")
    print(f"normalized_data_dir={output_root / 'data'}")


if __name__ == "__main__":
    main()
