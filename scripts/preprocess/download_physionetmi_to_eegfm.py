"""Download PhysioMI / EEG Motor Movement Imagery data for EEG-FM-Bench.

EEG-FM-Bench's ``motor_mv_img`` builder expects PhysioNet EDF files under:

    <raw-root>/Motor Movement Imagery/eeg-motor-movementimagery-dataset-1.0.0/files/

MNE downloads the same files under an ``MNE-eegbci-data/files/eegmmidb/1.0.0``
subtree. This script downloads the runs used by EEG-FM-Bench and creates the
``files`` symlink expected by the builder.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mne.datasets import eegbci


TASK_RUNS = [4, 6, 8, 10, 12, 14]
BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0"


def parse_subjects(value: str) -> list[int]:
    if value == "all":
        return list(range(1, 110))
    subjects: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            subjects.extend(range(int(start), int(end) + 1))
        else:
            subjects.append(int(part))
    return sorted(set(subjects))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/eegfm/raw/Motor Movement Imagery/eeg-motor-movementimagery-dataset-1.0.0"),
    )
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--runs", default=",".join(str(run) for run in TASK_RUNS))
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--method", choices=["direct", "mne"], default="direct")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    subjects = parse_subjects(args.subjects)
    runs = [int(run) for run in args.runs.split(",")]
    mne_files_root = output_root / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"

    if args.method == "mne":
        paths = eegbci.load_data(
            subjects,
            runs,
            path=str(output_root),
            update_path=False,
            verbose=True,
        )
    else:
        paths = download_direct(subjects, runs, mne_files_root, args.jobs)

    eegfm_files_root = output_root / "files"

    if eegfm_files_root.exists() or eegfm_files_root.is_symlink():
        if not eegfm_files_root.is_symlink() or eegfm_files_root.resolve() != mne_files_root:
            raise FileExistsError(
                f"{eegfm_files_root} exists and does not point to {mne_files_root}"
            )
    else:
        os.symlink(mne_files_root, eegfm_files_root)

    print(f"Downloaded/verified {len(paths)} EDF files.")
    print(f"EEG-FM-Bench files path: {eegfm_files_root}")


def download_direct(subjects: list[int], runs: list[int], output_root: Path, jobs: int) -> list[Path]:
    tasks: list[tuple[str, Path]] = []
    for subject in subjects:
        subject_name = f"S{subject:03d}"
        for run in runs:
            filename = f"{subject_name}R{run:02d}.edf"
            url = f"{BASE_URL}/{subject_name}/{filename}"
            target = output_root / subject_name / filename
            tasks.append((url, target))

    output_root.mkdir(parents=True, exist_ok=True)
    completed: list[Path] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(fetch_one, url, target) for url, target in tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            completed.append(future.result())
            if idx % 10 == 0 or idx == len(futures):
                print(f"Downloaded/verified {idx}/{len(futures)} EDF files", flush=True)
    return completed


def fetch_one(url: str, target: Path) -> Path:
    if target.exists() and target.stat().st_size > 0:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    for attempt in range(3):
        try:
            if shutil.which("curl"):
                subprocess.run(
                    [
                        "curl",
                        "-L",
                        "--fail",
                        "--retry",
                        "2",
                        "--connect-timeout",
                        "15",
                        "--max-time",
                        "120",
                        "--silent",
                        "--show-error",
                        "-o",
                        str(tmp),
                        url,
                    ],
                    check=True,
                )
            else:
                with urllib.request.urlopen(url, timeout=30) as response, tmp.open("wb") as out:
                    shutil.copyfileobj(response, out)
            tmp.replace(target)
            return target
        except Exception:
            if tmp.exists():
                tmp.unlink()
            if attempt == 2:
                raise
    return target


if __name__ == "__main__":
    main()
