#!/usr/bin/env python
"""Export MOABB BNCI2014_001 runs to EEG-FM-Bench BCIC-2a EEGLAB layout.

EEG-FM-Bench's ``bcic_2a`` builder expects ``.set`` files under:

    <raw_root>/BCI Competition IV/2a/set/

with filenames whose fourth character marks session type (``T``/``E``) and
whose final character is an integer session/run id, e.g. ``A01T0.set``.

MOABB exposes BNCI2014_001 as subjects -> sessions -> six runs, so this script
exports each run as one EEGLAB file and maps MOABB event names to the labels
expected by EEG-FM-Bench's BCIC-2a builder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
from moabb.datasets import BNCI2014_001


LABEL_MAP = {
    "left_hand": "left",
    "right_hand": "right",
    "feet": "foot",
    "tongue": "tongue",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "data/eegfm/raw/BCI Competition IV/2a/set"
        ),
        help="Directory where EEG-FM-Bench should find BCIC-2a .set files.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        type=int,
        default=list(range(1, 10)),
        help="BNCI2014_001 subject IDs to export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .set files.",
    )
    return parser.parse_args()


def session_code(session_name: str) -> str:
    name = session_name.lower()
    if "train" in name:
        return "T"
    if "test" in name or "eval" in name:
        return "E"
    raise ValueError(f"Cannot infer BCIC session code from session name: {session_name}")


def normalize_annotations(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    annotations = raw.annotations
    descriptions = [LABEL_MAP.get(str(desc), str(desc)) for desc in annotations.description]
    raw.set_annotations(
        mne.Annotations(
            onset=annotations.onset,
            duration=annotations.duration,
            description=descriptions,
            orig_time=annotations.orig_time,
        )
    )
    return raw


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = BNCI2014_001()
    exported = 0
    skipped = 0

    for subject in args.subjects:
        subject_data = dataset.get_data(subjects=[subject])[subject]
        for session_name, runs in subject_data.items():
            code = session_code(session_name)
            for run_name, raw in sorted(runs.items(), key=lambda kv: int(kv[0])):
                run_id = int(run_name)
                out_file = output_dir / f"A{subject:02d}{code}{run_id}.set"
                if out_file.exists() and not args.overwrite:
                    skipped += 1
                    continue

                raw = normalize_annotations(raw.copy())
                raw.export(out_file, fmt="eeglab", overwrite=True)
                exported += 1
                print(f"exported {out_file}")

    print(f"done: exported={exported} skipped={skipped} output_dir={output_dir}")


if __name__ == "__main__":
    main()
