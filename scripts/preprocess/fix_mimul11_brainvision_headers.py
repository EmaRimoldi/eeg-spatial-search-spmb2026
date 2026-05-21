#!/usr/bin/env python3
"""Normalize Mimul-11 BrainVision header references.

The upstream RawData archive contains a subset of ``.vhdr`` files whose
``DataFile=`` / ``MarkerFile=`` entries do not point at the sibling files with
the same basename. EEG-FM-Bench's Mimul-11 loader expects standard BrainVision
triplets, so we rewrite those two header fields to:

    DataFile=<basename>.eeg
    MarkerFile=<basename>.vmrk

Only files that already have matching sibling ``.eeg`` and ``.vmrk`` files are
modified.
"""

import argparse
import os
import re
import sys


DATA_RE = re.compile(r"(?m)^DataFile=.*$")
MARKER_RE = re.compile(r"(?m)^MarkerFile=.*$")


def rewrite_header(path):
    base, _ = os.path.splitext(path)
    eeg_name = os.path.basename(base + ".eeg")
    vmrk_name = os.path.basename(base + ".vmrk")

    if not os.path.exists(base + ".eeg"):
        return ("missing_eeg", path)
    if not os.path.exists(base + ".vmrk"):
        return ("missing_vmrk", path)

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    new_text = DATA_RE.sub("DataFile=%s" % eeg_name, text)
    new_text = MARKER_RE.sub("MarkerFile=%s" % vmrk_name, new_text)

    if new_text == text:
        return ("unchanged", path)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(new_text)
    return ("fixed", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root directory containing RawData .vhdr files")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit("missing directory: %s" % root)

    counts = {"fixed": 0, "unchanged": 0, "missing_eeg": 0, "missing_vmrk": 0}
    missing = []

    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if not filename.endswith(".vhdr"):
                continue
            status, path = rewrite_header(os.path.join(dirpath, filename))
            counts[status] = counts.get(status, 0) + 1
            if status.startswith("missing_"):
                missing.append((status, path))

    print("Mimul-11 BrainVision header normalization summary:")
    for key in ("fixed", "unchanged", "missing_eeg", "missing_vmrk"):
        print("  %s: %s" % (key, counts.get(key, 0)))

    if missing:
        print("Missing sidecar files:")
        for status, path in missing:
            print("  %s\t%s" % (status, path))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
