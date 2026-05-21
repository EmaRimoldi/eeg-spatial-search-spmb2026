"""
Build canonical 10-20/10-10 coordinate table from MNE standard montages.

Requires MNE to be installed. Run once during environment setup.

Output: data/metadata/canonical_1020_coordinates.csv
"""

import os
import sys
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "canonical_1020_coordinates.csv")

# Regional labels for standard 10-20 channels
REGION_MAP = {
    "frontal_polar": ["Fp1", "Fp2", "FPz"],
    "frontal": ["F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
                "AF7", "AF5", "AF3", "AFz", "AF4", "AF6", "AF8"],
    "frontocentral": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
                      "FT7", "FT8", "FT9", "FT10"],
    "central": ["C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T7", "T8"],
    "centroparietal": ["CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
                       "TP7", "TP8", "TP9", "TP10"],
    "parietal": ["P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
                 "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8"],
    "occipital": ["O1", "Oz", "O2", "I1", "Iz", "I2"],
    "temporal": ["T7", "T8"],  # also in central
    "reference": ["A1", "A2", "M1", "M2", "Nz", "Iz"],
}

HEMISPHERE_MAP = {}
for ch in ["Fp1", "F7", "F5", "F3", "F1", "AF7", "AF5", "AF3", "FT9", "FT7",
           "FC5", "FC3", "FC1", "T7", "C5", "C3", "C1", "TP9", "TP7",
           "CP5", "CP3", "CP1", "P7", "P5", "P3", "P1", "PO7", "PO5", "PO3",
           "O1", "I1", "A1", "M1"]:
    HEMISPHERE_MAP[ch] = "left"
for ch in ["Fp2", "F8", "F6", "F4", "F2", "AF8", "AF6", "AF4", "FT10", "FT8",
           "FC6", "FC4", "FC2", "T8", "C6", "C4", "C2", "TP10", "TP8",
           "CP6", "CP4", "CP2", "P8", "P6", "P4", "P2", "PO8", "PO6", "PO4",
           "O2", "I2", "A2", "M2"]:
    HEMISPHERE_MAP[ch] = "right"
for ch in ["Fz", "AFz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz", "Iz", "Nz",
           "F1", "F2"]:
    HEMISPHERE_MAP[ch] = "midline"


def get_region(ch_name):
    for region, members in REGION_MAP.items():
        if ch_name in members:
            return region
    return "unknown"


def main():
    try:
        import mne
    except ImportError:
        print("ERROR: MNE is required. Install with: pip install mne")
        sys.exit(1)

    print("Loading MNE standard_1020 montage...")
    mon_1020 = mne.channels.make_standard_montage("standard_1020")
    pos_1020 = mon_1020.get_positions()["ch_pos"]

    print("Loading MNE standard_1005 montage...")
    try:
        mon_1005 = mne.channels.make_standard_montage("standard_1005")
        pos_1005 = mon_1005.get_positions()["ch_pos"]
    except Exception:
        pos_1005 = {}

    # Merge, with 1020 taking priority
    all_positions = {**pos_1005, **pos_1020}

    print(f"Total channels with 3D coordinates: {len(all_positions)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "canonical_channel_name",
            "x_3d", "y_3d", "z_3d",
            "x_2d", "y_2d",
            "region_label",
            "hemisphere",
            "coordinate_source",
            "coordinate_confidence",
            "notes",
        ])

        for ch_name, pos_3d in sorted(all_positions.items()):
            x3, y3, z3 = pos_3d

            # 2D projection: use x, y (azimuthal equidistant projection)
            # Simple approximation: project onto xy-plane normalized
            r = (x3**2 + y3**2 + z3**2) ** 0.5
            if r > 0:
                x2 = x3 / r
                y2 = y3 / r
            else:
                x2, y2 = 0.0, 0.0

            writer.writerow([
                ch_name,
                round(x3, 6), round(y3, 6), round(z3, 6),
                round(x2, 6), round(y2, 6),
                get_region(ch_name),
                HEMISPHERE_MAP.get(ch_name, "unknown"),
                "mne_standard_1020" if ch_name in pos_1020 else "mne_standard_1005",
                1.0,
                "",
            ])

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
