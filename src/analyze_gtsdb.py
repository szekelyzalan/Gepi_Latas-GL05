"""
analyze_gtsdb.py

Goal:
- Analyze GTSDB dataset
- Count all images
- Count all annotations
- Count all objects
- Count EXACT classTitle occurrences

Dataset structure:
gtsdb-DatasetNinja/
    train/
        ann/*.json
        img/*.png

    test/
        ann/*.json
        img/*.png
"""

from pathlib import Path
import json
from collections import Counter

# =========================================================
# CONFIG
# =========================================================

ROOT_DIR = Path("gtsdb_extracted")

SPLITS = [
    "train",
    "test"
]

# =========================================================
# ANALYZE SPLIT
# =========================================================

def analyze_split(split_name: str):

    split_dir = ROOT_DIR / split_name

    ann_dir = split_dir / "ann"

    img_dir = split_dir / "img"

    print("\n================================")
    print(f"SPLIT: {split_name}")
    print("================================")

    # =====================================================
    # FILE COUNTS
    # =====================================================

    image_files = list(img_dir.glob("*"))

    json_files = list(ann_dir.glob("*.json"))

    print(f"\nImages:         {len(image_files)}")

    print(f"Annotations:    {len(json_files)}")

    # =====================================================
    # STATISTICS
    # =====================================================

    total_objects = 0

    malformed_json = 0

    empty_annotations = 0

    class_counter = Counter()

    # =====================================================
    # PROCESS JSONS
    # =====================================================

    for json_path in sorted(json_files):

        try:

            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

        except Exception as e:

            malformed_json += 1

            print(
                f"[JSON ERROR] "
                f"{json_path.name}: {e}"
            )

            continue

        objects = data.get("objects", [])

        if len(objects) == 0:

            empty_annotations += 1

        total_objects += len(objects)

        # =================================================
        # COUNT LABELS
        # =================================================

        for obj in objects:

            label = obj.get("classTitle")

            if label is None:

                label = "UNKNOWN"

            class_counter[label] += 1

    # =====================================================
    # REPORT
    # =====================================================

    print()

    print(
        f"Malformed JSON files: "
        f"{malformed_json}"
    )

    print(
        f"Empty annotations:    "
        f"{empty_annotations}"
    )

    print(
        f"Total objects:        "
        f"{total_objects}"
    )

    print()

    print("================================")
    print("LABEL DISTRIBUTION")
    print("================================\n")

    for label, count in class_counter.most_common():

        print(f"{label}: {count}")

# =========================================================
# MAIN
# =========================================================

def main():

    print("\n================================")
    print("GTSDB DATASET ANALYSIS")
    print("================================")

    for split in SPLITS:

        analyze_split(split)

# =========================================================

if __name__ == "__main__":

    main()