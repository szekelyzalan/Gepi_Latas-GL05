"""
convert_gtsdb.py

Goal:
- Convert GTSDB annotations into
  MTSD-like format

- Keep ALL images

- Keep ONLY the target labels
  defined in LABEL_MAP_GTSDB

Output:
gtsdb_train/
    *.png
    annotations/
        *.json
"""

from pathlib import Path

import json
import shutil
import uuid

# =========================================================
# CONFIG
# =========================================================

INPUT_DIR = Path(
    "gtsdb_extracted/train"
)

INPUT_IMG_DIR = INPUT_DIR / "img"

INPUT_ANN_DIR = INPUT_DIR / "ann"

OUTPUT_DIR = Path("gtsdb_train")

OUTPUT_ANN_DIR = (
    OUTPUT_DIR / "annotations"
)

VALID_IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg"
]

# =========================================================
# LABEL MAP
# =========================================================

LABEL_MAP_GTSDB: dict[str, str] = {
    "speed limit 100": "speed_limit",
    "speed limit 120": "speed_limit",
    "speed limit 20": "speed_limit",
    "speed limit 30": "speed_limit",
    "speed limit 50": "speed_limit",
    "speed limit 60": "speed_limit",
    "speed limit 70": "speed_limit",
    "speed limit 80": "speed_limit",

    "pedestrian crossing":
        "pedestrian_crossing",

    "give way":
        "yield",

    "stop":
        "stop",

    "no entry":
        "no_entry"
}

# =========================================================
# HELPERS
# =========================================================

def find_image(stem: str) -> Path | None:
    """
    Find image corresponding
    to annotation stem.
    """

    for ext in VALID_IMAGE_EXTENSIONS:

        candidate = (
            INPUT_IMG_DIR /
            f"{stem}{ext}"
        )

        if candidate.exists():

            return candidate

    return None

# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_ANN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    json_files = sorted(
        INPUT_ANN_DIR.glob("*.json")
    )

    total_images = 0

    total_annotations = 0

    total_objects_before = 0

    total_objects_after = 0

    skipped_images = 0

    # per-class stats
    class_counter = {}

    for json_path in json_files:

        stem = json_path.stem.split('.')[0]

        # =================================================
        # IMAGE
        # =================================================

        image_path = find_image(stem)

        if image_path is None:

            skipped_images += 1

            print(
                f"[WARN] Missing image: "
                f"{stem}"
            )

            continue

        total_images += 1

        # copy image
        shutil.copy2(
            image_path,
            OUTPUT_DIR / image_path.name
        )

        # =================================================
        # LOAD JSON
        # =================================================

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        width = data["size"]["width"]

        height = data["size"]["height"]

        objects = data.get("objects", [])

        total_objects_before += len(objects)

        converted_objects = []

        # =================================================
        # OBJECTS
        # =================================================

        for obj in objects:

            original_label = obj.get(
                "classTitle"
            )

            # keep only relevant labels
            if original_label not in LABEL_MAP_GTSDB:
                continue

            new_label = LABEL_MAP_GTSDB[
                original_label
            ]

            points = obj["points"]["exterior"]

            xmin = float(points[0][0])
            ymin = float(points[0][1])

            xmax = float(points[1][0])
            ymax = float(points[1][1])

            new_obj = {
                "key": str(uuid.uuid4())[:24],

                "label": new_label,

                "bbox": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                }
            }

            converted_objects.append(
                new_obj
            )

            total_objects_after += 1

            class_counter[new_label] = (
                class_counter.get(
                    new_label,
                    0
                ) + 1
            )

        # =================================================
        # CREATE NEW JSON
        # =================================================

        new_data = {
            "width": width,

            "height": height,

            "ispano": False,

            "objects": converted_objects
        }

        output_json_path = (
            OUTPUT_ANN_DIR /
            f"{json_path.name.split('.')[0]}.json"
        )

        with open(
            output_json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                new_data,
                f,
                indent=2,
                ensure_ascii=False
            )

        total_annotations += 1

    # =====================================================
    # REPORT
    # =====================================================

    print("\n================================")
    print("GTSDB CONVERSION DONE")
    print("================================")

    print(
        f"\nImages processed:      "
        f"{total_images}"
    )

    print(
        f"Annotations created:   "
        f"{total_annotations}"
    )

    print(
        f"Skipped images:        "
        f"{skipped_images}"
    )

    print()

    print(
        f"Original objects:      "
        f"{total_objects_before}"
    )

    print(
        f"Kept objects:          "
        f"{total_objects_after}"
    )

    print("\n================================")
    print("CLASS DISTRIBUTION")
    print("================================\n")

    for label, count in sorted(
        class_counter.items()
    ):

        print(f"{label}: {count}")

    print("\nOutput:")

    print(OUTPUT_DIR)

# =========================================================

if __name__ == "__main__":

    main()