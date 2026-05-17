import os
import glob
import shutil
from pathlib import Path
from utils.categories_to_keep import LABEL_MAP
import json

# =========================================================
# SETTINGS
# =========================================================

# MTSD
MTSD_IMAGE_DIR = Path("MTSD_images") / "images"

MTSD_ANNOTATION_DIR = (
    Path("MTSD_annotations")
    / "mtsd_v2_fully_annotated"
    / "annotations"
)

# ORIGINAL DATASET
DATASET_RAW_DIR = Path("dataset_raw")

TRAIN_DIR = DATASET_RAW_DIR / "train"
VAL_DIR = DATASET_RAW_DIR / "val"
TEST_DIR = DATASET_RAW_DIR / "test"

# OUTPUT
OUTPUT_DIR = Path("test_MTSD")

OUTPUT_ANNOTATION_DIR = OUTPUT_DIR / "annotations"

# maximum number of images
MAX_IMAGES = 2000

# =========================================================
# CREATE OUTPUT FOLDERS
# =========================================================

OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_ANNOTATION_DIR.mkdir(exist_ok=True)

CLASS_PRIORITY = {
    "stop": 10,
    "no_entry": 8,
    "pedestrian_crossing": 7,
    "yield": 5,
    "speed_limit": 3
}

# =========================================================
# GET USED IMAGE NAMES
# =========================================================

print("===================================")
print("HASZNÁLT KÉPEK BETÖLTÉSE")
print("===================================\n")

used_images = set()

dataset_splits = [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
]

for split_dir in dataset_splits:

    image_paths = glob.glob(str(split_dir / "*.jpg"))

    for image_path in image_paths:

        image_name = os.path.basename(image_path)

        used_images.add(image_name)

print(f"Talált korábban használt képek: {len(used_images)}\n")

# =========================================================
# GET MTSD IMAGES
# =========================================================

print("===================================")
print("MTSD KÉPEK BETÖLTÉSE")
print("===================================\n")

mtsd_image_paths = glob.glob(
    str(MTSD_IMAGE_DIR / "*.jpg")
)

print(f"Talált MTSD képek: {len(mtsd_image_paths)}\n")

# =========================================================
# PROCESS
# =========================================================

candidate_images = []

missing_annotation = 0

skipped_used = 0

skipped_no_annotation = 0

print("===================================")
print("MÁSOLÁS INDUL")
print("===================================\n")

for image_path in mtsd_image_paths:

    image_name = os.path.basename(image_path)

    image_stem = Path(image_name).stem

    # =====================================================
    # SKIP TRAIN/VAL/TEST IMAGES
    # =====================================================

    if image_name in used_images:

        skipped_used += 1

        continue

    # =====================================================
    # FIND ANNOTATION
    # =====================================================

    annotation_path = (
        MTSD_ANNOTATION_DIR
        / f"{image_stem}.json"
    )

    if not annotation_path.exists():

        skipped_no_annotation += 1

        continue

    # =====================================================
    # LOAD ANNOTATION
    # =====================================================

    with open(annotation_path, "r") as f:

        data = json.load(f)

    score = 0

    relevant_count = 0

    for obj in data.get("objects", []):

        label = obj.get("label")

        if label not in LABEL_MAP:
            continue

        mapped = LABEL_MAP[label]

        score += CLASS_PRIORITY[mapped]

        relevant_count += 1

    # multi-object bonus
    score += relevant_count * 2

    # ignore completely irrelevant images
    if relevant_count == 0:
        continue

    candidate_images.append({
        "image_path": image_path,
        "annotation_path": annotation_path,
        "score": score,
        "relevant_count": relevant_count
    })

print(f"Candidate images: {len(candidate_images)}")

# =========================================================
# SORT BY SCORE
# =========================================================

candidate_images.sort(
    key=lambda x: (
        x["score"],
        x["relevant_count"]
    ),
    reverse=True
)

# =========================================================
# COPY BEST IMAGES
# =========================================================

copied_count = 0

for item in candidate_images:

    if copied_count >= MAX_IMAGES:
        break

    image_path = item["image_path"]

    annotation_path = item["annotation_path"]

    image_name = os.path.basename(image_path)

    image_stem = Path(image_name).stem

    # =====================================================
    # COPY IMAGE
    # =====================================================

    output_image_path = OUTPUT_DIR / image_name

    shutil.copy2(
        image_path,
        output_image_path
    )

    # =====================================================
    # COPY ANNOTATION
    # =====================================================

    output_annotation_path = (
        OUTPUT_ANNOTATION_DIR
        / f"{image_stem}.json"
    )

    shutil.copy2(
        annotation_path,
        output_annotation_path
    )

    copied_count += 1

    if copied_count % 100 == 0:

        print(f"Másolva: {copied_count}")

# =========================================================
# FINAL REPORT
# =========================================================

print("\n===================================")
print("KÉSZ")
print("===================================\n")

print(f"Lemásolt képek:            {copied_count}")

print(f"Skip - train/val/test:     {skipped_used}")

print(f"Skip - nincs annotáció:    {skipped_no_annotation}")

print()

print(f"Output mappa: {OUTPUT_DIR}")