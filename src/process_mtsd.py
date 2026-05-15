import os
import glob
import shutil
from pathlib import Path

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
MAX_IMAGES = 5000

# =========================================================
# CREATE OUTPUT FOLDERS
# =========================================================

OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_ANNOTATION_DIR.mkdir(exist_ok=True)

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

copied_count = 0

missing_annotation = 0

skipped_used = 0

skipped_no_annotation = 0

print("===================================")
print("MÁSOLÁS INDUL")
print("===================================\n")

for image_path in mtsd_image_paths:

    if copied_count >= MAX_IMAGES:
        break

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

    # progress
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