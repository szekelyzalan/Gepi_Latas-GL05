"""
prepare_test_dataset.py

Goal:
- Read dataset_raw or a custom folder
- iterate over the annotation JSON files
- Keep only the labels used for training
- Change the labels according to LABEL_MAP
- Copy the images and the new annotations to the test folder

Output structure:
test/
    *.jpg
    annotations/
        *.json
"""
from pathlib import Path
import json
import shutil
from utils.categories_to_keep import LABEL_MAP

# Config
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "dataset_raw" / "test"
OUTPUT_DIR = PROJECT_ROOT / "test"

INPUT_ANNOTATIONS = INPUT_DIR / "annotations"
OUTPUT_ANNOTATIONS = OUTPUT_DIR / "annotations"

# Helpers
VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def find_image_for_annotation(annotation_path: Path) -> Path | None:
    """
    Find the image for an anootation.
    """
    stem = annotation_path.stem
    for ext in VALID_IMAGE_EXTENSIONS:
        candidate = INPUT_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    """
    Main entry point for preparing the test dataset.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ANNOTATIONS.mkdir(parents=True, exist_ok=True)

    json_files = sorted(INPUT_ANNOTATIONS.glob("*.json"))

    total_files = 0
    total_objects_before = 0
    total_objects_after = 0
    skipped_images = 0

    for json_path in json_files:

        total_files += 1

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        objects = data.get("objects", [])
        total_objects_before += len(objects)

        filtered_objects = []

        for obj in objects:
            original_label = obj.get("label")
            # Keep only the classes that the model is trained on
            if original_label not in LABEL_MAP:
                continue
            new_obj = obj.copy()
            # Change label
            new_obj["label"] = LABEL_MAP[original_label]
            filtered_objects.append(new_obj)
        total_objects_after += len(filtered_objects)

        # New JSON
        new_data = data.copy()
        new_data["objects"] = filtered_objects
        # Save annotation
        output_json_path = OUTPUT_ANNOTATIONS / json_path.name
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
        # Copy image
        image_path = find_image_for_annotation(json_path)
        if image_path is None:
            print(f"[WARN] No annotation for the image: {json_path.name}")
            skipped_images += 1
            continue

        output_image_path = OUTPUT_DIR / image_path.name
        shutil.copy2(image_path, output_image_path)

    # Report
    print("\n=== TEST DATASET PREPARED ===")
    print(f"Processed annotations: {total_files}")
    print(f"Number of original objects: {total_objects_before}")
    print(f"Number of keeped objects: {total_objects_after}")
    print(f"Skipped images: {skipped_images}")

    print(f"\nOutput:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
