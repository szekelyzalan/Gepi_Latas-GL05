"""
Converts dataset into YOLO format.

Supports:
    --raw
    --modified
    --augmented

Behavior:
    train split source:
        --raw       -> dataset_raw/train
        --modified  -> dataset_modified/train
        --augmented -> dataset_augmented/train

    val/test ALWAYS:
        dataset_raw/val
        dataset_raw/test

Creates:
    yolo_dataset/
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/

        data.yaml
"""
import json
import shutil
import argparse
from collections import Counter
from pathlib import Path
from utils.categories_to_keep import LABEL_MAP

YOLO_CLASSES = {
    "speed_limit": 0,
    "pedestrian_crossing": 1,
    "yield": 2,
    "no_entry": 3,
    "stop": 4,
}

# Config
RAW_DIR = Path("dataset_raw")
MODIFIED_DIR = Path("dataset_modified")
AUGMENTED_DIR = Path("dataset_augmented")
OUT_DIR = Path("yolo_dataset")
VAL_SOURCE = RAW_DIR / "val"
TEST_SOURCE = RAW_DIR / "test"

def parse_args():
    """
    Arguments expected:
        --raw
        --modified
        --augmented
    """
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--raw", action="store_true")
    group.add_argument("--modified", action="store_true")
    group.add_argument("--augmented", action="store_true")
    return parser.parse_args()

def convert_bbox(bbox, w: float, h:float) -> tuple[float, ...]:
    """
    Converts the bounding boxes into YOLO format.
    """
    xmin = max(0, min(bbox["xmin"], w))
    xmax = max(0, min(bbox["xmax"], w))
    ymin = max(0, min(bbox["ymin"], h))
    ymax = max(0, min(bbox["ymax"], h))
    bw = xmax - xmin
    bh = ymax - ymin
    xc = xmin + bw / 2
    yc = ymin + bh / 2
    return (
        xc / w,
        yc / h,
        bw / w,
        bh / h
    )

def export_split(split_name: str, split_dir: Path) -> dict:
    img_dir = split_dir
    ann_dir = split_dir / "annotations"
    out_img_dir = OUT_DIR / "images" / split_name
    out_lbl_dir = OUT_DIR / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(img_dir.glob("*.jpg"))
    exported = 0
    class_counter = Counter()
    object_count = 0
    empty_images = 0
    for img_path in image_paths:
        json_path = ann_dir / f"{img_path.stem}.json"
        if not json_path.exists():
            continue
        with open(json_path, "r") as f:
            data = json.load(f)
        width = data["width"]
        height = data["height"]
        lines = []
        relevant_found = False
        for obj in data.get("objects", []):
            original_label = obj["label"]

            # Keep only relevant labels
            if original_label not in LABEL_MAP:
                continue
            relevant_found = True
            yolo_label = LABEL_MAP[original_label]
            class_counter[yolo_label] += 1
            object_count += 1
            class_id = YOLO_CLASSES[yolo_label]
            xc, yc, bw, bh = convert_bbox(
                obj["bbox"],
                width,
                height
            )
            lines.append(
                f"{class_id} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{bw:.6f} "
                f"{bh:.6f}"
            )
        if not relevant_found:
            empty_images += 1
        shutil.copy(img_path, out_img_dir / img_path.name)

        # Save label file
        txt_path = out_lbl_dir / f"{img_path.stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        exported += 1
    print(
        f"{split_name}: "
        f"{exported} images exported"
    )

    # Return stats
    return {
        "images": exported,
        "objects": object_count,
        "empty": empty_images,
        "classes": class_counter
    }

def create_yaml():
    yaml_path = OUT_DIR / "data.yaml"
    lines = [
        f"path: {OUT_DIR.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
        "  0: speed_limit",
        "  1: pedestrian_crossing",
        "  2: yield",
        "  3: no_entry",
        "  4: stop",
    ]
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines))

def main():
    """
    Main entry point for converting images into YOLO format.
    """
    args = parse_args()
    if args.raw:
        train_source = RAW_DIR / "train"
    elif args.modified:
        train_source = MODIFIED_DIR / "train"
    elif args.augmented:
        train_source = AUGMENTED_DIR / "train"
    else:
        raise ValueError("No dataset selected")
    if not train_source.exists():
        raise FileNotFoundError("Please create the train dataset first!")

    print("\n================================")
    print("YOLO EXPORT")
    print("================================")
    print(f"\nTrain source: {train_source}")
    train_stats = export_split("train", train_source)
    val_stats = export_split("val", VAL_SOURCE)
    test_stats = export_split("test", TEST_SOURCE)
    create_yaml()

    # Report
    print("\n================================")
    print("DATASET SUMMARY")
    print("================================")
    split_stats = {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats
    }
    total_counter = Counter()
    total_images = 0
    total_objects = 0
    total_empty = 0
    for split_name, stats in split_stats.items():
        print(f"\n--- {split_name.upper()} ---")
        print(f"images: {stats['images']}")
        print(f"objects: {stats['objects']}")
        print(f"empty images: {stats['empty']}")
        print("\nclass distribution:")
        for cls in YOLO_CLASSES:
            count = stats["classes"][cls]
            print(f"  {cls}: {count}")
            total_counter[cls] += count
        total_images += stats["images"]
        total_objects += stats["objects"]
        total_empty += stats["empty"]
    print("\n================================")
    print("TOTAL")
    print("================================")
    print(f"\nimages: {total_images}")
    print(f"objects: {total_objects}")
    print(f"empty images: {total_empty}")
    print("\nTOTAL CLASS DISTRIBUTION:")
    for cls in YOLO_CLASSES:
        print(
            f"  {cls}: "
            f"{total_counter[cls]}"
        )
    print("\nDONE")


if __name__ == "__main__":
    main()
