"""
This script splits the data into train, validation, and test datasets.
It keeps only the traffic signs that are necessary for the project:
    - pedestrian cross
    - speed limits
    - stop
    - no entry
    - yield

Creates data.yaml for YOLO training. 
Creates YOLO .txt formats for the bounding boxes and labels.
"""
import json
import random
import shutil
from pathlib import Path
from collections import Counter
from src.utils.categories_to_keep import CLASS_COUNTS

# CONFIG
IMG_DIR = Path("train")
ANN_DIR = Path("train/annotations")
OUT_DIR = Path("yolo_dataset")
SPLIT = {"train": 0.7, "val": 0.2, "test": 0.1}
RARE_THRESHOLD = 10 # necessary?

def load_dataset(class_counts: dict[str, int]) -> dict:
    """
    Groups data into a format that is necessary to create 
    the input files for YOLO training.
    """
    images = {p.stem: p for p in IMG_DIR.glob("*.jpg")}
    jsons = {p.stem: p for p in ANN_DIR.glob("*.json")}
    common = set(images) & set(jsons)
    dataset = []

    for name in common:
        with open(jsons[name]) as f:
            data = json.load(f)
        objects = []
        classes = set()
        for obj in data.get("objects", []):
            label = obj["label"]
            if label not in class_counts:
                continue
            bbox = obj["bbox"]
            objects.append((label, bbox))
            classes.add(label)
        if not objects:
            continue
        dataset.append({
            "img": images[name],
            "width": data["width"],
            "height": data["height"],
            "objects": objects,
            "classes": classes
        })
    return dataset

def stratified_split(
        dataset: dict,
        class_counts: dict[str, int]
    ) -> dict[str, list]:
    """
    Splits data into train, validation, test datasets. 
    Uses a greedy algorithm to achieve the closest split as possible.
    """
    # rare = {c for c, n in class_counts.items() if n < RARE_THRESHOLD}
    splits = {"train": [], "val": [], "test": []}
    dist = {k: Counter() for k in splits}
    # remaining = []

    # rare → train
    # for item in dataset:
    #     if any(c in rare for c in item["classes"]):
    #         splits["train"].append(item)
    #         for c in item["classes"]:
    #             dist["train"][c] += 1
    #     else:
    #         remaining.append(item)

    # greedy -> balanced split
    with open("greedy.txt", 'w', encoding="utf-8") as f:
        for item in dataset: # remaining
            best_split = None
            best_score = float("inf")
            f.write(f"{item["img"]} \n {item["classes"]}\n")
            for split in splits:
                score = 0
                for c in item["classes"]:
                    total = class_counts.get(c, 1)
                    desired = total * SPLIT[split]
                    f.write(f"\t\ttotal: {total}, desired: {desired}\n")
                    score += (dist[split][c] + 1) - desired
                f.write(f"\t{split}: {score}\n")
                if score < best_score:
                    best_score = score
                    best_split = split
                f.write(f"\tbest_split: {best_split}, best_score: {best_score}\n")
            splits[best_split].append(item)
            for c in item["classes"]:
                dist[best_split][c] += 1

    print("\nSplit stat:")
    for split, items in splits.items():
        print(f"{split}: {len(items)} képek")
    return splits

def convert_bbox(bbox: dict[str, float], w: int, h: int) -> tuple[int, ...]:
    """
    Converts the bounding box from the JSONs into 
    a format that YOLO expects.
    """
    xmin = max(0, min(bbox["xmin"], w))
    xmax = max(0, min(bbox["xmax"], w))
    ymin = max(0, min(bbox["ymin"], h))
    ymax = max(0, min(bbox["ymax"], h))
    xc = (xmin + xmax) / 2 / w
    yc = (ymin + ymax) / 2 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return xc, yc, bw, bh

def export(splits: dict[str, list], class_map: dict[str, int]):
    """
    Groups the gathered and split data. 
    Exports them to a folder structure that YOLO expects:
     - images
        - train (.jpg)
        - val   (.jpg)
        - test  (.jpg)
     - labels
        - train (.txt)
        - val   (.txt)
        - test  (.txt)
     - data.yaml
    """
    for split, items in splits.items():
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        for item in items:
            img_out = OUT_DIR / "images" / split / item["img"].name
            lbl_out = OUT_DIR / "labels" / split / (item["img"].stem + ".txt")
            shutil.copy(item["img"], img_out)
            lines = []
            for label, bbox in item["objects"]:
                xc, yc, w, h = convert_bbox(bbox, item["width"], item["height"])
                lines.append(f"{class_map[label]} {xc} {yc} {w} {h}")
            with open(lbl_out, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))


def create_yaml(class_map):
    """
    Creates the config file for YOLO training:
    data.yaml:
        path to dataset
        train path
        val path
        test path

        names
    """
    yaml_path = OUT_DIR / "data.yaml"
    lines = [
        f"path: {OUT_DIR}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:"
    ]
    # determinisztikus sorrend
    for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
        lines.append(f"  {idx}: {name}")
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines))


def main():
    """
    Main entry point for split.
    """
    random.seed(42)
    class_map = {name: i for i, name in enumerate(CLASS_COUNTS.keys())}
    dataset = load_dataset(CLASS_COUNTS)
    random.shuffle(dataset)
    dataset = sorted(dataset, key=lambda x: x["img"].name)
    # for data in dataset:
    #     for key, value in data.items():
    #         print(f"{key}: {value}")

    print(f"\nHasznált képek: {len(dataset)}")
    splits = stratified_split(dataset, CLASS_COUNTS)
    export(splits, class_map)
    create_yaml(class_map)


if __name__ == "__main__":
    main()
