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
from utils.categories_to_keep import CLASS_COUNTS, LABEL_MAP

# CONFIG
IMG_DIR = Path("train")
ANN_DIR = Path("train/annotations")
OUT_DIR = Path("yolo_dataset")
SPLIT = {"train": 0.7, "val": 0.2, "test": 0.1}

def load_dataset(label_map: dict[str, str]) -> dict:
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
            data: dict = json.load(f)
        objects = []
        classes = set()
        for obj in data.get("objects", []):
            label = obj["label"]
            if label not in label_map:
                continue
            label_mod = label_map[label]
            bbox = obj["bbox"]
            objects.append((label_mod, bbox))
            classes.add(label_mod)
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

def stratified_split(dataset: list, class_counts: dict[str, int]) -> dict[str, list]:
    """
    Multi-label aware stratified split with:
    - class balance (normalized L2 loss)
    - split size constraint (hard quota)
    - rare class prioritization
    """
    splits = {"train": [], "val": [], "test": []}
    dist = {k: Counter() for k in splits}
    split_size = {k: 0 for k in splits}

    n = len(dataset)
    target_size = {
        "train": int(n * SPLIT["train"]),
        "val": int(n * SPLIT["val"]),
        "test": n - int(n * SPLIT["train"]) - int(n * SPLIT["val"])
    }

    def compute_score(item, split):
        score = 0.0
        for c in item["classes"]:
            total = class_counts.get(c, 1)
            # current + this assignment
            actual = (dist[split][c] + 1) / total
            target = SPLIT[split]
            # rare classes get higher weight
            weight = 1 / total
            score += weight * abs(actual - target) ** 5
        # split size penalty
        size_ratio = (split_size[split] + 1) / target_size[split]
        score += 0.3 * (size_ratio ** 2)
        return score

    # rare classes come first
    dataset_sorted = sorted(
        dataset,
        key=lambda x: sum(class_counts.get(c, 0) for c in x["classes"])
    )
    for item in dataset_sorted:
        best_split = None
        best_score = float("inf")
        for split in splits:
            # hard constraint: don't exceed target size
            if split_size[split] >= target_size[split]:
                continue
            score = compute_score(item, split)
            if score < best_score:
                best_score = score
                best_split = split

        # fallback
        if best_split is None:
            best_split = min(splits, key=lambda s: split_size[s])
        splits[best_split].append(item)
        split_size[best_split] += 1
        for c in item["classes"]:
            dist[best_split][c] += 1

    print("\nSplit stat:")
    for split, items in splits.items():
        print(f"{split}: {len(items)} képek (target: {target_size[split]})")

    print("\nClass distribution:")
    for c in class_counts:
        total = class_counts[c]
        print(f"\n{c}:")
        for split in splits:
            ratio = dist[split][c] / total if total > 0 else 0
            print(f"  {split}: {ratio:.3f}")
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
                if class_map[label] == 20:
                    print(label)
            with open(lbl_out, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

def create_yaml(class_map: dict[str, int]):
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
    # ensure deterministic order
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
    print(class_map)
    dataset = load_dataset(LABEL_MAP)
    random.shuffle(dataset)
    dataset = sorted(dataset, key=lambda x: x["img"].name)
    print(f"\nHasznált képek: {len(dataset)}")
    splits = stratified_split(dataset, CLASS_COUNTS)
    export(splits, class_map)
    create_yaml(class_map)


if __name__ == "__main__":
    main()
