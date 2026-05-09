"""
RAW DATASET SPLIT

Pipeline step #1

Goal:
- Split ORIGINAL dataset into:
    train / val / test

- Preserve target-class distribution
  according to LABEL_MAP

- Keep ORIGINAL json annotations unchanged

Important:
- Images WITHOUT relevant classes
  go automatically into TRAIN

- Validation and test contain ONLY
  images with relevant target classes

Output:
dataset_raw/
    train/
        *.jpg
        annotations/*.json

    val/
        *.jpg
        annotations/*.json

    test/
        *.jpg
        annotations/*.json
"""
import json
import random
import shutil
from pathlib import Path
from collections import Counter
from utils.categories_to_keep import LABEL_MAP

# =========================================================
# CONFIG
# =========================================================
IMG_DIR = Path("train")
ANN_DIR = IMG_DIR / "annotations"
OUT_DIR = Path("dataset_raw")
SPLIT = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}
RANDOM_SEED = 42

def load_dataset() -> dict:
    """
    Goes through the images and annotations inside the train folder. 
    Returns the extracted data in dictionary, containing the relevant 
    data for train/val/test split.
    """
    images = {p.stem: p for p in IMG_DIR.glob("*.jpg")}
    jsons = {p.stem: p for p in ANN_DIR.glob("*.json")}
    common = sorted(set(images) & set(jsons))
    dataset = []
    for name in common:
        with open(jsons[name], "r") as f:
            data: dict = json.load(f)
        target_classes = set()
        for obj in data.get("objects", []):
            label = obj["label"]
            if label in LABEL_MAP:
                target_classes.add(LABEL_MAP[label])
        dataset.append({
            "img": images[name],
            "json": jsons[name],
            "target_classes": target_classes,
            "has_target": len(target_classes) > 0
        })
    return dataset

def compute_class_counts(dataset: dict) -> dict:
    """
    Counts the number of occurances of the target 
    classes in the whole datasdet.
    """
    counts = Counter()
    for item in dataset:
        for c in item["target_classes"]:
            counts[c] += 1
    return counts

def stratified_split(dataset: dict) -> dict[str, list]:
    """
    Implements a greedy algorithm for finding the best 
    train/val/test split. 
    The goal is to obtain a split that contains the labels in the 
    same distribution as the predefined split ration, while maintaining 
    the ratio between the number of images.
    """
    splits = {
        "train": [],
        "val": [],
        "test": []
    }
    split_dist = {k: Counter() for k in splits}
    split_sizes = {k: 0 for k in splits}

    target_items = [x for x in dataset if x["has_target"]]
    non_target_items = [x for x in dataset if not x["has_target"]]
    class_counts = compute_class_counts(target_items)

    # Split sizes
    # ONLY useful images count here
    n = len(target_items)
    target_size = {
        "train": int(n * SPLIT["train"]),
        "val": int(n * SPLIT["val"]),
        "test": n - int(n * SPLIT["train"]) - int(n * SPLIT["val"])}

    # Rare-first ordering
    target_items = sorted(
        target_items,
        key=lambda x: sum(
            class_counts[c]
            for c in x["target_classes"]
        )
    )
    # Score function
    def compute_score(item, split) -> float:
        """
        Computes the score that is used to determine which image 
        goes to which split. The lowest score wins.
        """
        score = 0.0
        for c in item["target_classes"]:
            total = class_counts[c]
            actual_ratio = (split_dist[split][c] + 1) / total
            target_ratio = SPLIT[split]
            weight = 1 / total
            score += (weight * abs(actual_ratio - target_ratio) ** 5)
        # split size penalty
        size_ratio = (split_sizes[split] + 1) / target_size[split]
        score += 0.4 * (size_ratio ** 2)
        return score

    # Main assignment
    for item in target_items:
        best_split = None
        best_score = float("inf")
        for split in splits:
            # hard size constraint
            if split_sizes[split] >= target_size[split]:
                continue
            score = compute_score(item, split)
            if score < best_score:
                best_score = score
                best_split = split

        # fallback
        if best_split is None:
            best_split = min(
                splits,
                key=lambda s: split_sizes[s]
            )
        splits[best_split].append(item)
        split_sizes[best_split] += 1
        for c in item["target_classes"]:
            split_dist[best_split][c] += 1

    # Non-target images -> TRAIN
    splits["train"].extend(non_target_items)

    # REPORT
    print("\n==============================")
    print("SPLIT REPORT")
    print("==============================")
    total_all = len(dataset)
    for split, _ in splits.items():
        print(
            f"\n{split}: "
            f"{len(splits[split])} images "
            f"({len(splits[split]) / total_all:.2%})"
        )
    print("\n==============================")
    print("CLASS DISTRIBUTION")
    print("==============================")
    for c in class_counts:
        total = class_counts[c]
        print(f"\n{c}")
        for split in splits:
            count = split_dist[split][c]
            ratio = count / total
            print(
                f"  {split}: "
                f"{count} ({ratio:.3f})"
            )
    return splits

def export_splits(splits: dict[str, list]):
    """
    Export the splits to the defined folder.
    """
    for split, items in splits.items():
        img_out_dir = OUT_DIR / split
        ann_out_dir = OUT_DIR / split / "annotations"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        ann_out_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            shutil.copy(item["img"], img_out_dir / item["img"].name)
            shutil.copy(item["json"], ann_out_dir / item["json"].name)


def main():
    """
    Main entry point for dataset split.
    """
    random.seed(RANDOM_SEED)
    dataset = load_dataset()
    print(
        f"\nLoaded images: "
        f"{len(dataset)}"
    )
    splits = stratified_split(dataset)
    export_splits(splits)
    print("\nDONE")


if __name__ == "__main__":
    main()
