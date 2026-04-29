"""
Examines split results by percentage
Expected outcome:
{
    category number: 
        all: all in that category from data.yaml
        train: all found in train TXTs
        val: trin: all found in val TXTs
        test: trin: all found in test TXTs
}
"""
from pathlib import Path
from collections import Counter, defaultdict

def examine_txt(path: Path) -> Counter:
    """
    Processes a .txt that contains the labels for YOLO training.
    """
    counts = []
    for txt in path.iterdir():
        with open(txt, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            if not line or line == '\n':
                continue
            cnt = int(line.split(' ')[0])
            counts .append(cnt)
    return Counter(counts)

def get_expected_cnt(cnt_path: Path) -> dict[int, dict[str, int]]:
    """
    Processes categories_to_keep.txt.
    """
    exp_cnt = defaultdict(dict)
    with open(cnt_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        cnt = int(line.split(':')[1].strip())
        exp_cnt[i] = {
            "all": cnt,
            "train": 0,
            "val": 0,
            "test": 0
        }
    return exp_cnt

def main():
    """
    Analyzes train, val, test ratios with respect ot each 
    label.
    """
    train_path = Path("yolo_dataset/labels/train")
    val_path = Path("yolo_dataset/labels/val")
    test_path = Path("yolo_dataset/labels/test")

    train_cnt = examine_txt(train_path)
    val_cnt = examine_txt(val_path)
    test_cnt = examine_txt(test_path)
    exp_cnt = get_expected_cnt(Path("categories_to_keep.txt"))

    for category, cnt in train_cnt.items():
        exp_cnt[category]["train"] = cnt
    for category, cnt in val_cnt.items():
        exp_cnt[category]["val"] = cnt
    for category, cnt in test_cnt.items():
        exp_cnt[category]["test"] = cnt

    for category_number, types in exp_cnt.items():
        print(f"{category_number}: ")
        percentage_train = types["train"] / types["all"]
        percentage_val = types["val"] / types["all"]
        percentage_test = types["test"] / types["all"]
        print(f"\ttrain: {percentage_train}")
        print(f"\tval: {percentage_val}")
        print(f"\ttest: {percentage_test}")


if __name__ == "__main__":
    main()
