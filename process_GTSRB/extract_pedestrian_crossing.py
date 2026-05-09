"""
The blue pedestrian crossing traffic signs are not included 
in the GTSRB dataset.

The GTSRB dataset can be downloaded from here:
    https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

This script extracts the pedestrian crossing signs from our own dataset, 
and saves it into the '20' folder.

After the folder is created, the user have to clear the GTSRB dataset, 
delete folder 20, and copy the created one.
"""
import json
import cv2
from pathlib import Path

# CONFIG
IMG_DIR = Path("train")
ANN_DIR = IMG_DIR / "annotations"
OUT_DIR = Path("20")
OUT_DIR.mkdir(exist_ok=True)
TARGET_LABELS = {
    "information--pedestrians-crossing--g1",
    "warning--pedestrians-crossing--g1"
}

def generate_name(counter: int) -> str:
    """
    Generates the names of the saved images.
    It follows the GTSRB nameing convention.
    """
    group1 = counter // 30
    group2 = counter % 30
    return f"00000_{group1:05d}_{group2:05d}.png"

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def crop_object(img, bbox):
    """
    Extracts the pedestrian crossing traffic sign from an image.
    """
    xmin = int(bbox["xmin"])
    ymin = int(bbox["ymin"])
    xmax = int(bbox["xmax"])
    ymax = int(bbox["ymax"])
    h, w = img.shape[:2]
    # safety clamp
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)
    return img[ymin:ymax, xmin:xmax]


def process_dataset():
    """
    Main entry point for extracting pedestrian crossing signs.
    """
    counter = 0
    for json_file in ANN_DIR.glob("*.json"):
        data: dict = load_json(json_file)
        img_file = IMG_DIR / (json_file.stem + ".jpg")
        if not img_file.exists():
            img_file = IMG_DIR / (json_file.stem + ".png")
        if not img_file.exists():
            continue
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        for obj in data.get("objects", []):
            label = obj.get("label")
            if label not in TARGET_LABELS:
                continue
            bbox = obj.get("bbox")
            if bbox is None:
                continue
            crop = crop_object(img, bbox)
            if crop.size == 0:
                continue
            filename = generate_name(counter)
            out_path = OUT_DIR / filename
            cv2.imwrite(str(out_path), crop)
            counter += 1
    print(f"Done. Extracted {counter} objects.")


if __name__ == "__main__":
    process_dataset()
