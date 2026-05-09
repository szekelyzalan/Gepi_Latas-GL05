"""
This script changes some traffic signs on an image to a sign 
that needs to be recognized in this project.

This way the training dataset is enriched.

TO_CHANGE caontains the sign labels that will be replaced 70% of the time.
other-signs are replaced 10% of the time, if the sign is large enough, 
and has a clean circle, or triangle shape.
"""
import cv2
import json
import random
import numpy as np
from pathlib import Path
from utils.change_traffic_signs import (
    CHANGE_BUNDLE,
    TO_CHANGE,
    SEMANTIC_CLASS_MAP,
    WEIGHTS,
    OLD_SIGN_SHAPES
)

# Config
TRAIN_DIR = Path("dataset_raw/train")
IMG_DIR = TRAIN_DIR
ANN_DIR = TRAIN_DIR / "annotations"
CLEAN_SIGNS_DIR = Path("clean_signs")
OUT_DIR = Path("dataset_modified/train")
OUT_IMG_DIR = OUT_DIR
OUT_ANN_DIR = OUT_DIR / "annotations"
OUT_IMG_DIR.mkdir(
    parents=True,
    exist_ok=True
)
OUT_ANN_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Settings
REPLACEMENT_PROBABILITY = 0.7
INPAINT_RADIUS = 7
EDGE_BLUR = 5
TOP_K_MATCHES = 10
TARGET_SIGN_SHAPES = {
    "regulatory--maximum-speed-limit-70--g1": "circle",
    "regulatory--maximum-speed-limit-50--g1": "circle",
    "regulatory--maximum-speed-limit-60--g1": "circle",
    "regulatory--maximum-speed-limit-30--g1": "circle",
    "regulatory--maximum-speed-limit-80--g1": "circle",
    "regulatory--maximum-speed-limit-100--g1": "circle",
    "regulatory--maximum-speed-limit-20--g1": "circle",
    "regulatory--maximum-speed-limit-120--g1": "circle",
    "regulatory--no-entry--g1": "circle",
    "warning--pedestrians-crossing--g1": "triangle",
    "information--pedestrians-crossing--g1": "triangle",
    "regulatory--yield--g1": "triangle",
    "regulatory--stop--g1": "octagon",
}
OTHER_SIGN_LABEL = "other-sign"
OTHER_SIGN_REPLACEMENT_PROB = 0.10
MIN_OTHER_SIGN_SIZE = 32 * 32

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# Semantic groups
SEMANTIC_TO_LABELS = {}
for label, semantic in SEMANTIC_CLASS_MAP.items():
    if semantic not in SEMANTIC_TO_LABELS:
        SEMANTIC_TO_LABELS[semantic] = []
    SEMANTIC_TO_LABELS[semantic].append(label)

def get_random_clean_sign(label: str):
    """
    Selects a clean sign randomly from a randomly selected folder, 
    based on the expected shape of a traffic sign.
    """
    folder = CLEAN_SIGNS_DIR / label
    if not folder.exists():
        return None
    valid_ext = [".png", ".jpg", ".jpeg"]
    imgs = [
        p for p in folder.iterdir()
        if p.suffix.lower() in valid_ext
    ]
    if not imgs:
        return None
    img_path = random.choice(imgs)
    return cv2.imread(str(img_path))

def estimate_other_sign_shape(bbox):
    """
    Estimates the shape of an other-sign labeled traffic sign.
    """
    xmin = bbox["xmin"]
    xmax = bbox["xmax"]
    ymin = bbox["ymin"]
    ymax = bbox["ymax"]
    w = xmax - xmin
    h = ymax - ymin
    if w <= 0 or h <= 0:
        return None
    ratio = w / h
    area = w * h

    # too small
    if area < MIN_OTHER_SIGN_SIZE:
        return None

    # nearly square -> likely circle
    if 0.8 <= ratio <= 1.2:
        return "circle"

    # wide / tall -> rectangle
    if ratio > 1.35 or ratio < 0.75:
        return "rectangle"

    # maybe diamond
    return "diamond"

def choose_target_label(old_shape: str) -> str:
    """
    Implements a weighted target label selection algorithm.
    """
    semantic_classes = list(WEIGHTS.keys())
    semantic_weights = [WEIGHTS[s] for s in semantic_classes]
    for _ in range(50):
        chosen_semantic = random.choices(
            semantic_classes,
            weights=semantic_weights,
            k=1
        )[0]
        candidate_labels = SEMANTIC_TO_LABELS[chosen_semantic]

        # Weights to choose speed limit
        if chosen_semantic == "speed-limit":
            speed_weights = {
                "regulatory--maximum-speed-limit-20--g1": 0.6,
                "regulatory--maximum-speed-limit-30--g1": 1.4,
                "regulatory--maximum-speed-limit-50--g1": 2.0,
                "regulatory--maximum-speed-limit-60--g1": 1.5,
                "regulatory--maximum-speed-limit-70--g1": 1.5,
                "regulatory--maximum-speed-limit-80--g1": 1.7,
                "regulatory--maximum-speed-limit-100--g1": 1.0,
                "regulatory--maximum-speed-limit-120--g1": 0.5,
            }
            weights = [speed_weights[label] for label in candidate_labels]
            target_label = random.choices(
                candidate_labels,
                weights=weights,
                k=1
            )[0]
        else:
            target_label = random.choice(candidate_labels)

        # Shape match
        target_shape = TARGET_SIGN_SHAPES.get(
            target_label,
            "rectangle"
        )
        if target_shape == old_shape:
            return target_label
    return random.choice(
        list(CHANGE_BUNDLE.keys())
    )

def create_foreground_mask(patch):
    """
    The input image is patch with black background.
    This function creates a foreground mask for the patch.
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = (
        (gray > 15) &
        (v > 30) &
        (s > 20)
    ).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # keep only largest contour
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        clean = np.zeros_like(mask)
        cv2.drawContours(
            clean,
            [largest],
            -1,
            255,
            -1
        )
        mask = clean
    mask = cv2.GaussianBlur(
        mask,
        (EDGE_BLUR, EDGE_BLUR),
        0
    )
    return mask.astype(np.float32) / 255.0

def create_shape_mask(shape: str, w: float, h: float):
    """
    Creates a shape mask for inpaint.
    This is needed to ensure realistic replacement of the signs.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if shape == "circle":
        cv2.circle(
            mask,
            (w // 2, h // 2),
            int(min(w, h) * 0.42),
            255,
            -1
        )
    elif shape == "triangle":
        pts = np.array([
            [w // 2, int(h * 0.08)],
            [int(w * 0.08), int(h * 0.9)],
            [int(w * 0.92), int(h * 0.9)]
        ])
        cv2.fillPoly(mask, [pts], 255)
    elif shape == "octagon":
        r = min(w, h) * 0.38
        cx = w // 2
        cy = h // 2
        pts = []
        for i in range(8):
            theta = np.deg2rad(i * 45 + 22.5)
            x = int(cx + r * np.cos(theta))
            y = int(cy + r * np.sin(theta))
            pts.append([x, y])
        pts = np.array(pts)
        cv2.fillPoly(mask, [pts], 255)
    elif shape == "diamond":
        pts = np.array([
            [w // 2, int(h * 0.05)],
            [int(w * 0.95), h // 2],
            [w // 2, int(h * 0.95)],
            [int(w * 0.05), h // 2]
        ])
        cv2.fillPoly(mask, [pts], 255)
    else:
        cv2.rectangle(
            mask,
            (0, 0),
            (w, h),
            255,
            -1
        )
    return mask

def match_brightness_and_color(patch, target_roi):
    """
    Matches the brightness and color of a patch 
    with the original sign to ensure smooth replacement.
    """
    patch = patch.astype(np.float32)
    target = target_roi.astype(np.float32)
    for c in range(3):
        patch_mean = np.mean(patch[:, :, c])
        patch_std = np.std(patch[:, :, c])
        target_mean = np.mean(target[:, :, c])
        target_std = np.std(target[:, :, c])
        if patch_std < 1:
            patch_std = 1
        patch[:, :, c] = (
            (patch[:, :, c] - patch_mean)
            * (target_std / patch_std)
            + target_mean
        )
    return np.clip(patch, 0, 255).astype(np.uint8)

def remove_old_sign(img, bbox, shape):
    """
    Removes the old sign from an image and creates the inpaint.
    If the sign is too small, removal is skipped.
    """
    xmin = max(0, int(bbox["xmin"]))
    ymin = max(0, int(bbox["ymin"]))
    xmax = min(img.shape[1], int(bbox["xmax"]))
    ymax = min(img.shape[0], int(bbox["ymax"]))
    w = xmax - xmin
    h = ymax - ymin
    if w <= 1 or h <= 1:
        return img
    local_mask = create_shape_mask(shape, w, h)
    full_mask = np.zeros(
        img.shape[:2],
        dtype=np.uint8
    )
    full_mask[ymin:ymax, xmin:xmax] = local_mask
    cleaned = cv2.inpaint(
        img,
        full_mask,
        INPAINT_RADIUS,
        cv2.INPAINT_TELEA
    )
    return cleaned

def paste_new_sign(img, patch, bbox):
    """
    Pastes the new traffic sign into the image.
    """
    xmin = max(0, int(bbox["xmin"]))
    ymin = max(0, int(bbox["ymin"]))
    xmax = min(img.shape[1], int(bbox["xmax"]))
    ymax = min(img.shape[0], int(bbox["ymax"]))
    w = xmax - xmin
    h = ymax - ymin
    if w <= 1 or h <= 1:
        return img
    # resize patch
    patch = cv2.resize(
        patch,
        (w, h),
        interpolation=cv2.INTER_AREA
    )
    roi = img[ymin:ymax, xmin:xmax]
    # brightness adaptation
    patch = match_brightness_and_color(patch, roi)
    patch = cv2.GaussianBlur(
        patch,
        (3, 3),
        0
    )
    # foreground mask
    mask = create_foreground_mask(patch)
    mask = np.expand_dims(mask, axis=2)
    # blend
    blended = (
        patch.astype(np.float32) * mask
        + roi.astype(np.float32) * (1 - mask)
    )
    img[ymin:ymax, xmin:xmax] = blended.astype(np.uint8)
    return img


def process():
    """
    Main entry point for replacing traffic signs.
    """
    for img_path in IMG_DIR.glob("*.jpg"):
        json_path = ANN_DIR / (img_path.stem + ".json")
        if not json_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        data = load_json(json_path)
        modified = False
        for obj in data.get("objects", []):
            label = obj["label"]

            # Normal replacement
            if label in TO_CHANGE:
                if random.random() > REPLACEMENT_PROBABILITY:
                    continue
                shape = OLD_SIGN_SHAPES.get(label, "rectangle")

            # other-sign replacement
            elif label == OTHER_SIGN_LABEL:
                if random.random() > OTHER_SIGN_REPLACEMENT_PROB:
                    continue
                shape = estimate_other_sign_shape(obj["bbox"])

                # unknown shape
                if shape is None:
                    continue
            else:
                continue

            target_label = choose_target_label(shape)
            patch = get_random_clean_sign(target_label)
            if patch is None:
                continue

            img = remove_old_sign(
                img,
                obj["bbox"],
                shape
            )
            img = paste_new_sign(
                img,
                patch,
                obj["bbox"]
            )

            # Update label
            obj["label"] = target_label
            modified = True

        # If modification happend -> save
        if modified:
            cv2.imwrite(
                str(OUT_IMG_DIR / img_path.name),
                img
            )
            save_json(
                OUT_ANN_DIR / json_path.name,
                data
            )


if __name__ == "__main__":
    process()
