"""
Training image augmentation pipeline

TURN OFF OR MINIMIZE DURING YOLO TRAINING:
hsv_h
hsv_s
hsv_v
degrees
translate
scale
shear
perspective
mosaic
mixup
copy_paste
"""
import cv2
import json
import random
import shutil
import numpy as np
from pathlib import Path

# Config
INPUT_DIR = Path("dataset_modified/train")
INPUT_ANN_DIR = INPUT_DIR / "annotations"
OUTPUT_DIR = Path("dataset_augmented/train")
OUTPUT_ANN_DIR = OUTPUT_DIR / "annotations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ANN_DIR.mkdir(parents=True, exist_ok=True)

# Settings
# extra augmented copies per image
BASE_AUG_PER_IMAGE = 2
# stop oversampling
STOP_EXTRA_COPIES = 4
# probabilities
P_MOTION_BLUR = 0.25
P_JPEG = 0.30
P_GAMMA = 0.35
P_NOISE = 0.25
P_PERSPECTIVE = 0.0 # Needs bounding box transform to work.
P_DEFOCUS = 0.20
P_COLOR_SHIFT = 0.20
P_SHADOW = 0.20
P_CONTRAST = 0.25
P_DOWNSCALE = 0.25

# Stop signs are rare -> make more copies.
STOP_LABEL = "regulatory--stop--g1"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def motion_blur(img):
    """
    Creates motion blur.
    """
    k = random.choice([3, 5, 7, 9])
    kernel = np.zeros((k, k))
    if random.random() < 0.5:
        kernel[k // 2, :] = 1
    else:
        kernel[:, k // 2] = 1
    kernel /= k
    return cv2.filter2D(img, -1, kernel)

def defocus_blur(img):
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)

def jpeg_compression(img):
    quality = random.randint(35, 95)
    encode_param = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        quality
    ]
    _, encimg = cv2.imencode(
        ".jpg",
        img,
        encode_param
    )
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def gamma_transform(img):
    gamma = random.uniform(0.6, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")
    return cv2.LUT(img, table)

def add_noise(img):
    """
    Adds Gaussian-noise to the image.
    """
    sigma = random.uniform(2, 6)
    noise = np.random.normal(
        0,
        sigma,
        img.shape
    ).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def random_contrast(img):
    alpha = random.uniform(0.75, 1.35)
    img = img.astype(np.float32)
    img = img * alpha
    return np.clip(img, 0, 255).astype(np.uint8)

def random_shadow(img):
    h, w = img.shape[:2]
    shadow = np.zeros((h, w), dtype=np.float32)
    x1 = random.randint(0, w)
    x2 = random.randint(0, w)
    cv2.line(
        shadow,
        (x1, 0),
        (x2, h),
        1,
        thickness=random.randint(
            int(w * 0.2),
            int(w * 0.5)
        )
    )
    shadow = cv2.GaussianBlur(
        shadow,
        (101, 101),
        0
    )
    strength = random.uniform(0.4, 0.75)
    img = img.astype(np.float32)
    for c in range(3):
        img[:, :, c] *= (
            1 - shadow * strength
        )
    return np.clip(img, 0, 255).astype(np.uint8)

def perspective_warp(img):
    h, w = img.shape[:2]
    margin = int(min(h, w) * 0.05)
    src = np.float32([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h]
    ])
    dst = np.float32([
        [random.randint(0, margin), random.randint(0, margin)],
        [w - random.randint(0, margin), random.randint(0, margin)],
        [random.randint(0, margin), h - random.randint(0, margin)],
        [w - random.randint(0, margin), h - random.randint(0, margin)]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        img,
        M,
        (w, h),
        borderMode=cv2.BORDER_REFLECT101
    )
    return warped

def color_shift(img):
    """
    Shifts the color temperature to simulate different lightings.
    """
    img = img.astype(np.float32)
    r_shift = random.uniform(0.9, 1.1)
    g_shift = random.uniform(0.9, 1.1)
    b_shift = random.uniform(0.9, 1.1)
    img[:, :, 2] *= r_shift
    img[:, :, 1] *= g_shift
    img[:, :, 0] *= b_shift
    return np.clip(img, 0, 255).astype(np.uint8)

def random_downscale(img):
    h, w = img.shape[:2]
    scale = random.uniform(0.4, 0.8)
    nw = int(w * scale)
    nh = int(h * scale)
    small = cv2.resize(
        img,
        (nw, nh),
        interpolation=cv2.INTER_LINEAR
    )
    restored = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )
    return restored

def augment_image(img):
    """
    Implements the full augmentation pipeline.
    """
    aug = img.copy()

    # Photometric
    if random.random() < P_GAMMA:
        aug = gamma_transform(aug)
    if random.random() < P_CONTRAST:
        aug = random_contrast(aug)
    if random.random() < P_COLOR_SHIFT:
        aug = color_shift(aug)
    if random.random() < P_SHADOW:
        aug = random_shadow(aug)

    # Blur
    if random.random() < P_MOTION_BLUR:
        aug = motion_blur(aug)
    if random.random() < P_DEFOCUS:
        aug = defocus_blur(aug)

    # Sensor
    if random.random() < P_NOISE:
        aug = add_noise(aug)
    if random.random() < P_DOWNSCALE:
        aug = random_downscale(aug)

    # Lastly: compression
    if random.random() < P_JPEG:
        aug = jpeg_compression(aug)
    return aug

def contains_stop(annotation: dict) -> bool:
    """
    Checks if an image contains a stop sign.
    """
    for obj in annotation.get("objects", []):
        if obj["label"] == STOP_LABEL:
            return True
    return False


def process():
    """
    Main entry point of the augmentation pipeline.
    """
    image_paths = list(INPUT_DIR.glob("*.jpg"))
    total_saved = 0
    for img_path in image_paths:
        json_path = INPUT_ANN_DIR / (
            img_path.stem + ".json"
        )
        if not json_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ann = load_json(json_path)

        # Save original
        shutil.copy(
            img_path,
            OUTPUT_DIR / img_path.name
        )
        shutil.copy(
            json_path,
            OUTPUT_ANN_DIR / json_path.name
        )
        total_saved += 1

        # How many augmentations?
        num_augs = BASE_AUG_PER_IMAGE
        if contains_stop(ann):
            num_augs += STOP_EXTRA_COPIES

        # Generate augmentations
        for i in range(num_augs):
            aug = augment_image(img)
            out_name = f"{img_path.stem}_aug_{i}.jpg"
            out_img_path = OUTPUT_DIR / out_name
            out_json_path = OUTPUT_ANN_DIR / (
                f"{img_path.stem}_aug_{i}.json"
            )
            cv2.imwrite(
                str(out_img_path),
                aug
            )
            save_json(
                out_json_path,
                ann
            )
            total_saved += 1
    print(f"\nSaved images: {total_saved}")


if __name__ == "__main__":
    process()
