"""
The GTSRB dataset contains cropped images of traffic signs grouped by category.

This script further processes these images, 
and saves them into the clean_signs folder. 

The processed signs are further cropped, and 
saved with a black canvas outside the traffic sign.

A cleaning is introduced, but it is not perfect. 
The badly extracted signs are saved in the failed_signs folder. 
Manual cleaning may be required in the clean_signs folder!

The resultant images can be used to enrich the data used for YOLO training.
"""
import cv2
import numpy as np
from pathlib import Path

CHANGE_BUNDLE: dict[str, Path] = {
    "regulatory--maximum-speed-limit-70--g1": Path("clean_signs/regulatory--maximum-speed-limit-70--g1"),
    "regulatory--maximum-speed-limit-50--g1": Path("clean_signs/regulatory--maximum-speed-limit-50--g1"),
    "regulatory--maximum-speed-limit-60--g1": Path("clean_signs/regulatory--maximum-speed-limit-60--g1"),
    "regulatory--maximum-speed-limit-30--g1": Path("clean_signs/regulatory--maximum-speed-limit-30--g1"),
    "regulatory--maximum-speed-limit-80--g1": Path("clean_signs/regulatory--maximum-speed-limit-80--g1"),
    "regulatory--maximum-speed-limit-100--g1": Path("clean_signs/regulatory--maximum-speed-limit-100--g1"),
    "regulatory--maximum-speed-limit-20--g1": Path("clean_signs/regulatory--maximum-speed-limit-20--g1"),
    "regulatory--maximum-speed-limit-120--g1": Path("clean_signs/regulatory--maximum-speed-limit-120--g1"),
    "regulatory--no-entry--g1": Path("clean_signs/regulatory--no-entry--g1"),
    "warning--pedestrians-crossing--g1": Path("clean_signs/warning--pedestrians-crossing--g1"),
    "information--pedestrians-crossing--g1": Path("clean_signs/information--pedestrians-crossing--g1"),
    "regulatory--stop--g1": Path("clean_signs/regulatory--stop--g1"),
    "regulatory--yield--g1": Path("clean_signs/regulatory--yield--g1"),
}

# Config
OUTPUT_DIR = Path("clean_signs")
FAILED_DIR = Path("failed_signs")
OUTPUT_DIR.mkdir(exist_ok=True)
FAILED_DIR.mkdir(exist_ok=True)

# SETTINGS
MIN_CONTOUR_AREA_RATIO = 0.08
BLUR_KERNEL = 5
BORDER_MARGIN = 3
MIN_FOREGROUND_RATIO = 0.15
MIN_ASPECT_RATIO = 0.6
MAX_ASPECT_RATIO = 1.4
MIN_SIZE = 20

def save_failed(img, label, reason, counter):
    """
    Saves the failed sample into label/reason folder.
    """
    folder = FAILED_DIR / label / reason
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{counter:06d}.png"
    cv2.imwrite(str(path), img)

def extract_sign(img):
    """
    Extracts a patch of a sign from an image.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red mask
    lower_red1 = np.array([0, 60, 40])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 60, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)

    # Blue mask
    lower_blue = np.array([90, 50, 40])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combined mask
    mask = cv2.bitwise_or(red_mask, blue_mask)

    # Cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, "no_contour"
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # Foreground ratio filter
    foreground_ratio = area / (h * w)
    if foreground_ratio < MIN_FOREGROUND_RATIO:
        return None, "small_foreground"

    # Min contour area
    if area < (h * w * MIN_CONTOUR_AREA_RATIO):
        return None, "small_contour"

    # Bounding rectangle
    x, y, bw, bh = cv2.boundingRect(largest)

    # Border touch filter
    touches_border = (
        x <= BORDER_MARGIN or
        y <= BORDER_MARGIN or
        x + bw >= w - BORDER_MARGIN or
        y + bh >= h - BORDER_MARGIN
    )
    if touches_border:
        return None, "touches_border"

    # Aspect ratio filter
    aspect_ratio = bw / bh
    if (
        aspect_ratio < MIN_ASPECT_RATIO or
        aspect_ratio > MAX_ASPECT_RATIO
    ):
        return None, "bad_aspect_ratio"

    # Min size filter
    if bw < MIN_SIZE or bh < MIN_SIZE:
        return None, "too_small"

    # Create sign mask
    sign_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(
        sign_mask,
        [largest],
        -1,
        255,
        -1
    )

    # Smooth edge
    sign_mask = cv2.GaussianBlur(
        sign_mask,
        (BLUR_KERNEL, BLUR_KERNEL),
        0
    )

    # Apply mask
    result = cv2.bitwise_and(img, img, mask=sign_mask)

    # Tight crop
    cropped = result[y:y + bh, x:x + bw]
    return cropped, None


def process():
    """
    Main entry point for cleaning GTSRB images.
    """
    for label, folder in CHANGE_BUNDLE.items():
        print(f"\nProcessing: {label}")
        out_folder = OUTPUT_DIR / label
        out_folder.mkdir(parents=True, exist_ok=True)
        valid_ext = [".png", ".jpg", ".jpeg", ".ppm"]
        images = [
            p for p in folder.iterdir()
            if p.suffix.lower() in valid_ext
        ]
        success_counter = 0
        failed_counter = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            sign, fail_reason = extract_sign(img)

            # Failed sample
            if sign is None:
                save_failed(
                    img,
                    label,
                    fail_reason,
                    failed_counter
                )
                failed_counter += 1
                continue

            # SAVE CLEAN SIGN
            out_path = out_folder / f"{success_counter:06d}.png"
            cv2.imwrite(str(out_path), sign)
            success_counter += 1

        print(f"Saved clean signs : {success_counter}")
        print(f"Saved failed signs: {failed_counter}")


if __name__ == "__main__":
    process()
