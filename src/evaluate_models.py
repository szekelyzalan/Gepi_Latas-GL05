from ultralytics import YOLO
import os
import glob
import json
from collections import defaultdict

# =========================================================
# PATHS
# =========================================================

MODEL_PATH = os.path.join("models", "best.pt")

IMAGE_DIR = os.path.join("test_2")

LABEL_DIR = os.path.join(IMAGE_DIR, "annotations")

# =========================================================
# SETTINGS
# =========================================================

CONF_THRESHOLD = 0.35

IOU_THRESHOLD = 0.5

VALID_CLASSES = {
    "stop",
    "no_entry",
    "pedestrian_crossing",
    "yield",
    "speed_limit"
}

# =========================================================
# LOAD MODEL
# =========================================================

print("===================================")
print("YOLO KIÉRTÉKELÉS")
print("===================================\n")

print("Modell betöltése...")

model = YOLO(MODEL_PATH)

print("Modell betöltve!\n")

# =========================================================
# IOU CALCULATION
# =========================================================

def calculate_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])

    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)

    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    boxA_area = (
        (boxA[2] - boxA[0]) *
        (boxA[3] - boxA[1])
    )

    boxB_area = (
        (boxB[2] - boxB[0]) *
        (boxB[3] - boxB[1])
    )

    union_area = (
        boxA_area +
        boxB_area -
        inter_area
    )

    return inter_area / union_area

# =========================================================
# STATISTICS
# =========================================================

total_images = 0

total_gt_objects = 0

TP = 0
FP = 0
FN = 0

perfect_images = 0
partial_images = 0
failed_images = 0

missing_annotations = 0

empty_images = 0
empty_images_with_fp = 0

class_stats = defaultdict(lambda: {
    "TP": 0,
    "FP": 0,
    "FN": 0
})

# =========================================================
# IMAGE LIST
# =========================================================

image_paths = []

image_paths += glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
image_paths += glob.glob(os.path.join(IMAGE_DIR, "*.png"))
image_paths += glob.glob(os.path.join(IMAGE_DIR, "*.jpeg"))

print(f"Talált képek száma: {len(image_paths)}\n")

# =========================================================
# PROCESS IMAGES
# =========================================================

for idx, image_path in enumerate(image_paths):

    total_images += 1

    # progress indicator
    if (idx + 1) % 100 == 0:

        print(
            f"Feldolgozva: "
            f"{idx + 1}/{len(image_paths)}"
        )

    image_name = os.path.basename(image_path)

    image_stem = os.path.splitext(image_name)[0]

    # =====================================================
    # ANNOTATION FILE
    # =====================================================

    label_path = os.path.join(
        LABEL_DIR,
        image_stem + ".json"
    )

    if not os.path.exists(label_path):

        missing_annotations += 1

        continue

    # =====================================================
    # LOAD GROUND TRUTH
    # =====================================================

    try:

        with open(label_path, "r") as f:

            data = json.load(f)

    except Exception:

        continue

    gt_objects = []

    for obj in data["objects"]:

        label = obj["label"]

        if label not in VALID_CLASSES:
            continue

        bbox = obj["bbox"]

        gt_box = [
            float(bbox["xmin"]),
            float(bbox["ymin"]),
            float(bbox["xmax"]),
            float(bbox["ymax"])
        ]

        gt_objects.append({
            "label": label,
            "bbox": gt_box,
            "matched": False
        })

    total_gt_objects += len(gt_objects)

    if len(gt_objects) == 0:

        empty_images += 1

    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    try:

        results = model.predict(
            image_path,
            conf=CONF_THRESHOLD,
            verbose=False
        )

    except Exception:

        continue

    pred_objects = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls_id = int(box.cls[0])

            class_name = model.names[cls_id]

            if class_name not in VALID_CLASSES:
                continue

            xyxy = box.xyxy[0].cpu().numpy()

            pred_box = [
                float(xyxy[0]),
                float(xyxy[1]),
                float(xyxy[2]),
                float(xyxy[3])
            ]

            pred_objects.append({
                "label": class_name,
                "bbox": pred_box,
                "matched": False
            })

    # =====================================================
    # MATCHING
    # =====================================================

    image_tp = 0
    image_fp = 0
    image_fn = 0

    # GT -> prediction matching
    for gt in gt_objects:

        best_iou = 0.0
        best_pred = None

        for pred in pred_objects:

            if pred["matched"]:
                continue

            if pred["label"] != gt["label"]:
                continue

            iou = calculate_iou(
                gt["bbox"],
                pred["bbox"]
            )

            if iou > best_iou:

                best_iou = iou
                best_pred = pred

        if best_iou >= IOU_THRESHOLD:

            TP += 1
            image_tp += 1

            gt["matched"] = True
            best_pred["matched"] = True

            class_stats[gt["label"]]["TP"] += 1

        else:

            FN += 1
            image_fn += 1

            class_stats[gt["label"]]["FN"] += 1

    # unmatched predictions = FP
    for pred in pred_objects:

        if not pred["matched"]:

            FP += 1
            image_fp += 1

            class_stats[pred["label"]]["FP"] += 1

    # =====================================================
    # IMAGE LEVEL RESULT
    # =====================================================

    if len(gt_objects) == 0 and image_fp > 0:

        empty_images_with_fp += 1

    if image_fn == 0 and image_fp == 0:

        perfect_images += 1

    elif image_tp > 0:

        partial_images += 1

    else:

        failed_images += 1

# =========================================================
# FINAL METRICS
# =========================================================

print("\n===================================")
print("VÉGSŐ KIÉRTÉKELÉS")
print("===================================\n")

precision = (
    TP / (TP + FP)
    if (TP + FP) > 0
    else 0
)

recall = (
    TP / (TP + FN)
    if (TP + FN) > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)

fp_per_image = FP / total_images

empty_fp_rate = (
    empty_images_with_fp / empty_images
    if empty_images > 0
    else 0
)

print(f"Összes kép:                  {total_images}")

print(f"Hiányzó annotáció:           {missing_annotations}")

print()

print(f"Üres képek (nincs GT):       {empty_images}")

print(
    f"Üres képek FP-vel:           "
    f"{empty_images_with_fp}"
)

print(
    f"Empty-image FP rate:         "
    f"{empty_fp_rate:.4f}"
)

print()

print(f"Összes GT tábla:             {total_gt_objects}")

print()

print(f"TP (jó detektálás):          {TP}")

print(f"FP (fals pozitív):           {FP}")

print(f"FN (kihagyott tábla):        {FN}")

print()

print(f"Precision:                   {precision:.4f}")

print(f"Recall:                      {recall:.4f}")

print(f"F1-score:                    {f1:.4f}")

print()

print(f"FP / image:                  {fp_per_image:.4f}")

print()

print(f"Tökéletes képek:             {perfect_images}")

print(f"Részben jó képek:            {partial_images}")

print(f"Hibás képek:                 {failed_images}")

# =========================================================
# PER-CLASS STATS
# =========================================================

print("\n===================================")
print("OSZTÁLYONKÉNTI STATISZTIKA")
print("===================================\n")

for cls_name, stats in class_stats.items():

    cls_tp = stats["TP"]
    cls_fp = stats["FP"]
    cls_fn = stats["FN"]

    cls_precision = (
        cls_tp / (cls_tp + cls_fp)
        if (cls_tp + cls_fp) > 0
        else 0
    )

    cls_recall = (
        cls_tp / (cls_tp + cls_fn)
        if (cls_tp + cls_fn) > 0
        else 0
    )

    cls_f1 = (
        2 * cls_precision * cls_recall /
        (cls_precision + cls_recall)
        if (cls_precision + cls_recall) > 0
        else 0
    )

    print(f"Class: {cls_name}")

    print(f"  TP: {cls_tp}")

    print(f"  FP: {cls_fp}")

    print(f"  FN: {cls_fn}")

    print(f"  Precision: {cls_precision:.4f}")

    print(f"  Recall:    {cls_recall:.4f}")

    print(f"  F1-score:  {cls_f1:.4f}")

    print()

print("===================================")
print("KIÉRTÉKELÉS KÉSZ")
print("===================================")