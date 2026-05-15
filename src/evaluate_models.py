from ultralytics import YOLO
import os
import glob
import json

# =========================
# PATHS
# =========================

MODEL_PATH = "best.pt"

# képek mappája
IMAGE_DIR = "train"

# json annotációk mappája
LABEL_DIR = "train/annotations"

# confidence threshold
CONF_THRESHOLD = 0.25


# =========================
# LOAD MODEL
# =========================

print("Modell betöltése...")

model = YOLO(MODEL_PATH)

print("Modell betöltve!\n")


# =========================
# MAP YOLO CLASS -> MAIN CLASS
# =========================

def map_prediction_to_main_class(pred_name):

    pred_name = pred_name.lower()

    if "warning" in pred_name:
        return "Warning"

    elif "regulatory" in pred_name:

        if "blue" in pred_name:
            return "regulatory-blue"

        else:
            return "regulatory-red"

    elif "complementary" in pred_name:
        return "complementary"

    elif "priority" in pred_name:
        return "priority road"

    else:
        return "other-sign"


# =========================
# STATISTICS
# =========================

total_images = 0

correct = 0
wrong = 0
not_detected = 0
missing_annotation = 0


# =========================
# IMAGE LIST
# =========================

image_paths = []

image_paths += glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
image_paths += glob.glob(os.path.join(IMAGE_DIR, "*.png"))
image_paths += glob.glob(os.path.join(IMAGE_DIR, "*.jpeg"))

print(f"Talált képek száma: {len(image_paths)}\n")


# =========================
# PROCESS IMAGES
# =========================

for image_path in image_paths:

    total_images += 1

    image_name = os.path.basename(image_path)
    image_stem = os.path.splitext(image_name)[0]

    # JSON annotáció path
    label_path = os.path.join(
        LABEL_DIR,
        image_stem + ".json"
    )

    # =========================
    # CHECK ANNOTATION
    # =========================

    if not os.path.exists(label_path):

        missing_annotation += 1

        print(f"[NINCS ANNOTÁCIÓ] {image_name}")

        continue

    # =========================
    # LOAD GROUND TRUTH
    # =========================

    gt_classes = []

    try:

        with open(label_path, "r") as f:

            data = json.load(f)

            for obj in data["objects"]:

                class_name = obj["label"]

                gt_classes.append(class_name)

    except Exception as e:

        print(f"[JSON HIBA] {image_name}: {e}")

        continue

    # duplikációk törlése
    gt_classes = list(set(gt_classes))

    # =========================
    # MODEL PREDICTION
    # =========================

    try:

        results = model.predict(
            image_path,
            conf=CONF_THRESHOLD,
            verbose=False
        )

    except Exception as e:

        print(f"[PREDIKCIÓS HIBA] {image_name}: {e}")

        continue

    predicted_classes = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls_id = int(box.cls[0])

            # YOLO class name
            class_name = model.names[cls_id]

            # convert detailed class -> main category
            mapped_class = map_prediction_to_main_class(class_name)

            predicted_classes.append(mapped_class)

    # duplikációk törlése
    predicted_classes = list(set(predicted_classes))

    # =========================
    # EVALUATION
    # =========================

    if len(predicted_classes) == 0:

        not_detected += 1

        print(f"[NEM FELISMERTE] {image_name}")

        continue

    # intersection
    matches = set(predicted_classes) & set(gt_classes)

    if len(matches) > 0:

        correct += 1

        print(f"[HELYES] {image_name}")

    else:

        wrong += 1

        print(f"[HIBÁS] {image_name}")

        print(f"  Ground truth: {gt_classes}")
        print(f"  Prediction:   {predicted_classes}")

    # =========================
    # DEBUG OUTPUT
    # =========================

    print(f"  GT:   {gt_classes}")
    print(f"  Pred: {predicted_classes}")
    print()


# =========================
# FINAL RESULTS
# =========================

print("\n=========================")
print("KIÉRTÉKELÉS")
print("=========================\n")

print(f"Összes kép:           {total_images}")
print(f"Helyes:               {correct}")
print(f"Hibás:                {wrong}")
print(f"Nem felismerte:       {not_detected}")
print(f"Hiányzó annotáció:    {missing_annotation}")

# accuracy
valid_images = total_images - missing_annotation

if valid_images > 0:

    accuracy = (correct / valid_images) * 100

    print(f"\nAccuracy: {accuracy:.2f}%")

else:

    print("\nNincs kiértékelhető kép.")