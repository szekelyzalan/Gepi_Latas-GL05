from ultralytics import YOLO
import os

def main():
    # -----------------------------
    # CONFIGURATION
    # -----------------------------
    DATA_YAML = "yolo_dataset\data.yaml"   # path to your yaml
    MODEL_NAME = "yolov8n.pt"              # nano / small / medium etc.
    EPOCHS = 30
    IMG_SIZE = 640
    BATCH_SIZE = 16

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    model = YOLO(MODEL_NAME)

    # -----------------------------
    # TRAIN
    # -----------------------------
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name="yolo_training",
        workers=4,
        device=0  # use "cpu" if no GPU
    )

    # -----------------------------
    # VALIDATE
    # -----------------------------
    metrics = model.val()
    print("Validation metrics:", metrics)

    # -----------------------------
    # EXPORT MODEL (optional)
    # -----------------------------
    model.export(format="onnx")  # or "torchscript", "engine"

    print("Training complete!")

if __name__ == "__main__":
    main()