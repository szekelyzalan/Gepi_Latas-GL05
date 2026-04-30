"""
Example code for training a YOLO network.
"""

from ultralytics import YOLO

def train_yolo():
    """
    Model settings and training.
    """
    model = YOLO("yolov8s.pt")
    results = model.train(
        data="yolo_dataset/data.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        patience=20,

        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=5,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,

        fliplr=0.5,
        flipud=0.0,

        mosaic=1.0,
        mixup=0.1,

        device="cpu", # If there is GPU -> 0
        workers=4,
        project="runs/detect",
        name="traffic_signs",
        exist_ok=True
    )

    print("Training finished.")
    print(results)


if __name__ == "__main__":
    train_yolo()
