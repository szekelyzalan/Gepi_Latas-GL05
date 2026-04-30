from ultralytics import YOLO

model = YOLO("runs/detect/traffic_signs/weights/best.pt")

results = model.predict(
    source="path/to/image.jpg",
    conf=0.25,
    save=True
)