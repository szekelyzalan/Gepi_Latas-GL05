from ultralytics import YOLO

# pretrained modell (nagyon fontos!)
model = YOLO("yolov8n.pt")  # kezdésnek nano

model.train(
    data="yolo_dataset/data.yaml",
    epochs=30,
    imgsz=832,              # kisebb objektumok miatt nagyobb kép
    batch=8,                # ha kifutsz VRAM-ból
    name="traffic_signs",

    # 🔥 fontosak:
    patience=20,            # early stopping
    mosaic=1.0,             # augmentation
    mixup=0.1,
    degrees=10.0,           # forgatás (tábláknál fontos)
    scale=0.5,
    fliplr=0.5,

    # class imbalance ellen:
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
)
