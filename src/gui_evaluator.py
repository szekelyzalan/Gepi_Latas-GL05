import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2


# =========================
# MODEL PATH
# =========================
MODEL_PATH = "best.pt"


# =========================
# LOAD MODEL
# =========================
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Model betöltési hiba: {e}")
    model = None


# =========================
# GUI
# =========================
root = tk.Tk()
root.title("Közlekedési tábla felismerő")
root.geometry("1000x700")


# =========================
# IMAGE LABEL
# =========================
image_label = tk.Label(root)
image_label.pack(pady=10)


# =========================
# RESULT TEXT
# =========================
result_text = tk.Text(root, height=10, width=80)
result_text.pack(pady=10)


# =========================
# DETECTION FUNCTION
# =========================
def detect_image():

    if model is None:
        messagebox.showerror("Hiba", "A modell nem tölthető be!")
        return

    file_path = filedialog.askopenfilename(
        filetypes=[("Images", "*.png *.jpg *.jpeg")]
    )

    if not file_path:
        return

    # Kép betöltése
    image = cv2.imread(file_path)

    if image is None:
        messagebox.showerror("Hiba", "Nem sikerült megnyitni a képet!")
        return

    # YOLO inference
    results = model.predict(image, conf=0.25)

    detected_labels = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Saját class nevek helyett:
            class_name = model.names[cls_id]

            detected_labels.append(
                f"{class_name} ({conf:.2f})"
            )

            # Bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            cv2.putText(
                image,
                f"{class_name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # BGR -> RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # PIL image
    pil_image = Image.fromarray(image_rgb)

    # GUI méretezés
    pil_image.thumbnail((900, 500))

    # Tk image
    tk_image = ImageTk.PhotoImage(pil_image)

    image_label.config(image=tk_image)
    image_label.image = tk_image

    # Szövegmező törlése
    result_text.delete(1.0, tk.END)

    if detected_labels:

        result_text.insert(
            tk.END,
            "Felismert közlekedési táblák:\n\n"
        )

        for label in detected_labels:
            result_text.insert(
                tk.END,
                f"- {label}\n"
            )

    else:
        result_text.insert(
            tk.END,
            "Nem talált közlekedési táblát."
        )


# =========================
# BUTTON
# =========================
detect_button = tk.Button(
    root,
    text="Kép megnyitása és felismerés",
    command=detect_image,
    font=("Arial", 14),
    padx=10,
    pady=5
)

detect_button.pack(pady=10)


# =========================
# START GUI
# =========================
root.mainloop()