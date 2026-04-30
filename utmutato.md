# 🚦 Traffic Sign Recognition Project – Használati útmutató

---

## 🔧 GitHub használat

Lehet manuálisan is feltöltögetni fájlokat, de az hamar macerássá válik.

### ✅ Ajánlott setup
- Visual Studio Code  
- Git  
- VS Code Git extension (általában alapból van)

### 📥 Repo letöltése

    git clone <repo_url>

A repo URL a GitHub-on a zöld `<Code>` gomb alatt található.

### 🔄 Alap Git workflow

VS Code → Source Control panel

vagy terminálból:

    git pull
    git add .
    git commit -m "valami értelmes üzenet"
    git push

---

## 🐍 Virtuális környezet (venv)

Nagyon ajánlott, hogy mind ugyanazt a környezetet használjuk.

### 📦 Létrehozás

    python -m venv .venv

### ▶️ Aktiválás (Windows)

    .\.venv\Scripts\activate.bat

Ha aktív, akkor látni fogod a `(.venv)` prefixet.

### 🐍 Python verzió

A projekt Python **3.12.9**-re lett írva.

Ha több Python verziód van:

    py -3.12 -m venv .venv

---

## 📚 Package-ek kezelése

### Telepítés:

    pip install -r requirements.txt

### Új package hozzáadása után:

    pip freeze > requirements.txt

### ⚠️ PyTorch GPU

Ha van NVIDIA GPU-d:
1. Töröld a PyTorch sort a requirements.txt-ből  
2. Menj ide: https://pytorch.org/  
3. Válaszd ki a megfelelő CUDA verziót  
4. Onnan telepítsd  

---

## ☁️ Adatok

Az adatok Google Drive-on vannak tárolva.  
👉 https://drive.google.com/drive/folders/1TdVCIIskzAEpzrn1uQlRjEkD1SUI1anz?usp=sharing

---

## 🧠 Tanítás (Google Colab)

Ajánlott Colab használata GPU miatt.

### 🔑 GitHub access token

Mivel private repo:

1. GitHub → Profile → Settings  
2. Developer settings  
3. Personal access tokens → Fine-grained tokens  
4. Generate new token  

Beállítás:
- Repository access: *only selected repo*  
- Permissions:
  - ✅ Contents: Read  

⚠️ A tokent csak egyszer látod → mentsd el!

---

### 🔗 Colab + GitHub összekötés

    from google.colab import drive
    drive.mount('/content/drive')

    import os
    token = "IDE_ÍRD_A_TOKENED"
    repo_url = f"https://{token}@github.com/<username>/<repo>.git"

    !git clone $repo_url
    %cd <repo>

Alternatíva:

    import os
    os.environ['GITHUB_TOKEN'] = "token"

    !git clone https://$GITHUB_TOKEN@github.com/<username>/<repo>.git

---

## 📂 Elkészült kódok

### 🔍 data_inspector.py

Elvárt struktúra:

    root/
     └── train/

Mit csinál:
- `visualized/` mappa létrehozása (bounding boxos képek)
- `unimportant_logs/report.txt` generálása

👉 VS Code keresővel könnyen szűrhető

**Konklúzió:**  
Metaadatok nem szükségesek a tanításhoz.

---

### ✂️ split_data.py

Input:

    train/

Output:

    yolo_dataset/
     ├── train/
     ├── val/
     └── test/

Arány:
- 70% train  
- 20% val  
- 10% test  

👉 YOLO kompatibilis formátum

Extra:
- `utils/categories_to_keep.py`
  - kategória összevonás  
  - kevés adat → össze kell húzni  

---

### 📊 split_inspector.py

Nem kötelező, de hasznos.

Mit csinál:
- kategóriák eloszlását ellenőrzi  
- train/val/test összehasonlítás  

---

### 🧠 train.py

- Alap tanítás CPU-n  
- 30 epoch  

👉 Jó baseline modell, de fejleszthető

---

### 🧪 test_best.py

- Inference script  
- képre predikció  

👉 még nincs teljesen letesztelve

---

## 📌 Feladatok

### ✅ Adatelőkészítés
✔ Kész

---

### 🔄 Adat augmentáció

Cél: robusztus modell

Lehetőségek:
- saját pipeline (OpenCV, Albumentations)  
- YOLO beépített augmentáció  

Ötletek:
- blur  
- forgatás  
- zaj  
- graffiti / matricák  
- duplikálás  
- synthetic image generation  

👤 Felelős: Zalán (+ Feri ha beszáll)

---

### 🧠 Tanítás

Fontos:
- YOLO dokumentáció  
- GitHub repo-k  
- hyperparameter tuning  

Modellek:
- YOLOv5 / YOLOv8  

Sok különböző futtatás kell:
- különböző modellek  
- különböző configok  

⚠️ Colab nem ment → manuálisan le kell tölteni!

👤 Felelős: Kristóf, Zsombi

---

### 📊 Kiértékelés

YOLO ad:
- loss görbék  
- confusion matrix  
- mAP  

Feladat:
- modellek összehasonlítása  
- robusztusság vizsgálat  

👤 Felelős: Gergő

---

## 💡 Extra ötletek

### 🚧 Sebességkorlátozó táblák (Ez lehet hogy inkább kötelező)

Jelenleg csak azt mondja meg, hogy "speed limit sign".

Lehetne:
- külön modell a szám felismerésére  

Dataset:
- GTSRB

---

### 🚗 Live teszt

- kamera + valós idejű felismerés  
- autózás közben  

👉 Dokumentációban nagyon jól mutatna 😄
