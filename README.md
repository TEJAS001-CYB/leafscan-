# 🌿 LeafScan — Plant Disease Detection using Deep Learning

A full-stack AI application that detects plant leaf diseases from real-world images using a fine-tuned **EfficientNetB3** model trained on the PlantVillage dataset.

---

## 🚀 Live Demo

* 🌐 Hugging Face Space: https://huggingface.co/spaces/tktejask/leafscan

---

## 🧠 Project Overview

LeafScan is a real-time plant disease detection system that:

* Accepts **real-world leaf images**
* Detects **38 disease classes + 1 non-leaf class**
* Provides:

  * Disease name
  * Confidence score
  * Severity
  * Description
  * Treatment suggestion
  * Top-5 predictions

---

## 📊 Dataset

* Source: PlantVillage Dataset
* Link: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
* Size: ~54,000 images
* Classes: 38 diseases + healthy + 1 synthetic "not a leaf" class

---

## 🏗️ Model Architecture

```
Input Image (300×300)
        ↓
EfficientNetB3 (Pretrained on ImageNet)
        ↓
Feature Vector (1536)
        ↓
Custom Head:
  Dense → GELU → Dropout
  Dense → GELU → Dropout
  Output Layer (39 classes)
        ↓
Softmax Probabilities
```

---

## ⚙️ Training Strategy

| Phase   | Description                |
| ------- | -------------------------- |
| Phase 1 | Train only classifier head |
| Phase 2 | Unfreeze last layers       |
| Phase 3 | Full fine-tuning           |

Techniques used:

* Transfer Learning
* Test Time Augmentation (TTA ×6)
* AdamW optimizer
* Label smoothing
* Class balancing

---

## 🔬 Inference Pipeline

```
Input Image
   ↓
Preprocessing (Resize → Normalize)
   ↓
Model Prediction
   ↓
TTA Averaging
   ↓
Confidence + Decision Logic
   ↓
Final Output + Top-5 Classes
```

---

## 🧪 Features

* ✅ Works on **real-world images (not just dataset)**
* ✅ Detects **non-leaf images**
* ✅ REST API support
* ✅ Beautiful frontend UI
* ✅ Deployable locally + cloud

---

## 🖥️ Local Deployment

### 1. Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### 2. Run server

```bash
python app.py
```

---

### 3. Output

```
Model ready.
 * Running on http://127.0.0.1:7860
```

---

### 4. Open in browser

```
http://localhost:7860
```

---

## 🌐 Hugging Face Deployment

* Platform: Hugging Face Spaces
* Runtime: Flask (Docker/Spaces)
* URL: https://huggingface.co/spaces/tktejask/leafscan

### What was done:

* Uploaded model + backend + frontend
* Configured app.py to run on port 7860
* Added README config block

---

## 🔗 API Endpoints

| Endpoint              | Description         |
| --------------------- | ------------------- |
| `/api/predict`        | Upload image        |
| `/api/predict-url`    | Predict from URL    |
| `/api/predict-base64` | Predict from base64 |
| `/api/classes`        | List classes        |
| `/api/health`         | Server status       |

---

## 📈 Model Performance

* Accuracy: ~96% (on PlantVillage test set)
* Supports: 38 disease classes
* Handles real-world noise via TTA

---

## ⚠️ Limitations

* Trained on controlled dataset → real-world variation may reduce accuracy
* Needs clear leaf image
* Heavy model → slow on CPU

---

## 🔥 Key Highlights (Interview Points)

* Built **end-to-end ML system**
* Used **transfer learning (EfficientNetB3)**
* Implemented **TTA for robustness**
* Designed **Flask API + frontend integration**
* Deployed on **Hugging Face Spaces**
* Handled **real-world inference issues**

---

## 📦 Project Structure

```
leaf scan/
├── app.py
├── model.py
├── predict.py
├── metrics.py
├── models/
│   └── best_model.pth
├── data/
│   └── classes.txt
├── frontend/
│   └── index.html
└── requirements.txt
```

---

## 🛠️ Tech Stack

* Python
* PyTorch
* timm
* Flask
* HTML/CSS/JS
* Hugging Face Spaces

---

## 📄 License

Educational project. Dataset is public (PlantVillage).
