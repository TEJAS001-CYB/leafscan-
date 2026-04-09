"""
predict.py — FIXED (PRODUCTION VERSION)

Major fixes:
1. Removed over-strict rejection logic
2. Lowered confidence threshold (0.65 → 0.40)
3. Top-2 gap based decision (more reliable)
4. Reduced TTA (6 → 3 transforms)
5. Never reject obvious leaves
6. Better handling of low-confidence predictions
"""

import json
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import build_model

# ─── CONFIG ───────────────────────────────────────────────────────────────

MODEL_PATH = Path("models/best_model.pth")
CLASSES_PATH = Path("data/classes.txt")
DISEASE_INFO_PATH = Path("data/disease_info.json")

IMG_SIZE = 300
RESIZE_TO = 332

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 🔥 FIXED THRESHOLDS
CONF_THRESHOLD = 0.40       # was 0.65 ❌
TOP2_GAP_THRESHOLD = 0.15   # was 0.25 ❌
NOT_LEAF_CLASS = "not_a_leaf"

USE_TTA = True


# ─── MODEL ────────────────────────────────────────────────────────────────

class LeafDiseasePredictor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._load()

    def _load(self):
        print("Loading model...")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load classes
        with open(CLASSES_PATH) as f:
            self.classes = [x.strip() for x in f if x.strip()]

        self.num_classes = len(self.classes)

        # Load model
        self.model = build_model(self.num_classes, pretrained=False)
        ckpt = torch.load(MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        # Disease info
        if DISEASE_INFO_PATH.exists():
            with open(DISEASE_INFO_PATH) as f:
                self.disease_info = json.load(f)
        else:
            self.disease_info = {}

        # Transform (correct)
        self.transform = transforms.Compose([
            transforms.Resize((RESIZE_TO, RESIZE_TO)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        # 🔥 REDUCED TTA (3 instead of 6)
        self.tta_transforms = [
            self.transform,
            transforms.Compose([
                transforms.Resize((RESIZE_TO, RESIZE_TO)),
                transforms.CenterCrop(IMG_SIZE),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]),
            transforms.Compose([
                transforms.Resize((RESIZE_TO, RESIZE_TO)),
                transforms.RandomCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]),
        ]

        print("Model ready.")

    # ─── IMAGE LOADING ─────────────────────────────────────────────────────

    def _load_image(self, source):
        if isinstance(source, Image.Image):
            return source.convert("RGB")

        if isinstance(source, np.ndarray):
            return Image.fromarray(source).convert("RGB")

        source = str(source)

        if source.startswith("http"):
            with urllib.request.urlopen(source) as r:
                return Image.open(BytesIO(r.read())).convert("RGB")

        return Image.open(source).convert("RGB")

    # ─── PREDICTION CORE ───────────────────────────────────────────────────

    @torch.no_grad()
    def _predict_probs(self, img):
        probs_all = []

        if USE_TTA:
            for tf in self.tta_transforms:
                x = tf(img).unsqueeze(0).to(self.device)
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                probs_all.append(probs)

            return np.mean(probs_all, axis=0)

        else:
            x = self.transform(img).unsqueeze(0).to(self.device)
            logits = self.model(x)
            return F.softmax(logits, dim=-1).cpu().numpy()[0]

    # ─── MAIN PREDICT ──────────────────────────────────────────────────────

    def predict(self, source) -> Dict:
        try:
            img = self._load_image(source)
        except Exception as e:
            return self._error(f"Invalid image: {e}")

        probs = self._predict_probs(img)

        # Top-5
        top5_idx = probs.argsort()[::-1][:5]
        top5 = [
            {"class": self.classes[i], "probability": float(probs[i])}
            for i in top5_idx
        ]

        pred_idx = int(probs.argmax())
        pred_cls = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        # Top-2 gap
        second_prob = float(probs[top5_idx[1]])
        gap = confidence - second_prob

        # ─────────────────────────────────────────
        # 🔥 NEW DECISION LOGIC (CORE FIX)
        # ─────────────────────────────────────────

        # Case 1: VERY CLEAR prediction → accept
        if confidence > CONF_THRESHOLD and gap > TOP2_GAP_THRESHOLD:
            is_leaf = True

        # Case 2: Medium confidence but still reasonable → accept with warning
        elif confidence > 0.30:
            is_leaf = True

        # Case 3: Very low confidence → only then reject
        else:
            return self._not_leaf(top5, probs, confidence)

        # ─────────────────────────────────────────
        # Parse result
        # ─────────────────────────────────────────

        parts = pred_cls.split("___")
        plant = parts[0].replace("_", " ")
        disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

        info = self.disease_info.get(pred_cls, {})

        warning = None
        if confidence < 0.50:
            warning = "Low confidence — try another image for confirmation."

        return {
            "is_leaf": is_leaf,
            "predicted_class": pred_cls,
            "plant": plant,
            "disease": disease,
            "confidence": confidence,
            "confidence_pct": f"{confidence:.1%}",
            "severity": info.get("severity", "Unknown"),
            "description": info.get("description", ""),
            "treatment": info.get("treatment", ""),
            "top5": top5,
            "warning": warning,
        }

    # ─── HELPERS ────────────────────────────────────────────────────────────

    def _not_leaf(self, top5, probs, confidence):
        return {
            "is_leaf": False,
            "predicted_class": NOT_LEAF_CLASS,
            "plant": "N/A",
            "disease": "N/A",
            "confidence": confidence,
            "confidence_pct": f"{confidence:.1%}",
            "severity": "N/A",
            "description": "Image not recognized as a leaf.",
            "treatment": "Upload a clear leaf image.",
            "top5": top5,
            "warning": "Model is unsure — likely not a valid leaf image.",
        }

    def _error(self, msg):
        return {
            "is_leaf": False,
            "error": msg
        }