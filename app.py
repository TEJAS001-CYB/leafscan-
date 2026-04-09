"""
app.py
------
Flask REST API backend for Leaf Disease Detector.
"""

import argparse
import base64
import io
import time
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

from predict import LeafDiseasePredictor

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})

MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

_predictor = None


def get_predictor() -> LeafDiseasePredictor:
    global _predictor
    if _predictor is None:
        _predictor = LeafDiseasePredictor()
    return _predictor


# ─── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def error_response(message: str, code: int = 400):
    return jsonify({"success": False, "error": message}), code


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = f(*args, **kwargs)
        elapsed = (time.time() - t0) * 1000
        try:
            data = result[0].get_json()
            if data:
                data["inference_ms"] = round(elapsed, 1)
                return jsonify(data), result[1]
        except Exception:
            pass
        return result
    return wrapper


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    frontend_path = Path("frontend/index.html")
    if frontend_path.exists():
        return send_from_directory("frontend", "index.html")
    return jsonify({"error": "Frontend not found"}), 404


@app.route("/api/health", methods=["GET"])
def health():
    try:
        p = get_predictor()
        return jsonify({
            "status": "ok",
            "num_classes": p.num_classes,
            "device": str(p.device),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 503


@app.route("/api/classes", methods=["GET"])
def list_classes():
    predictor = get_predictor()
    classes_info = []

    for cls in predictor.classes:
        info = predictor.disease_info.get(cls, {})
        parts = cls.split("___")

        classes_info.append({
            "class_id": cls,
            "plant": parts[0].replace("_", " ") if len(parts) > 0 else cls,
            "disease": parts[1].replace("_", " ") if len(parts) > 1 else "",
            "severity": info.get("severity", "Unknown"),
        })

    return jsonify({"success": True, "classes": classes_info, "count": len(classes_info)})


@app.route("/api/predict", methods=["POST"])
@timing
def predict_file():
    if "image" not in request.files:
        return error_response("No image file provided.")

    file = request.files["image"]

    if file.filename == "":
        return error_response("Empty filename.")

    if not allowed_file(file.filename):
        return error_response("Unsupported file type.")

    data = file.read()

    if len(data) > MAX_FILE_SIZE:
        return error_response("File too large.")

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        return error_response(f"Cannot open image: {e}")

    try:
        predictor = get_predictor()
        result = predictor.predict(img)

        thumb = img.copy()
        thumb.thumbnail((300, 300))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=75)

        result["thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        return jsonify({"success": True, "result": result}), 200

    except Exception as e:
        return error_response(f"Prediction failed: {e}", 500)


@app.route("/api/predict-url", methods=["POST"])
@timing
def predict_url():
    body = request.get_json(silent=True)

    if not body or "url" not in body:
        return error_response("URL required.")

    url = body["url"]

    try:
        predictor = get_predictor()
        result = predictor.predict(url)
        return jsonify({"success": True, "result": result}), 200

    except Exception as e:
        return error_response(f"Prediction failed: {e}", 500)


@app.route("/api/predict-base64", methods=["POST"])
@timing
def predict_base64():
    body = request.get_json(silent=True)

    if not body or "image" not in body:
        return error_response("Base64 image required.")

    try:
        b64 = body["image"].split(",")[-1]
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    except Exception as e:
        return error_response(f"Invalid base64: {e}")

    try:
        predictor = get_predictor()
        result = predictor.predict(img)
        return jsonify({"success": True, "result": result}), 200

    except Exception as e:
        return error_response(f"Prediction failed: {e}", 500)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("🌿 LeafScan API starting...")
    
    get_predictor()

    app.run(host=args.host, port=args.port, debug=args.debug)