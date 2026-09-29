"""app.py — backend for the FedAvg Brain MRI dashboard.

Serves:
  GET  /                -> index.html
  GET  /api/metrics      -> contents of metrics.json (client + global stats)
  POST /api/predict      -> runs an uploaded MRI image through the active model

Run from the webapp/ folder:
    pip install flask torch torchvision pillow
    python app.py
Then open http://localhost:5000
"""
import io
import json
import os
import sys

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
import torch
import torchvision.transforms as T

# so we can import models.base_model from the repo root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.base_model import get_base_model

APP_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(APP_DIR, "metrics.json")

# Map each model id -> checkpoint filename (place these .pt files in webapp/models/)
MODEL_DIR = os.path.join(APP_DIR, "models")
MODEL_FILES = {
    "client1": "resnet18.pt",                     # wrapped checkpoint (has "state_dict")
    "client2": "client2_bangladesh_weights.pt",    # plain state_dict
    "client3": "client3_weights.pt",               # plain state_dict
    "global": "global_model.pt",                   # plain state_dict, from FedAvg
}

CLASS_ORDER = ["glioma", "meningioma", "notumor", "pituitary"]

TRANSFORM = T.Compose([
    T.Resize((128, 128)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

app = Flask(__name__, static_folder=None)
_model_cache = {}


def load_state_dict_any(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def get_model(model_id):
    """Loads (and caches) the model for a given id."""
    if model_id in _model_cache:
        return _model_cache[model_id]

    filename = MODEL_FILES.get(model_id)
    if filename is None:
        raise ValueError(f"Unknown model id: {model_id}")

    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint for '{model_id}' not found at {path}. "
            f"Copy the .pt file into webapp/models/."
        )

    sd = load_state_dict_any(path)
    model = get_base_model(arch="resnet18", num_classes=4, pretrained=False)
    model.load_state_dict(sd, strict=True)
    model.eval()
    _model_cache[model_id] = model
    return model


@app.route("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.route("/api/metrics")
def metrics():
    with open(METRICS_PATH) as f:
        return jsonify(json.load(f))


@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"detail": "No file uploaded."}), 400

    file = request.files["file"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"detail": "Could not read image file."}), 400

    # which model to use: whichever metrics.json marks as active
    with open(METRICS_PATH) as f:
        active_model_id = json.load(f).get("active_model_id", "global")

    try:
        model = get_model(active_model_id)
    except FileNotFoundError as e:
        return jsonify({"detail": str(e)}), 503

    x = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    top_idx = int(probs.argmax())
    return jsonify({
        "predicted_class": CLASS_ORDER[top_idx],
        "confidence": float(probs[top_idx]),
        "model_used": active_model_id,
        "all_probabilities": [
            {"class": c, "prob": float(p)} for c, p in zip(CLASS_ORDER, probs)
        ],
    })


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    app.run(debug=True, port=5000)