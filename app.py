"""
app.py — Flask Backend for Drawing Classifier
Handles canvas image preprocessing, CNN prediction, and debug image saving.
"""

import os
import io
import base64
import datetime
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from tensorflow import keras

# ─── Configuration ───────────────────────────────────────────────
MODEL_PATH = "model/quickdraw_model.keras"
CATEGORIES_PATH = "model/categories.txt"
DEBUG_DIR = "debug_images"
IMG_SIZE = 28
TOP_K = 3

app = Flask(__name__)

# ─── Global Model & Categories ───────────────────────────────────
model = None
categories = []


def load_model():
    """Load trained model and category labels."""
    global model, categories

    # Load model
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        print(f"✓ Model loaded from {MODEL_PATH}")
    else:
        print(f"✗ WARNING: Model not found at {MODEL_PATH}")
        print("  Run 'python train_model.py' first.")

    # Load categories
    if os.path.exists(CATEGORIES_PATH):
        with open(CATEGORIES_PATH, "r") as f:
            categories = [line.strip() for line in f.readlines()]
        print(f"✓ Categories loaded ({len(categories)} classes)")
    else:
        print(f"✗ WARNING: Categories file not found at {CATEGORIES_PATH}")


def get_model():
    """Ensure model is loaded (works for Render/Gunicorn)."""
    global model
    if model is None:
        load_model()
    return model


# ─── Image Preprocessing Pipeline ───────────────────────────────
def preprocess_canvas_image(data_url):
    """Decode base64 canvas → preprocess → return 28×28 tensor."""
    header, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # Composite onto white background
    white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(white_bg, image)

    # Grayscale
    grayscale = composite.convert("L")

    # Invert
    inverted = 255 - np.array(grayscale)

    # Crop to bounding box
    coords = np.argwhere(inverted > 20)
    if coords.size == 0:
        return np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype="float32")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped = inverted[y_min:y_max + 1, x_min:x_max + 1]

    # Pad to square
    h, w = cropped.shape
    max_dim = max(h, w)
    margin = int(max_dim * 0.15)
    padded_size = max_dim + 2 * margin

    padded = np.zeros((padded_size, padded_size), dtype="uint8")
    y_offset = (padded_size - h) // 2
    x_offset = (padded_size - w) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = cropped

    # Resize to 28×28
    final = Image.fromarray(padded).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    final_array = np.array(final).astype("float32") / 255.0

    return final_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def save_debug_image(processed_image, prediction_label):
    """Save the preprocessed 28×28 image for debugging."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{prediction_label}.png"
    filepath = os.path.join(DEBUG_DIR, filename)

    img_array = (processed_image.reshape(IMG_SIZE, IMG_SIZE) * 255).astype("uint8")
    Image.fromarray(img_array).save(filepath)
    return filepath


# ─── Routes ──────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept base64 canvas → preprocess → predict → return top-3."""
    model = get_model()

    if model is None or not categories:
        return jsonify({"error": "Model or categories not loaded."}), 503

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided."}), 400

    try:
        processed = preprocess_canvas_image(data["image"])
        predictions = model.predict(processed, verbose=0)[0]

        # Top-K
        top_indices = predictions.argsort()[-TOP_K:][::-1]
        results = [
            {
                "label": categories[idx] if idx < len(categories) else f"class_{idx}",
                "confidence": round(float(predictions[idx]) * 100, 2),
            }
            for idx in top_indices
        ]

        # Save debug image
        save_debug_image(processed, results[0]["label"])

        return jsonify({"predictions": results})

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500


# ─── Load model at import time (REQUIRED FOR RENDER) ─────────────
load_model()


# ─── Local Development ───────────────────────────────────────────
if __name__ == "__main__":
    print("\n🎨 Drawing Classifier — Starting Local Server\n")
    print("Model loaded. Running at http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
