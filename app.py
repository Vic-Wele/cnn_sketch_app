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

# ─── Load Model & Categories ────────────────────────────────────
model = None
categories = []


def load_model():
    """Load trained model and category labels."""
    global model, categories
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        print(f"  Model loaded from {MODEL_PATH}")
    else:
        print(f"  WARNING: Model not found at {MODEL_PATH}")
        print(f"  Run 'python train_model.py' first.")

    if os.path.exists(CATEGORIES_PATH):
        with open(CATEGORIES_PATH, "r") as f:
            categories = [line.strip() for line in f.readlines()]
        print(f"  Categories loaded: {categories}")
    else:
        print(f"  WARNING: Categories file not found at {CATEGORIES_PATH}")


def get_model():
    """Lazy-load model (works for both local and Render/Gunicorn)."""
    global model
    if model is None:
        load_model()
    return model


# ─── Image Preprocessing Pipeline ───────────────────────────────
def preprocess_canvas_image(data_url):
    """
    Full preprocessing pipeline:
    1. Decode base64 → PIL Image
    2. Composite onto white background (handle transparency)
    3. Convert to grayscale
    4. Invert colors (white drawing → black on white → white on black)
    5. Crop to bounding box of drawing
    6. Pad to square with margin
    7. Resize to 28×28
    """
    # Step 1: Decode base64
    header, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # Step 2: Composite onto white background
    white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(white_bg, image)

    # Step 3: Convert to grayscale
    grayscale = composite.convert("L")

    # Step 4: Invert (canvas has dark strokes on white → we need white on black)
    inverted = np.array(grayscale)
    inverted = 255 - inverted

    # Step 5: Crop to bounding box
    coords = np.argwhere(inverted > 20)
    if coords.size == 0:
        # Empty canvas — return blank image
        return np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype="float32")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped = inverted[y_min:y_max + 1, x_min:x_max + 1]

    # Step 6: Pad to square with margin
    h, w = cropped.shape
    max_dim = max(h, w)
    margin = int(max_dim * 0.15)
    padded_size = max_dim + 2 * margin

    padded = np.zeros((padded_size, padded_size), dtype="uint8")
    y_offset = (padded_size - h) // 2
    x_offset = (padded_size - w) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = cropped

    # Step 7: Resize to 28×28
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
    """Serve the main drawing page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept canvas image (base64), preprocess, predict, return top-3.
    Request JSON: { "image": "data:image/png;base64,..." }
    Response JSON: { "predictions": [{"label": "cat", "confidence": 0.95}, ...] }
    """
    # Ensure model is loaded (works on Render/Gunicorn too)
    model = get_model()
    if model is None or not categories:
        return jsonify({"error": "Model or categories not loaded."}), 503

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided."}), 400

    try:
        # Preprocess
        processed = preprocess_canvas_image(data["image"])

        # Predict
        predictions = model.predict(processed, verbose=0)[0]

        # Get top-K predictions
        top_indices = predictions.argsort()[-TOP_K:][::-1]
        results = []
        for idx in top_indices:
            label = categories[idx] if idx < len(categories) else f"class_{idx}"
            results.append({
                "label": label,
                "confidence": round(float(predictions[idx]) * 100, 2),
            })

        # Save debug image
        debug_path = save_debug_image(processed, results[0]["label"])
        print(f"  Debug image saved: {debug_path}")

        return jsonify({"predictions": results})

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500


# ─── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🎨 Drawing Classifier — Starting Server\n")
    load_model()
    print(f"\n  Server running at http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
