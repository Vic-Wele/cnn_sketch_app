# Drawing Classifier

**AI-powered sketch recognition using a CNN trained on Google's QuickDraw dataset.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Draw on a canvas and let a Convolutional Neural Network predict what you sketched — with top-3 predictions and animated confidence bars.

---

## Features

- **Interactive Drawing Canvas** — Mouse and touch (mobile) support
- **Real-time CNN Predictions** — Top-3 results with confidence percentages
- **Animated Confidence Bars** — Visual feedback with color-coded rankings
- **Debug Image Saving** — Preprocessed 28x28 PNGs saved automatically
- **Dark Mode UI** — Clean, minimal, responsive design
- **One-click Deploy** — Ready for Render with `render.yaml`

---

## Categories

The model recognizes 10 sketch categories:

`apple` · `bicycle` · `cat` · `dog` · `fish` · `house` · `star` · `tree` · `umbrella` · `smiley face`

---

## Project Structure

```
drawing-classifier/
├── app.py                  # Flask backend + /predict API
├── train_model.py          # CNN training pipeline
├── model/                  # Trained model (generated)
├── static/
│   ├── css/style.css       # Dark-mode responsive styles
│   └── js/app.js           # Canvas engine + prediction UI
├── templates/
│   └── index.html          # Main page
├── debug_images/           # Preprocessed images (generated)
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment config
├── Procfile                # Gunicorn start command
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/drawing-classifier.git
cd drawing-classifier
```

### 2. Set up virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

This downloads QuickDraw data (~200 MB), trains the CNN for 15 epochs, and saves the model to `model/quickdraw_model.keras`.

### 4. Run the server

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Tech Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| Backend         | Flask 3.1                           |
| ML Framework    | TensorFlow / Keras 2.19            |
| Frontend        | Vanilla JavaScript (ES6)           |
| Styling         | Custom CSS (dark mode, responsive) |
| Dataset         | Google QuickDraw (10K per category) |
| Deployment      | Render (gunicorn)                  |

---

## API Reference

### `POST /predict`

**Request:**

```json
{
  "image": "data:image/png;base64,iVBORw0KGgo..."
}
```

**Response:**

```json
{
  "predictions": [
    { "label": "cat", "confidence": 92.45 },
    { "label": "dog", "confidence": 4.12 },
    { "label": "fish", "confidence": 1.88 }
  ]
}
```

---

## Model Architecture

```
Conv2D(32) → BatchNorm → ReLU → MaxPool
Conv2D(64) → BatchNorm → ReLU → MaxPool
Conv2D(128) → BatchNorm → ReLU → MaxPool
Flatten → Dense(256) → Dropout(0.4) → Dense(10, softmax)
```

---

## Preprocessing Pipeline

1. Decode base64 canvas image
2. Composite onto white background (handle transparency)
3. Convert to grayscale
4. Invert colors (dark strokes on white → white on black)
5. Crop to bounding box of the drawing
6. Pad to square with 15% margin
7. Resize to 28x28 pixels
8. Normalize pixel values to [0, 1]

---

## Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and configures everything
5. Click **Deploy**

---

## Debug Images

Every prediction saves the preprocessed 28x28 input image to `debug_images/` with a timestamp and predicted label:

```
debug_images/
├── 20260419_141523_123456_cat.png
├── 20260419_141530_789012_star.png
└── ...
```

Use these to inspect how the preprocessing pipeline transforms your canvas drawing before it reaches the model.

---

## License

MIT License — free for personal and commercial use.
