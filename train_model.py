"""
train_model.py — QuickDraw CNN Training Pipeline
Downloads 10 categories from Google QuickDraw, trains a CNN, saves model + categories.
"""

import os
import numpy as np
import urllib.request
from tensorflow import keras
from tensorflow.keras import layers

# ─── Configuration ───────────────────────────────────────────────
CATEGORIES = [
    "apple", "bicycle", "cat", "dog", "fish",
    "house", "star", "tree", "umbrella", "smiley face"
]
SAMPLES_PER_CATEGORY = 10000
IMG_SIZE = 28
EPOCHS = 15
BATCH_SIZE = 128
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "quickdraw_model.keras")
CATEGORIES_PATH = os.path.join(MODEL_DIR, "categories.txt")
DATA_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"


def download_data():
    """Download .npy files for each category from Google QuickDraw."""
    os.makedirs("data", exist_ok=True)
    X, y = [], []

    for idx, category in enumerate(CATEGORIES):
        filename = category.replace(" ", "%20") + ".npy"
        filepath = os.path.join("data", category + ".npy")

        if not os.path.exists(filepath):
            url = DATA_URL + filename
            print(f"  Downloading: {category}...")
            urllib.request.urlretrieve(url, filepath)
        else:
            print(f"  Found cached: {category}")

        data = np.load(filepath)
        data = data[:SAMPLES_PER_CATEGORY]
        X.append(data)
        y.extend([idx] * len(data))

    X = np.concatenate(X, axis=0)
    y = np.array(y)
    return X, y


def build_model(num_classes):
    """Build a 3-layer CNN with batch normalization and dropout."""
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("\n📦 Step 1: Downloading QuickDraw data...\n")
    X, y = download_data()

    # Preprocess
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0

    # Shuffle
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # Train/test split (80/20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}")

    print("\n🏗️  Step 2: Building CNN model...\n")
    model = build_model(num_classes=len(CATEGORIES))
    model.summary()

    print("\n🚀 Step 3: Training...\n")
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    print("\n📊 Step 4: Evaluating...\n")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\n  Test Accuracy: {accuracy:.4f}")
    print(f"  Test Loss:     {loss:.4f}")

    print("\n💾 Step 5: Saving model...\n")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"  Model saved to: {MODEL_PATH}")

    with open(CATEGORIES_PATH, "w") as f:
        for cat in CATEGORIES:
            f.write(cat + "\n")
    print(f"  Categories saved to: {CATEGORIES_PATH}")

    print("\n✅ Done! Run 'python app.py' to start the server.\n")


if __name__ == "__main__":
    main()
