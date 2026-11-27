"""
predict.py
----------
Load the saved best_model.keras and predict a brain tumour class for
one MRI image.

Usage:
    python predict.py --image path/to/mri.jpg
"""

import argparse
import json
import os
import sys

import numpy as np

# Suppress TF info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS   = "artifacts"
MODEL_PATH  = os.path.join(ARTIFACTS, "best_model.keras")
CLASS_PATH  = os.path.join(ARTIFACTS, "class_names.json")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Brain Tumour Predictor")
parser.add_argument("--image",    required=True, help="Path to MRI image file")
parser.add_argument("--img_size", type=int, default=128, help="Image size (default 128)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Validate files
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found at '{MODEL_PATH}'.")
    print("  Please run:  python train.py  first.")
    sys.exit(1)

if not os.path.exists(CLASS_PATH):
    print(f"[ERROR] class_names.json not found at '{CLASS_PATH}'.")
    sys.exit(1)

if not os.path.exists(args.image):
    print(f"[ERROR] Image not found: '{args.image}'")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load model & class names
# ---------------------------------------------------------------------------
print(f"[INFO] Loading model from '{MODEL_PATH}' …")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH) as f:
    class_names = json.load(f)

IMG_SIZE = args.img_size

# ---------------------------------------------------------------------------
# Preprocess image
# ---------------------------------------------------------------------------
def preprocess_image(image_path: str, img_size: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)   # shape (1, H, W, 3)

print(f"[INFO] Preprocessing '{args.image}' …")
img_array = preprocess_image(args.image, IMG_SIZE)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
print("[INFO] Running inference …\n")
probs = model.predict(img_array, verbose=0)[0]           # shape (num_classes,)
pred_idx   = int(np.argmax(probs))
pred_class = class_names[pred_idx]
confidence = float(probs[pred_idx]) * 100.0

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
bar = "=" * 50
print(bar)
print("  BRAIN TUMOUR CLASSIFICATION — PREDICTION")
print(bar)
print(f"  Image            : {args.image}")
print(f"  Predicted Class  : {pred_class.upper()}")
print(f"  Confidence       : {confidence:.2f} %")
print(bar)
print("\n  Confidence Scores (all classes):")
for i, cls in enumerate(class_names):
    score  = float(probs[i]) * 100.0
    filled = int(score / 5)
    bar_str = "█" * filled + "░" * (20 - filled)
    print(f"    {cls:<15} [{bar_str}] {score:6.2f} %")
print()

# ---------------------------------------------------------------------------
# Save result JSON (optional helper for the Streamlit app)
# ---------------------------------------------------------------------------
result = {
    "image": args.image,
    "predicted_class": pred_class,
    "confidence": round(confidence, 2),
    "scores": {cls: round(float(probs[i]) * 100, 2)
               for i, cls in enumerate(class_names)},
}
print("[INFO] Result dict:")
print(json.dumps(result, indent=2))
