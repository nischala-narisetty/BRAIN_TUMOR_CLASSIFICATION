"""
train.py
--------
Train a MobileNetV2-based brain tumour classifier on the prepared small
dataset (data_small/).  Optimised for CPU-only execution.

Usage:
    python train.py [--data_dir data_small] [--epochs 5] [--batch_size 8] [--img_size 128]
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# CLI arguments  (defined early so we can configure TF before it imports)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train Brain Tumour Classifier")
parser.add_argument("--data_dir",   default="data_small", help="Prepared dataset dir")
parser.add_argument("--epochs",     type=int, default=5,   help="Max training epochs")
parser.add_argument("--batch_size", type=int, default=8,   help="Batch size")
parser.add_argument("--img_size",   type=int, default=128, help="Image size (square)")
args = parser.parse_args()

# CPU-only: limit TF threads to avoid overloading laptop cores
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"]       = "4"
os.environ["TF_NUM_INTRAOP_THREADS"]= "4"
os.environ["TF_NUM_INTEROP_THREADS"]= "2"
os.environ["KERAS_HOME"]            = os.path.join(os.path.expanduser("~"), ".keras")

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = args.data_dir
EPOCHS      = args.epochs
BATCH_SIZE  = args.batch_size
IMG_SIZE    = args.img_size

ARTIFACTS   = "artifacts"
RESULTS     = "results"
os.makedirs(ARTIFACTS, exist_ok=True)
os.makedirs(RESULTS,   exist_ok=True)

MODEL_PATH  = os.path.join(ARTIFACTS, "best_model.keras")
CLASS_PATH  = os.path.join(ARTIFACTS, "class_names.json")
METRICS_PATH= os.path.join(ARTIFACTS, "test_metrics.json")
HISTORY_PATH= os.path.join(ARTIFACTS, "history.json")

# ---------------------------------------------------------------------------
# Validate dataset
# ---------------------------------------------------------------------------
for split in ["train", "val", "test"]:
    split_dir = os.path.join(DATA_DIR, split)
    if not os.path.isdir(split_dir):
        print(f"[ERROR] '{split_dir}' not found.")
        print(f"  Run:  python prepare_data.py  first.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Discover class names from training split
# ---------------------------------------------------------------------------
train_dir = os.path.join(DATA_DIR, "train")
class_names = sorted([
    d for d in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, d))
])
NUM_CLASSES = len(class_names)
print(f"\n[INFO] Classes ({NUM_CLASSES}): {class_names}")

with open(CLASS_PATH, "w") as f:
    json.dump(class_names, f)

# ---------------------------------------------------------------------------
# Data pipeline  — tf.keras.utils.image_dataset_from_directory
# ---------------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

def build_dataset(split: str, shuffle: bool):
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, split),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        class_names=class_names,
        shuffle=shuffle,
        seed=42,
    )
    return ds

train_ds = build_dataset("train", shuffle=True)
val_ds   = build_dataset("val",   shuffle=False)
test_ds  = build_dataset("test",  shuffle=False)

# Normalise to [0, 1]  (MobileNetV2 expects [-1, 1] via preprocess_input;
#  we apply it inside the model so images stay portable)
def preprocess(img, label):
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label

train_ds = (train_ds
            .map(preprocess, num_parallel_calls=AUTOTUNE)
            .cache()
            .prefetch(AUTOTUNE))

val_ds   = (val_ds
            .map(preprocess, num_parallel_calls=AUTOTUNE)
            .cache()
            .prefetch(AUTOTUNE))

test_ds  = (test_ds
            .map(preprocess, num_parallel_calls=AUTOTUNE)
            .prefetch(AUTOTUNE))

# ---------------------------------------------------------------------------
# Count samples for class-weight computation
# ---------------------------------------------------------------------------
label_counts = np.zeros(NUM_CLASSES, dtype=int)
for _, labels in train_ds.unbatch():
    label_counts[int(labels.numpy())] += 1

total = label_counts.sum()
class_weight = {i: total / (NUM_CLASSES * cnt) if cnt > 0 else 1.0
                for i, cnt in enumerate(label_counts)}
print(f"[INFO] Train samples per class: {dict(zip(class_names, label_counts.tolist()))}")
print(f"[INFO] Class weights: {class_weight}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(num_classes: int, img_size: int) -> tf.keras.Model:
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False          # freeze backbone

    inputs  = tf.keras.Input(shape=(img_size, img_size, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(128, activation="relu")(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="BrainTumour_MobileNetV2")

model = build_model(NUM_CLASSES, IMG_SIZE)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
cb_list = [
    callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1
    ),
    callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    ),
]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
print(f"\n[INFO] Starting training — {EPOCHS} epochs, batch={BATCH_SIZE}, img={IMG_SIZE}x{IMG_SIZE}")
t0 = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cb_list,
    class_weight=class_weight,
    verbose=1,
)

elapsed = time.time() - t0
print(f"\n[INFO] Training completed in {elapsed/60:.1f} min")

# Save history
hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(HISTORY_PATH, "w") as f:
    json.dump(hist_dict, f, indent=2)

# ---------------------------------------------------------------------------
# Plots — Accuracy & Loss
# ---------------------------------------------------------------------------
def plot_history(hist: dict, results_dir: str):
    epochs_range = range(1, len(hist["accuracy"]) + 1)

    # Accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, hist["accuracy"],     label="Train Accuracy", marker="o")
    ax.plot(epochs_range, hist["val_accuracy"], label="Val Accuracy",   marker="o")
    ax.set_title("Training & Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "accuracy.png"), dpi=100)
    plt.close(fig)

    # Loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, hist["loss"],     label="Train Loss", marker="o")
    ax.plot(epochs_range, hist["val_loss"], label="Val Loss",   marker="o")
    ax.set_title("Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "loss.png"), dpi=100)
    plt.close(fig)

plot_history(hist_dict, RESULTS)
print("[INFO] Accuracy and loss plots saved.")

# ---------------------------------------------------------------------------
# Evaluation on test set
# ---------------------------------------------------------------------------
print("\n[INFO] Evaluating on test set …")

# Load best model weights that were saved
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("[WARNING] best_model.keras not found, using last weights.")

test_loss, test_acc = model.evaluate(test_ds, verbose=1)

# Collect true labels and predictions
y_true, y_pred = [], []
for imgs, labels in test_ds:
    preds = model.predict(imgs, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy().tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
fig.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(NUM_CLASSES),
    yticks=np.arange(NUM_CLASSES),
    xticklabels=class_names,
    yticklabels=class_names,
    title="Confusion Matrix",
    ylabel="True Label",
    xlabel="Predicted Label",
)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
thresh = cm.max() / 2.0
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "confusion_matrix.png"), dpi=100)
plt.close(fig)
print("[INFO] Confusion matrix saved.")

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
report_path = os.path.join(RESULTS, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(f"Test Loss     : {test_loss:.4f}\n")
    f.write(f"Test Accuracy : {test_acc:.4f}\n\n")
    f.write(report)

print(f"\n{'='*50}")
print("TEST RESULTS")
print(f"{'='*50}")
print(f"  Loss     : {test_loss:.4f}")
print(f"  Accuracy : {test_acc:.4f}")
print(f"\n{report}")

# Per-class metrics
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None,
                                                    labels=list(range(NUM_CLASSES)))
metrics = {
    "test_loss":     round(float(test_loss), 4),
    "test_accuracy": round(float(test_acc),  4),
    "classes":       class_names,
    "per_class": {
        cls: {
            "precision": round(float(prec[i]), 4),
            "recall":    round(float(rec[i]),  4),
            "f1_score":  round(float(f1[i]),   4),
        }
        for i, cls in enumerate(class_names)
    },
    "confusion_matrix": cm.tolist(),
    "training_history": hist_dict,
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n[OK] All artifacts saved to '{ARTIFACTS}/' and results to '{RESULTS}/'")
print(f"[OK] Model: {MODEL_PATH}")
