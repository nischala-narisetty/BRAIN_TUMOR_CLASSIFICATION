"""
prepare_data.py
---------------
Scans the original dataset/, detects the 4 brain-tumor classes, selects a
balanced subset of up to 100 images per class, then performs a stratified
70 / 15 / 15 train / val / test split and copies everything into data_small/.

Usage:
    python prepare_data.py [--src dataset] [--dst data_small] [--max_per_class 100] [--seed 42]
"""

import argparse
import json
import os
import random
import shutil
import sys

# ---------------------------------------------------------------------------
# Class name normalisation map  (all keys are lower-case)
# ---------------------------------------------------------------------------
CLASS_ALIASES = {
    "glioma": "glioma",
    "glioma_tumor": "glioma",
    "meningioma": "meningioma",
    "meningioma_tumor": "meningioma",
    "pituitary": "pituitary",
    "pituitary_tumor": "pituitary",
    "notumor": "notumor",
    "no_tumor": "notumor",
    "no tumor": "notumor",
    "normal": "notumor",
}

REQUIRED_CLASSES = {"glioma", "meningioma", "pituitary", "notumor"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_class(folder_name: str) -> str | None:
    """Return canonical class name or None if unrecognised."""
    return CLASS_ALIASES.get(folder_name.strip().lower().replace(" ", "_"))


def collect_images(split_dir: str) -> dict[str, list[str]]:
    """Return {canonical_class: [abs_image_paths]} for one split directory."""
    result: dict[str, list[str]] = {}
    if not os.path.isdir(split_dir):
        return result
    for folder in os.listdir(split_dir):
        cls = normalise_class(folder)
        if cls is None:
            continue
        folder_path = os.path.join(split_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        imgs = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ]
        result.setdefault(cls, []).extend(imgs)
    return result


def copy_images(image_list: list[str], dest_dir: str):
    """Copy images into dest_dir, renaming to avoid collisions."""
    os.makedirs(dest_dir, exist_ok=True)
    for i, src in enumerate(image_list):
        ext = os.path.splitext(src)[1]
        dst = os.path.join(dest_dir, f"img_{i:05d}{ext}")
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare small balanced dataset")
    parser.add_argument("--src", default="dataset", help="Original dataset root")
    parser.add_argument("--dst", default="data_small", help="Output directory")
    parser.add_argument("--max_per_class", type=int, default=100,
                        help="Max images per class (across train+val; test kept separate)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    src = args.src
    dst = args.dst

    if not os.path.isdir(src):
        print(f"[ERROR] Source dataset directory '{src}' not found.")
        print("  Please place your dataset inside the 'dataset/' folder and re-run.")
        sys.exit(1)

    # Discover sub-directories
    sub_dirs = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
    has_training = any(d.lower() == "training" for d in sub_dirs)
    has_testing  = any(d.lower() == "testing"  for d in sub_dirs)

    print(f"\n[INFO] Source : {os.path.abspath(src)}")
    print(f"[INFO] Destination : {os.path.abspath(dst)}")
    print(f"[INFO] Max images per class (train+val pool) : {args.max_per_class}")
    print(f"[INFO] Random seed : {args.seed}\n")

    # ------------------------------------------------------------------
    # Collect images
    # ------------------------------------------------------------------
    if has_training:
        train_pool = collect_images(os.path.join(src, "Training"))
    else:
        # Flat dataset — treat whole src as the pool
        train_pool = collect_images(src)

    if has_testing:
        test_pool = collect_images(os.path.join(src, "Testing"))
    else:
        test_pool = {}

    # Merge test into train pool for unified splitting when no Testing dir
    if not has_testing:
        all_pool = train_pool
    else:
        all_pool = train_pool  # test handled separately

    # Validate classes
    found = set(all_pool.keys()) | set(test_pool.keys())
    missing = REQUIRED_CLASSES - found
    if missing:
        print(f"[WARNING] The following required classes were NOT found: {missing}")
        print(f"[INFO]    Found classes: {found}")

    if not found:
        print("[ERROR] No recognisable image classes found in the dataset.")
        sys.exit(1)

    # Clean destination — handle OneDrive/junction-linked dirs on Windows
    if os.path.exists(dst):
        try:
            shutil.rmtree(dst)
        except PermissionError:
            # Fallback: delete contents file-by-file then rmdir
            for root_dir, dirs, files in os.walk(dst, topdown=False):
                for fname in files:
                    try:
                        os.remove(os.path.join(root_dir, fname))
                    except Exception:
                        pass
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root_dir, d))
                    except Exception:
                        pass
            try:
                os.rmdir(dst)
            except Exception:
                pass
    os.makedirs(dst, exist_ok=True)

    summary: dict[str, dict[str, int]] = {}

    for cls in sorted(found & REQUIRED_CLASSES):
        print(f"--- Class: {cls} ---")

        # --- Train / Val pool ---
        pool = list(all_pool.get(cls, []))
        rng.shuffle(pool)
        pool = pool[: args.max_per_class]  # cap at max_per_class

        if len(pool) < 5:
            print(f"  [ERROR] Only {len(pool)} images available for '{cls}'. "
                  "Need at least 5. Please expand your dataset.")
            continue

        # Stratified 70/15/15 on pool (train+val side)
        n_train = max(1, int(round(len(pool) * 0.70)))
        n_val   = max(1, int(round(len(pool) * 0.15)))
        # rest goes to val to keep it clean if no sep Testing dir
        train_imgs = pool[:n_train]
        val_imgs   = pool[n_train: n_train + n_val]

        # --- Test ---
        if has_testing:
            test_imgs = list(test_pool.get(cls, []))
            rng.shuffle(test_imgs)
            # cap test to same proportion (15 %)
            test_cap = max(1, int(round(args.max_per_class * 0.15)))
            test_imgs = test_imgs[:test_cap]
        else:
            test_imgs = pool[n_train + n_val:]

        # Copy
        copy_images(train_imgs, os.path.join(dst, "train", cls))
        copy_images(val_imgs,   os.path.join(dst, "val",   cls))
        copy_images(test_imgs,  os.path.join(dst, "test",  cls))

        summary[cls] = {
            "train": len(train_imgs),
            "val":   len(val_imgs),
            "test":  len(test_imgs),
            "total": len(train_imgs) + len(val_imgs) + len(test_imgs),
        }

        print(f"  Train : {len(train_imgs)}")
        print(f"  Val   : {len(val_imgs)}")
        print(f"  Test  : {len(test_imgs)}")
        print()

    # Save summary
    with open(os.path.join(dst, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    totals = {
        split: sum(summary[c][split] for c in summary)
        for split in ["train", "val", "test", "total"]
    }

    print("=" * 45)
    print("DATASET SUMMARY")
    print("=" * 45)
    for cls, counts in summary.items():
        print(f"  {cls:<15}  train={counts['train']:3d}  val={counts['val']:3d}  test={counts['test']:3d}")
    print("-" * 45)
    print(f"  {'TOTAL':<15}  train={totals['train']:3d}  val={totals['val']:3d}  test={totals['test']:3d}")
    print("=" * 45)
    print(f"\n[OK] Small dataset saved to '{dst}/'")


if __name__ == "__main__":
    main()
