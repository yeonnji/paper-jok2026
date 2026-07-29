"""
generate_soft_masks.py
======================
Generates per-image soft masks (light-source confidence maps) from RGB images
using HSV and LAB color-space analysis, as described in the thesis.

Algorithm
---------
1. Convert RGB → HSV.  Score_HSV(i,j) = (1 - S) * V
   High S means saturated/non-white → low score.
   High V means bright → high score.
   Bright, desaturated regions (e.g. light sources) get scores near 1.

2. Convert RGB → LAB.  Score_LAB(i,j) = L* / 100
   Normalises CIE L* to [0, 1].

3. Combine:  confidence = Score_HSV * Score_LAB
   (element-wise product; both channels must agree)

4. Normalise to [0, 1] per image and save as uint8 PNG.

Output
------
For each input image at <source_path>/images/<name>.<ext> the script writes
  <source_path>/masks/<name>_soft.png

Usage
-----
  python generate_soft_masks.py -s /path/to/scene
  python generate_soft_masks.py -s /path/to/scene --images images_2 --threshold 0.15
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path


def compute_soft_mask(bgr_image: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Return a float32 soft mask in [0, 1] for a single BGR image.

    Parameters
    ----------
    bgr_image : np.ndarray  shape (H, W, 3), dtype uint8
    threshold : float
        Pixels with confidence below this value are zeroed out.

    Returns
    -------
    mask : np.ndarray  shape (H, W), dtype float32, values in [0, 1]
    """
    # --- HSV score: (1 - S) * V ---
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    S = hsv[:, :, 1] / 255.0   # saturation  [0, 1]
    V = hsv[:, :, 2] / 255.0   # value        [0, 1]
    score_hsv = (1.0 - S) * V  # [0, 1]

    # --- LAB score: L* / 100 ---
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab).astype(np.float32)
    L_star = lab[:, :, 0]      # OpenCV encodes L* in [0, 255] → divide by 255*100/255
    score_lab = L_star / 255.0  # [0, 1]  (L* 0..100 → 0..1 after /255 since OCV uses 0-255)

    # --- Combined confidence ---
    confidence = score_hsv * score_lab  # [0, 1]

    # --- Per-image normalisation ---
    c_max = confidence.max()
    if c_max > 1e-6:
        confidence = confidence / c_max

    # --- Optional threshold ---
    if threshold > 0.0:
        confidence[confidence < threshold] = 0.0

    return confidence.astype(np.float32)


def generate_soft_masks(source_path: str,
                        images_folder: str = "images",
                        threshold: float = 0.0,
                        overwrite: bool = False) -> None:
    images_dir = Path(source_path) / images_folder
    masks_dir  = Path(source_path) / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    image_files = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in image_exts
    )

    if not image_files:
        print(f"[ERROR] No images found in {images_dir}")
        sys.exit(1)

    print(f"Generating soft masks for {len(image_files)} images → {masks_dir}")

    skipped = 0
    for img_path in image_files:
        out_path = masks_dir / f"{img_path.stem}_soft.png"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  [WARN] Could not read {img_path.name}, skipping.")
            continue

        mask = compute_soft_mask(bgr, threshold=threshold)

        # Save as uint8 PNG (0–255)
        mask_uint8 = (mask * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(out_path), mask_uint8)

    total_written = len(image_files) - skipped
    print(f"Done. Written: {total_written}  Skipped (already exist): {skipped}")
    if skipped:
        print("  Run with --overwrite to regenerate existing masks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate HSV/LAB-based soft light-source masks.")
    parser.add_argument("-s", "--source_path", required=True,
                        help="Root directory of the scene (contains 'images/' subfolder).")
    parser.add_argument("--images", default="images",
                        help="Name of the images subfolder (default: 'images').")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Confidence values below this are zeroed out (default: 0.0).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing mask files.")
    args = parser.parse_args()

    generate_soft_masks(
        source_path=args.source_path,
        images_folder=args.images,
        threshold=args.threshold,
        overwrite=args.overwrite,
    )
