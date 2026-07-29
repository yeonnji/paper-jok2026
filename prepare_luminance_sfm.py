"""
prepare_luminance_sfm.py
========================
Converts RGB images to luminance (HSV V-channel) grayscale images for
luminance-centric SfM, as described in the thesis (Module 1).

The luminance images are used ONLY as COLMAP input so that camera poses and
the initial point cloud are estimated from illumination-stable features.
3DGS training still uses the original RGB images (--images images flag).

Workflow
--------
1. Run this script to create <source_path>/luminance_images/
2. Run COLMAP on luminance_images/ to produce <source_path>/sparse_luminance/
3. Train 3DGS with --source_path pointing to a directory that has
   sparse_luminance/ as its sparse/ folder, and --images images (original RGB).

Usage
-----
  python prepare_luminance_sfm.py -s /path/to/scene
  python prepare_luminance_sfm.py -s /path/to/scene --images images_2

Then run COLMAP (or convert.py) with:
  --source_path /path/to/scene_luminance_sfm
  where scene_luminance_sfm contains:
    input/  → symlink or copy of luminance_images/

See the ablation study commands in the README for the full pipeline.
"""

import argparse
import os
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path


def rgb_to_luminance_bgr(bgr_image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to a 3-channel grayscale (V channel of HSV).

    Returns a uint8 BGR image where R=G=B=V so COLMAP can read it as a
    standard 3-channel image while containing only luminance information.
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    v   = hsv[:, :, 2]          # V channel, uint8 [0-255]
    return cv2.merge([v, v, v])  # 3-channel grayscale as BGR


def prepare_luminance_images(source_path: str,
                              images_folder: str = "images",
                              output_folder: str = "luminance_images",
                              overwrite: bool = False) -> None:
    images_dir  = Path(source_path) / images_folder
    lum_dir     = Path(source_path) / output_folder
    lum_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    image_files = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in image_exts
    )

    if not image_files:
        print(f"[ERROR] No images found in {images_dir}")
        sys.exit(1)

    print(f"Converting {len(image_files)} images → {lum_dir}")

    skipped = 0
    for img_path in image_files:
        # Preserve original filename so COLMAP keeps the same image names
        out_path = lum_dir / img_path.name
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  [WARN] Could not read {img_path.name}, skipping.")
            continue

        lum_bgr = rgb_to_luminance_bgr(bgr)
        cv2.imwrite(str(out_path), lum_bgr)

    print(f"Done. Written: {len(image_files) - skipped}  "
          f"Skipped (already exist): {skipped}")


def print_colmap_commands(source_path: str, output_folder: str = "luminance_images") -> None:
    """Print example COLMAP commands for the luminance SfM pipeline."""
    s = source_path
    o = output_folder
    print("\n" + "=" * 70)
    print("Next steps: run COLMAP on the luminance images")
    print("=" * 70)
    print(f"""
# 1. Create a dedicated scene directory for luminance SfM
mkdir -p {s}/luminance_sfm/input
cp {s}/{o}/* {s}/luminance_sfm/input/

# 2. Run the standard 3DGS converter (or COLMAP directly)
python convert.py -s {s}/luminance_sfm

# This produces:
#   {s}/luminance_sfm/sparse/0/  ← camera poses + point cloud
#   {s}/luminance_sfm/images/    ← undistorted luminance images (ignore for training)

# 3. Copy the sparse result into the original scene so 3DGS can use it
#    while still training on original RGB images:
cp -r {s}/luminance_sfm/sparse {s}/sparse_luminance

# 4. Train 3DGS (Full model / ablation)
#    Use a wrapper directory that has:
#      sparse/ → {s}/sparse_luminance/0  (luminance poses)
#      images/ → {s}/images              (original RGB)
""")
    print("=" * 70)
    print("See ablation study commands at the end of this script's docstring.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare luminance (HSV V-channel) images for SfM input.")
    parser.add_argument("-s", "--source_path", required=True,
                        help="Root directory of the scene.")
    parser.add_argument("--images", default="images",
                        help="Source RGB images subfolder (default: 'images').")
    parser.add_argument("--output", default="luminance_images",
                        help="Output subfolder name (default: 'luminance_images').")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output images.")
    parser.add_argument("--print_commands", action="store_true",
                        help="Print example COLMAP commands after conversion.")
    args = parser.parse_args()

    prepare_luminance_images(
        source_path=args.source_path,
        images_folder=args.images,
        output_folder=args.output,
        overwrite=args.overwrite,
    )

    if args.print_commands:
        print_colmap_commands(args.source_path, args.output)
