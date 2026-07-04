"""
visualization/save_slices.py
============================

Extract middle axial, coronal and sagittal slices from a 3D MRI volume
and save them as PNG images.

Example
-------
python -m visualization.save_slices \
    --input experiments/lower_bound/outputs/10_pred.nii.gz \
    --output_dir experiments/lower_bound/slices
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input NIfTI volume (.nii or .nii.gz)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/slices",
        help="Directory to save PNG images",
    )

    return parser.parse_args()


def normalize(img):
    img = img.astype(np.float32)

    img -= img.min()

    if img.max() > 0:
        img /= img.max()

    return img


def save_slice(img, filename):
    plt.figure(figsize=(6, 6))

    plt.imshow(img, cmap="gray", origin="lower")

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close()


def main():

    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume = nib.load(args.input).get_fdata()

    volume = normalize(volume)

    d, h, w = volume.shape

    axial = volume[d // 2, :, :]
    coronal = volume[:, h // 2, :]
    sagittal = volume[:, :, w // 2]

    base = Path(args.input).stem.replace(".nii", "")

    save_slice(axial, output_dir / f"{base}_axial.png")
    save_slice(coronal, output_dir / f"{base}_coronal.png")
    save_slice(sagittal, output_dir / f"{base}_sagittal.png")

    print("=" * 60)
    print("Slices saved successfully")
    print("=" * 60)
    print(output_dir / f"{base}_axial.png")
    print(output_dir / f"{base}_coronal.png")
    print(output_dir / f"{base}_sagittal.png")


if __name__ == "__main__":
    main()