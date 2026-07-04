"""
visualization/create_comparison.py
=================================

Creates comparison figures for ALL predicted MRI volumes.

Layout:

                Input      Prediction      Ground Truth

Axial
Coronal
Sagittal

Usage
-----
python -m visualization.create_comparison
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        default="data/pmc_paired/PMC dataset/3D/created-3D/1o5T_final/T1_1o5T",
    )

    parser.add_argument(
        "--prediction_dir",
        default="experiments/lower_bound/outputs",
    )

    parser.add_argument(
        "--target_dir",
        default="data/pmc_paired/PMC dataset/3D/created-3D/3T_final/T1_3T",
    )

    parser.add_argument(
        "--output_dir",
        default="experiments/lower_bound/comparisons",
    )

    return parser.parse_args()


def normalize(img):

    img = img.astype(np.float32)

    img -= img.min()

    if img.max() > 0:
        img /= img.max()

    return img


def load_volume(path):

    volume = nib.load(str(path)).get_fdata()

    return normalize(volume)


def get_middle_slices(volume):

    d, h, w = volume.shape

    axial = volume[d // 2]

    coronal = volume[:, h // 2, :]

    sagittal = volume[:, :, w // 2]

    return [axial, coronal, sagittal]


def create_figure(input_vol, pred_vol, target_vol, save_path):

    input_slices = get_middle_slices(input_vol)
    pred_slices = get_middle_slices(pred_vol)
    target_slices = get_middle_slices(target_vol)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    column_titles = [
        "Input (1.5T)",
        "Prediction",
        "Ground Truth (3T)",
    ]

    row_titles = [
        "Axial",
        "Coronal",
        "Sagittal",
    ]

    for j in range(3):
        axes[0, j].set_title(
            column_titles[j],
            fontsize=12,
            fontweight="bold",
        )

    images = [
        input_slices,
        pred_slices,
        target_slices,
    ]

    for r in range(3):

        for c in range(3):

            axes[r, c].imshow(
                images[c][r],
                cmap="gray",
                origin="lower",
            )

            axes[r, c].axis("off")

        axes[r, 0].set_ylabel(
            row_titles[r],
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main():

    args = parse_args()

    input_dir = Path(args.input_dir)
    pred_dir = Path(args.prediction_dir)
    target_dir = Path(args.target_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("*_pred.nii.gz"))

    print(f"Found {len(pred_files)} predictions.\n")

    count = 0

    for pred_path in pred_files:

        patient_id = pred_path.name.replace("_pred.nii.gz", "")

        input_path = input_dir / f"{patient_id}.nii.gz"
        target_path = target_dir / f"{patient_id}.nii.gzWarped.nii.gz"

        if not input_path.exists():
            print(f"Missing input for {patient_id}")
            continue

        if not target_path.exists():
            print(f"Missing target for {patient_id}")
            continue

        input_vol = load_volume(input_path)
        pred_vol = load_volume(pred_path)
        target_vol = load_volume(target_path)

        save_path = output_dir / f"{patient_id}_comparison.png"

        create_figure(
            input_vol,
            pred_vol,
            target_vol,
            save_path,
        )

        count += 1
        print(f"[{count}/{len(pred_files)}] Saved {save_path.name}")

    print("\n" + "=" * 60)
    print(f"Finished. Generated {count} comparison figures.")
    print(f"Saved in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()