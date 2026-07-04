"""
evaluation/metrics.py
=======================

Evaluate reconstructed MRI volumes using
PSNR, SSIM, LPIPS and FID.
"""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from datasets.pmc_dataset import PMCPairedDataset
from datasets.transforms import MRITransform

from evaluation.psnr_ssim import compute_psnr, compute_ssim
from evaluation.lpips_metric import LPIPSMetric
from evaluation.fid import FIDMetric


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lf_dir", required=True)
    parser.add_argument("--hf_dir", required=True)
    parser.add_argument("--split_file", default="splits/pmc_test.txt")
    parser.add_argument("--pred_dir", default="experiments/lower_bound/outputs")
    parser.add_argument("--output_csv", default="experiments/lower_bound/metrics/results.csv")
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--skip_fid", action="store_true")

    return parser.parse_args()


def load_prediction(pred_dir, patient_id):

    pred_path = Path(pred_dir) / f"{patient_id}_pred.nii.gz"

    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    volume = nib.load(str(pred_path)).get_fdata().astype(np.float32)

    volume -= volume.min()

    if volume.max() > 0:
        volume /= volume.max()

    return volume


def main():

    args = parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    transform = MRITransform(training=False)

    dataset = PMCPairedDataset(
        lf_dir=args.lf_dir,
        hf_dir=args.hf_dir,
        split_file=args.split_file,
        transform=transform,
    )

    lpips_metric = None
    fid_metric = None

    if not args.skip_lpips:
        lpips_metric = LPIPSMetric(device=device)

    if not args.skip_fid:
        fid_metric = FIDMetric(device=device)

    results = []

    for sample in tqdm(dataset, desc="Evaluating"):

        patient_id = sample["patient_id"]

        target = sample["target"]

        pred = load_prediction(
            args.pred_dir,
            patient_id,
        )

        psnr = compute_psnr(pred, target)
        ssim = compute_ssim(pred, target)

        row = {
            "patient_id": patient_id,
            "psnr": psnr,
            "ssim": ssim,
        }

        if lpips_metric is not None:
            row["lpips"] = lpips_metric.compute(pred, target)

        if fid_metric is not None:
            fid_metric.add_batch(pred, real=False)
            fid_metric.add_batch(target, real=True)

        results.append(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    print(f"Samples    : {len(results)}")
    print(f"Mean PSNR  : {np.mean([r['psnr'] for r in results]):.4f}")
    print(f"Mean SSIM  : {np.mean([r['ssim'] for r in results]):.4f}")

    if lpips_metric is not None:
        print(f"Mean LPIPS : {np.mean([r['lpips'] for r in results]):.4f}")

    if fid_metric is not None:
        print(f"FID        : {fid_metric.compute():.4f}")

    print("=" * 60)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    main()