"""
evaluation/evaluate.py
======================

Runs evaluation on reconstructed MRI volumes.

Usage
-----
python -m evaluation.evaluate
"""

import os
import argparse

from evaluation.metrics import main as evaluate_main


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lf_dir",
        default="data/pmc_paired/PMC dataset/3D/created-3D/1o5T_final/T1_1o5T",
        type=str,
        help="Directory containing PMC low-field scans",
    )

    parser.add_argument(
        "--hf_dir",
        default="data/pmc_paired/PMC dataset/3D/created-3D/3T_final/T1_3T",
        type=str,
        help="Directory containing PMC ground-truth scans",
    )

    parser.add_argument(
        "--split_file",
        default="splits/pmc_test.txt",
        type=str,
    )

    parser.add_argument(
        "--pred_dir",
        default="experiments/lower_bound/outputs",
        type=str,
    )

    parser.add_argument(
        "--output_csv",
        default="experiments/lower_bound/metrics/results.csv",
        type=str,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
    )

    parser.add_argument(
        "--skip_lpips",
        action="store_true",
    )

    parser.add_argument(
        "--skip_fid",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    import sys

    sys.argv = [
        "metrics.py",
        "--lf_dir", args.lf_dir,
        "--hf_dir", args.hf_dir,
        "--split_file", args.split_file,
        "--pred_dir", args.pred_dir,
        "--output_csv", args.output_csv,
        "--device", args.device,
    ]

    if args.skip_lpips:
        sys.argv.append("--skip_lpips")

    if args.skip_fid:
        sys.argv.append("--skip_fid")

    evaluate_main()