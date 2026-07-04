"""
infer.py
========

Run inference using the trained lower-bound model.

Input:
    PMC 1.5T images

Output:
    Synthesized 3T images (.nii.gz)
"""

import os
from pathlib import Path

import yaml
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet3d import UNet3D
from datasets.pmc_dataset import PMCPairedDataset
from datasets.transforms import MRITransform


# -------------------------------------------------------
# Load configs
# -------------------------------------------------------

train_cfg = yaml.safe_load(open("configs/train.yaml"))

dataset_cfg = yaml.safe_load(open("configs/dataset.yaml"))

model_cfg = yaml.safe_load(open("configs/model.yaml"))

device = torch.device(train_cfg["device"])


# -------------------------------------------------------
# Model
# -------------------------------------------------------

model = UNet3D(

    in_channels=model_cfg["input_channels"],

    out_channels=model_cfg["output_channels"],

    base_channels=model_cfg["base_channels"],

).to(device)


# -------------------------------------------------------
# Load checkpoint
# -------------------------------------------------------

checkpoint_path = "experiments/lower_bound/checkpoints/best.pt"

checkpoint = torch.load(

    checkpoint_path,

    map_location=device,

)

model.load_state_dict(checkpoint["model"])

model.eval()

print(f"\nLoaded checkpoint : {checkpoint_path}")
print(f"Best model from epoch {checkpoint['epoch']}")


# -------------------------------------------------------
# Dataset
# -------------------------------------------------------

transform = MRITransform(training=False)

dataset = PMCPairedDataset(

    lf_dir=dataset_cfg["pmc"]["lf"],

    hf_dir=dataset_cfg["pmc"]["hf"],

    split_file="splits/pmc_test.txt",

    transform=transform,

)

loader = DataLoader(

    dataset,

    batch_size=1,

    shuffle=False,

)


# -------------------------------------------------------
# Output folder
# -------------------------------------------------------

output_dir = Path("experiments/lower_bound/outputs")

output_dir.mkdir(

    parents=True,

    exist_ok=True,

)


# -------------------------------------------------------
# Inference
# -------------------------------------------------------

print("\nRunning inference...\n")

with torch.no_grad():

    for batch in tqdm(loader):

        inputs = batch["input"].to(device)

        prediction = model(inputs)

        prediction = prediction.squeeze().cpu().numpy()

        prediction = np.clip(prediction, 0, 1)

        patient_id = batch["patient_id"][0]

        affine = np.eye(4)

        output_path = output_dir / f"{patient_id}_pred.nii.gz"

        nib.save(

            nib.Nifti1Image(

                prediction.astype(np.float32),

                affine,

            ),

            output_path,

        )


print("\n==========================================")

print(f"Saved {len(dataset)} reconstructed volumes")

print(f"Location : {output_dir}")

print("==========================================")