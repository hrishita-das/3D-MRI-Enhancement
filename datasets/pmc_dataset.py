import os
import glob
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class PMCPairedDataset(Dataset):
    """
    PMC Paired Dataset

    Input  : 1.5T MRI
    Target : Registered 3T MRI

    Supports validation/test split through a text file.

    Returns
    -------
    {
        "input": Tensor [1,D,H,W],
        "target": Tensor [1,D,H,W],
        "patient_id": str
    }
    """

    def __init__(
        self,
        lf_dir,
        hf_dir,
        split_file=None,
        transform=None,
    ):

        self.transform = transform

        # -------------------------------------------------
        # Read split file (optional)
        # -------------------------------------------------

        allowed_ids = None

        if split_file is not None:

            with open(split_file, "r") as f:

                allowed_ids = set(
                    line.strip()
                    for line in f
                    if line.strip()
                )

        # -------------------------------------------------
        # Find all LF images
        # -------------------------------------------------

        lf_files = sorted(
            glob.glob(
                os.path.join(lf_dir, "*.nii.gz")
            )
        )

        self.samples = []

        for lf_path in lf_files:

            patient_id = os.path.basename(
                lf_path
            ).replace(".nii.gz", "")

            # If using split, ignore patients not listed
            if allowed_ids is not None:

                if patient_id not in allowed_ids:
                    continue

            hf_filename = (
                patient_id + ".nii.gzWarped.nii.gz"
            )

            hf_path = os.path.join(
                hf_dir,
                hf_filename,
            )

            if os.path.exists(hf_path):

                self.samples.append({

                    "patient_id": patient_id,

                    "input_path": lf_path,

                    "target_path": hf_path

                })

        print("=" * 60)
        print("PMC Paired Dataset Loaded")

        if split_file is not None:
            print("Split File :", split_file)

        print(f"Total Samples : {len(self.samples)}")
        print("=" * 60)

    def __len__(self):

        return len(self.samples)

    def load_nifti(self, path):

        volume = nib.load(path).get_fdata()

        volume = volume.astype(np.float32)

        return volume

    def normalize(self, volume):

        volume -= volume.min()

        max_val = volume.max()

        if max_val > 0:

            volume /= max_val

        return volume

    def __getitem__(self, idx):

        sample = self.samples[idx]

        input_volume = self.load_nifti(
            sample["input_path"]
        )

        target_volume = self.load_nifti(
            sample["target_path"]
        )

        input_volume = self.normalize(input_volume)

        target_volume = self.normalize(target_volume)

        if self.transform is not None:

            input_tensor, target_tensor = self.transform(
                input_volume,
                target_volume,
            )

        else:

            input_tensor = torch.from_numpy(
                input_volume
            ).float().unsqueeze(0)

            target_tensor = torch.from_numpy(
                target_volume
            ).float().unsqueeze(0)

        return {

            "input": input_tensor,

            "target": target_tensor,

            "patient_id": sample["patient_id"]

        }