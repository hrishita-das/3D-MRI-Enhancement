import os
import glob
import random
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class IXIUnpairedDataset(Dataset):
    """
    Unpaired IXI Dataset

    Returns
    -------
    {
        'lf' : 1.5T image
        'hf' : randomly sampled 3T image
        'lf_name'
        'hf_name'
    }
    """

    def __init__(self,
                 lf_dir,
                 hf_dir,
                 transform=None):

        self.transform = transform

        self.lf_files = sorted(
            glob.glob(os.path.join(lf_dir, "*.nii.gz"))
        )

        self.hf_files = sorted(
            glob.glob(os.path.join(hf_dir, "*.nii.gz"))
        )

        print(f"IXI Dataset Loaded")
        print(f"1.5T Images : {len(self.lf_files)}")
        print(f"3T Images   : {len(self.hf_files)}")

    def __len__(self):

        return len(self.lf_files)

    def load_nifti(self, path):

        volume = nib.load(path).get_fdata()

        volume = volume.astype(np.float32)

        return volume

    def normalize(self, volume):

        volume -= volume.min()

        if volume.max() > 0:
            volume /= volume.max()

        return volume

    def __getitem__(self, idx):

        lf_path = self.lf_files[idx]

        hf_path = random.choice(self.hf_files)

        lf = self.load_nifti(lf_path)
        hf = self.load_nifti(hf_path)

        lf = self.normalize(lf)
        hf = self.normalize(hf)

        if self.transform is not None:

            lf, hf = self.transform(lf, hf)

        else:

            lf = torch.from_numpy(lf).float().unsqueeze(0)
            hf = torch.from_numpy(hf).float().unsqueeze(0)

        return {
    "input": lf,
    "target": hf,
    "patient_id": os.path.basename(lf_path),
    "reference_id": os.path.basename(hf_path)
}