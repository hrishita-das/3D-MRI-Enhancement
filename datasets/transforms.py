import numpy as np
import torch
import torch.nn.functional as F
import random


class MRITransform:
    """
    Common transform pipeline for both IXI and PMC.
    """

    def __init__(
        self,
        output_size=(160, 192, 160),
        training=True,
    ):

        self.output_size = output_size
        self.training = training

    def normalize(self, img):

        img = img.astype(np.float32)

        img -= img.min()

        if img.max() > 0:
            img /= img.max()

        return img

    def resize(self, img):

        img = torch.from_numpy(img).float()

        img = img.unsqueeze(0).unsqueeze(0)

        img = F.interpolate(
            img,
            size=self.output_size,
            mode="trilinear",
            align_corners=False,
        )

        return img.squeeze(0)

    def random_flip(self, lf, hf):

        if random.random() < 0.5:

            lf = torch.flip(lf, dims=[1])
            hf = torch.flip(hf, dims=[1])

        if random.random() < 0.5:

            lf = torch.flip(lf, dims=[2])
            hf = torch.flip(hf, dims=[2])

        return lf, hf

    def __call__(self, lf, hf):

        lf = self.normalize(lf)
        hf = self.normalize(hf)

        lf = self.resize(lf)
        hf = self.resize(hf)

        if self.training:

            lf, hf = self.random_flip(lf, hf)

        return lf, hf