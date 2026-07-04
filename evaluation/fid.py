"""
evaluation/fid.py
===================
Frechet Inception Distance (FID) for 3D MRI volumes.

FID compares two distributions of Inception-v3 features. Since Inception
is a 2D network, we extract features slice-by-slice from all volumes in
each set (predicted vs. target), pool them, and compute FID between the
two feature distributions.
"""

import numpy as np
import torch
from scipy import linalg

try:
    from pytorch_fid.inception import InceptionV3
except ImportError:
    raise ImportError(
        "pytorch-fid package not found. Install with: pip install pytorch-fid"
    )


class FIDMetric:
    """
    Accumulates Inception features from two volume sets (real / fake)
    and computes the Frechet distance between them.

    Usage
    -----
    fid_metric = FIDMetric(device="cuda:0")
    for pred_vol, target_vol in dataset:
        fid_metric.add_batch(pred_vol, real=False)
        fid_metric.add_batch(target_vol, real=True)
    score = fid_metric.compute()
    """

    def __init__(self, device="cuda:0", dims=2048):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(self.device)
        self.model.eval()

        self.real_features = []
        self.fake_features = []

    def _prepare_slice(self, slice_2d):
        """Convert 2D grayscale slice [H,W] in [0,1] to InceptionV3 input [1,3,H,W]."""
        slice_2d = np.clip(slice_2d, 0.0, 1.0)
        tensor = torch.from_numpy(slice_2d).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        tensor = tensor.repeat(1, 3, 1, 1)         # [1,3,H,W]
        return tensor.to(self.device)

    @torch.no_grad()
    def _extract_features(self, volume):
        """Extract pooled Inception features for every slice in a volume."""
        if isinstance(volume, torch.Tensor):
            volume = volume.detach().cpu().numpy()
        volume = np.squeeze(volume)

        features = []
        for d in range(volume.shape[0]):
            slice_tensor = self._prepare_slice(volume[d])
            feat = self.model(slice_tensor)[0]
            feat = feat.squeeze(-1).squeeze(-1).cpu().numpy()  # [1, dims]
            features.append(feat[0])

        return features

    def add_batch(self, volume, real: bool):
        """Add a single volume's slice features into the real or fake pool."""
        feats = self._extract_features(volume)
        if real:
            self.real_features.extend(feats)
        else:
            self.fake_features.extend(feats)

    @staticmethod
    def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return float(fid)

    def compute(self):
        """Compute FID between accumulated real and fake feature sets."""
        real = np.array(self.real_features)
        fake = np.array(self.fake_features)

        mu1, sigma1 = real.mean(axis=0), np.cov(real, rowvar=False)
        mu2, sigma2 = fake.mean(axis=0), np.cov(fake, rowvar=False)

        return self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    def reset(self):
        self.real_features = []
        self.fake_features = []