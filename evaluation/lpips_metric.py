"""
evaluation/lpips_metric.py
============================
LPIPS (Learned Perceptual Image Patch Similarity) for 3D MRI volumes.

LPIPS is inherently a 2D perceptual metric (built on pretrained CNNs like
AlexNet/VGG), so for 3D volumes we compute it slice-by-slice along the
depth axis and average.
"""

import numpy as np
import torch

try:
    import lpips
except ImportError:
    raise ImportError(
        "lpips package not found. Install with: pip install lpips"
    )


class LPIPSMetric:
    """
    Wraps the LPIPS network for 3D volume evaluation.

    Usage
    -----
    lpips_metric = LPIPSMetric(device="cuda:0")
    score = lpips_metric.compute(pred_volume, target_volume)
    """

    def __init__(self, net="alex", device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()

    def _prepare_slice(self, slice_2d):
        """
        Convert a single 2D grayscale slice [H,W] in [0,1] to LPIPS input
        format: [1,3,H,W] in [-1,1].
        """
        slice_2d = np.clip(slice_2d, 0.0, 1.0)
        tensor = torch.from_numpy(slice_2d).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)      # [1,1,H,W]
        tensor = tensor.repeat(1, 3, 1, 1)             # [1,3,H,W] (grayscale -> 3ch)
        tensor = tensor * 2.0 - 1.0                    # [0,1] -> [-1,1]
        return tensor.to(self.device)

    @torch.no_grad()
    def compute(self, pred, target):
        """
        Compute mean LPIPS over all depth slices of a 3D volume.

        Parameters
        ----------
        pred, target : np.ndarray or torch.Tensor, shape [1,D,H,W] or [D,H,W]
                        values normalized to [0, 1]

        Returns
        -------
        float : mean LPIPS score across slices (lower = more similar)
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        pred = np.squeeze(pred)
        target = np.squeeze(target)

        depth = pred.shape[0]
        scores = []

        for d in range(depth):
            pred_slice = self._prepare_slice(pred[d])
            target_slice = self._prepare_slice(target[d])

            score = self.model(pred_slice, target_slice)
            scores.append(score.item())

        return float(np.mean(scores))