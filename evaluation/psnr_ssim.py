"""
evaluation/psnr_ssim.py
========================
PSNR and SSIM computation for 3D MRI volumes.

Volumes are expected as numpy arrays or torch tensors, normalized to [0, 1],
shaped [D, H, W] or [1, D, H, W].
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def _to_numpy(volume):
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    volume = np.squeeze(volume)  # remove channel dim if present
    return volume.astype(np.float32)


def compute_psnr(pred, target, data_range=1.0):
    """
    Compute PSNR between predicted and target 3D volumes.

    Parameters
    ----------
    pred, target : np.ndarray or torch.Tensor, shape [1,D,H,W] or [D,H,W]
    data_range : float, dynamic range of the data (1.0 if normalized to [0,1])

    Returns
    -------
    float
    """
    pred = _to_numpy(pred)
    target = _to_numpy(target)

    return float(sk_psnr(target, pred, data_range=data_range))


def compute_ssim(pred, target, data_range=1.0):
    """
    Compute SSIM between predicted and target 3D volumes.

    Uses a 3D structural similarity window (skimage supports N-D arrays).

    Parameters
    ----------
    pred, target : np.ndarray or torch.Tensor, shape [1,D,H,W] or [D,H,W]
    data_range : float

    Returns
    -------
    float
    """
    pred = _to_numpy(pred)
    target = _to_numpy(target)

    return float(
        sk_ssim(
            target,
            pred,
            data_range=data_range,
            channel_axis=None,  # grayscale volume, no channel axis
        )
    )


def compute_psnr_ssim(pred, target, data_range=1.0):
    """Convenience wrapper returning both metrics as a dict."""
    return {
        "psnr": compute_psnr(pred, target, data_range),
        "ssim": compute_ssim(pred, target, data_range),
    }