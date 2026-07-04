import torch.nn as nn


class ReconstructionLoss(nn.Module):
    """
    Combined L1 + MSE Reconstruction Loss
    """

    def __init__(self, l1_weight=1.0, mse_weight=1.0):
        super().__init__()

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.l1_weight = l1_weight
        self.mse_weight = mse_weight

    def forward(self, prediction, target):

        l1_loss = self.l1(prediction, target)
        mse_loss = self.mse(prediction, target)

        total_loss = (
            self.l1_weight * l1_loss
            + self.mse_weight * mse_loss
        )

        return {
            "loss": total_loss,
            "l1": l1_loss,
            "mse": mse_loss
        }