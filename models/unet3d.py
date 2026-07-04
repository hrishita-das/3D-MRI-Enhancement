import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder


class UNet3D(nn.Module):
    """
    3D U-Net Baseline
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
        )

        self.decoder = Decoder(
            out_channels=out_channels,
            base_channels=base_channels,
        )

    def forward(self, x):

        x1, x2, x3, x4, x5 = self.encoder(x)

        out = self.decoder(
            x1,
            x2,
            x3,
            x4,
            x5,
        )

        return out