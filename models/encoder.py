import torch.nn as nn

from models.blocks import DoubleConv, Down


class Encoder(nn.Module):
    """
    3D U-Net Encoder

    Input
        ↓
    DoubleConv
        ↓
    Down
        ↓
    Down
        ↓
    Down
        ↓
    Down
    """

    def __init__(self,
                 in_channels=1,
                 base_channels=32):

        super().__init__()

        c = base_channels

        self.inc = DoubleConv(in_channels, c)

        self.down1 = Down(c, c * 2)

        self.down2 = Down(c * 2, c * 4)

        self.down3 = Down(c * 4, c * 8)

        self.down4 = Down(c * 8, c * 16)

    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        return x1, x2, x3, x4, x5