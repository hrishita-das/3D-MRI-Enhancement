import torch.nn as nn

from models.blocks import Up, OutConv


class Decoder(nn.Module):
    """
    3D U-Net Decoder
    """

    def __init__(self,
                 out_channels=1,
                 base_channels=32):

        super().__init__()

        c = base_channels

        self.up1 = Up(c * 16, c * 8)

        self.up2 = Up(c * 8, c * 4)

        self.up3 = Up(c * 4, c * 2)

        self.up4 = Up(c * 2, c)

        self.outc = OutConv(c, out_channels)

    def forward(self, x1, x2, x3, x4, x5):

        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        x = self.outc(x)

        return x