import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Conv3D -> BN -> ReLU -> Conv3D -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.InstanceNorm3d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.InstanceNorm3d(out_channels),

            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """
    MaxPool + DoubleConv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.MaxPool3d(2),

            DoubleConv(in_channels, out_channels)

        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """
    Upsampling + Skip Connection + DoubleConv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            in_channels,
            out_channels,
        )

    def forward(self, x1, x2):

        x1 = self.up(x1)

        diffD = x2.size(2) - x1.size(2)
        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)

        x1 = nn.functional.pad(
            x1,
            [
                diffW // 2,
                diffW - diffW // 2,
                diffH // 2,
                diffH - diffH // 2,
                diffD // 2,
                diffD - diffD // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1x1 convolution
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        return self.conv(x)