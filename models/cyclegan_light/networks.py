from torch import nn
import torch


class Downsampling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm=True,
        kernel_size=4,
        stride=2,
        padding=1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False)
        )
        if norm:
            self.block.append(nn.InstanceNorm2d(out_channels, affine=True))
        self.block.append(nn.LeakyReLU(0.3))

    def forward(self, x):
        return self.block(x)

class Upsampling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=False,
        kernel_size=4,
        stride=2,
        padding=1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
        )
        if dropout:
            self.block.append(nn.Dropout(0.5))
        self.block.append(nn.ReLU())

    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels):
        super().__init__()
        self.downsampling_blocks = nn.Sequential(
            Downsampling(in_channels, hid_channels, norm=False), #64x128x128
            Downsampling(hid_channels, hid_channels*2), #128x64x64
            Downsampling(hid_channels*2, hid_channels*4), #256x32x32
            Downsampling(hid_channels*4, hid_channels*8), #512x16x16
            Downsampling(hid_channels*8, hid_channels*8), #512x8x8
            Downsampling(hid_channels*8, hid_channels*8), #512x4x4
            Downsampling(hid_channels*8, hid_channels*8), #512x2x2
            Downsampling(hid_channels*8, hid_channels*8, norm=False), #512x1x1
        )
        self.upsampling_blocks = nn.Sequential(
            Upsampling(hid_channels*8, hid_channels*8, dropout=True), #(512+512)x2x2
            Upsampling(hid_channels*16, hid_channels*8, dropout=True), #(512+512)x4x4
            Upsampling(hid_channels*16, hid_channels*8, dropout=True), #(512+512)x8x8
            Upsampling(hid_channels*16, hid_channels*8), #(512+512)x16x16
            Upsampling(hid_channels*16, hid_channels*4), #(256+256)x32x32
            Upsampling(hid_channels*8, hid_channels*2), #(128+128)x64x64
            Upsampling(hid_channels*4, hid_channels), #(64+64)x128x128
        )
        self.feature_block = nn.Sequential(
            nn.ConvTranspose2d(hid_channels*2, out_channels,
                               kernel_size=4, stride=2, padding=1), #3x256x256
            nn.Tanh(),
        )

    def forward(self, x):
        skips = []
        for down in self.downsampling_blocks:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        for up, skip in zip(self.upsampling_blocks, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)

        return self.feature_block(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, hid_channels):
        super().__init__()
        self.block = nn.Sequential(
            Downsampling(in_channels, hid_channels, norm=False), #64x128x128
            Downsampling(hid_channels, hid_channels*2), #128x64x64
            Downsampling(hid_channels*2, hid_channels*4), #256x32x32
            Downsampling(hid_channels*4, hid_channels*8, stride=1), #512x31x31
            nn.Conv2d(hid_channels*8, 1, kernel_size=4, padding=1), #1x30x30
        )

    def forward(self, x):
        return self.block(x)
