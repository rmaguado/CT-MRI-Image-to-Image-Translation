"""

Modified from:
https://github.com/milesial/Pytorch-UNet/tree/master

"""

import torch.nn as nn
import torch

from models.unet.unet_parts import DoubleConv, Down, Up, OutConv


class UNetOutputs():
    def __init__(self, pred, loss):
        self.pred = pred
        self.loss = loss

class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_dim: int = 32,
        mask_value: int = 1024/(3071+1024),
        bilinear=False
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.base_dim = base_dim
        self.mask_value = mask_value
        self.bilinear = bilinear
        self.critereon = nn.MSELoss()

        self.inc = (DoubleConv(in_channels, base_dim))
        self.down1 = (Down(base_dim, base_dim * 2))
        self.down2 = (Down(base_dim * 2, base_dim * 4))
        self.down3 = (Down(base_dim * 4, base_dim * 8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(base_dim * 8, base_dim * 16 // factor))
        self.up1 = (Up(base_dim * 16, base_dim * 8 // factor, bilinear))
        self.up2 = (Up(base_dim * 8, base_dim * 4 // factor, bilinear))
        self.up3 = (Up(base_dim * 4, base_dim * 2 // factor, bilinear))
        self.up4 = (Up(base_dim * 2, base_dim, bilinear))
        self.outc = (OutConv(base_dim, 1))

    def forward(self, x, target):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        pred = self.outc(x)
        loss = self.forward_loss(pred, target)
        return UNetOutputs(pred, loss)
    
    def forward_loss(self, pred, target):
        """
        MSE loss for target pixels within mask.
        """
        mask = target != self.mask_value
        pred = pred[mask]
        target = target[mask]
        return self.critereon(pred, target)

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
