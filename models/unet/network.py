"""

Modified from:
https://github.com/milesial/Pytorch-UNet/tree/master

"""

import torch.nn as nn
import torch
import lightning.pytorch as pl

from models.unet.blocks import DoubleConv, Down, Up, OutConv
from schedulers.schedulers import CosineAnnealingWarmupRestarts

class UNetOutputs():
    def __init__(self, pred, loss):
        self.pred = pred
        self.loss = loss

class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_dim: int = 32,
        mask_values: list[float] = [1024/(3071+1024), 0.],
        bilinear=False
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.base_dim = base_dim
        self.mask_values = mask_values
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
        mask = (target != self.mask_values[0]) *\
            (target != self.mask_values[1])
        pred = pred[mask]
        target = target[mask]
        return self.critereon(pred, target)


class UNetLM(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = UNet(**config["model"])
        self.log_image_every_n_steps = config["log_image_every_n_steps"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["scheduler"]["min_lr"]
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer, **self.config["scheduler"]
        )
        return [optimizer], [{"scheduler":lr_scheduler, "interval": "step"}]

    def log_images(self, input_img, pred):
        self.logger.experiment.add_images(
            "input", input_img.detach().cpu().type(torch.float32), self.global_step
        )
        self.logger.experiment.add_images(
            "pred", pred, self.global_step
        )

    def training_step(self, batch, batch_idx):
        input_img, target = batch
        outputs = self.model(input_img, target)
        loss = outputs.loss
        self.log('loss', loss.item())
        if batch_idx % self.log_image_every_n_steps == 0:
            self.log_images(input_img, outputs.pred)
        return loss
