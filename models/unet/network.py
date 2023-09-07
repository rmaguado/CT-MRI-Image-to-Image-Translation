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
        mask_values: tuple[float] = (1024/(3071+1024), 0.),
        bilinear=False
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.base_dim = base_dim
        self.mask_values = mask_values
        self.bilinear = bilinear
        self.critereon = nn.MSELoss()

        self.inc = DoubleConv(in_channels, base_dim)
        self.down1 = Down(base_dim, base_dim * 2)
        self.down2 = Down(base_dim * 2, base_dim * 4)
        self.down3 = Down(base_dim * 4, base_dim * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_dim * 8, base_dim * 16 // factor)
        self.up1 = Up(base_dim * 16, base_dim * 8 // factor, bilinear)
        self.up2 = Up(base_dim * 8, base_dim * 4 // factor, bilinear)
        self.up3 = Up(base_dim * 4, base_dim * 2 // factor, bilinear)
        self.up4 = Up(base_dim * 2, base_dim, bilinear)
        self.outc = OutConv(base_dim, 1)

    def forward(self, input_image, target):
        layer_1 = self.inc(input_image)
        layer_2 = self.down1(layer_1)
        layer_3 = self.down2(layer_2)
        layer_4 = self.down3(layer_3)
        layer_5 = self.down4(layer_4)
        input_image = self.up1(layer_5, layer_4)
        input_image = self.up2(input_image, layer_3)
        input_image = self.up3(input_image, layer_2)
        input_image = self.up4(input_image, layer_1)
        pred = self.outc(input_image)
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

    def log_images(self, input_img, pred, batch_idx):
        self.logger.experiment.add_image(
            "input",
            input_img[0][0].detach().cpu().type(torch.float32),
            batch_idx,
            dataformats="WH"
        )
        self.logger.experiment.add_image(
            "pred",
            pred[0][0].detach().cpu().type(torch.float32),
            batch_idx,
            dataformats="WH"
        )

    def training_step(self, batch, batch_idx):
        input_img, target = batch
        outputs = self.model(input_img, target)
        loss = outputs.loss
        self.log('loss', loss.item())
        if batch_idx % self.log_image_every_n_steps == 0:
            self.log_images(input_img, outputs.pred, batch_idx)
        return loss
