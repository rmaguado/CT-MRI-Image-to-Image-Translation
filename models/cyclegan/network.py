import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F

from models.cyclegan.blocks import (
    GeneratorUNet, DiscriminatorUNet, GeneratorViT, DiscriminatorViT
)


class Cyclegan(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.learning_rate = config["lr"]
        self.betas = config["betas"]
        self.lambda_w = config["lambda_w"]
        self.accumulate_grad_batches = config["accumulate_grad_batches"]
        self.log_image_every_n_steps = config["log_image_every_n_steps"]

        if config["block_type"] == "CNN":
            self.gen_a = GeneratorUNet(**config["generator"])
            self.gen_b = GeneratorUNet(**config["generator"])
            self.disc_a = DiscriminatorUNet(**config["discriminator"])
            self.disc_b = DiscriminatorUNet(**config["discriminator"])
        elif config["block_type"] == "ViT":
            self.gen_a = GeneratorViT(**config["generator"])
            self.gen_b = GeneratorViT(**config["generator"])
            self.disc_a = DiscriminatorViT(**config["discriminator"])
            self.disc_b = DiscriminatorViT(**config["discriminator"])
        else:
            raise ValueError("Unrecognized value for block_type")

        self.automatic_optimization = False

    def forward(self, z):
        return self.gen_a(z)

    def adv_criterion(self, y_hat, y):
        return ((y_hat - y)**2).mean()#F.binary_cross_entropy_with_logits(y_hat, y)

    def recon_criterion(self, y_hat, y):
        return F.l1_loss(y_hat, y)

    def adv_loss(self, real_X, disc_Y, gen_XY):
        fake_Y = gen_XY(real_X)
        disc_fake_Y_hat = disc_Y(fake_Y)
        adv_loss_XY = self.adv_criterion(disc_fake_Y_hat, torch.ones_like(disc_fake_Y_hat))
        return adv_loss_XY, fake_Y

    def id_loss(self, real_X, gen_YX):
        id_X = gen_YX(real_X)
        id_loss_X = self.recon_criterion(id_X, real_X)
        return id_loss_X

    def cycle_loss(self, real_X, fake_Y, gen_YX):
        cycle_X = gen_YX(fake_Y)
        cycle_loss_X = self.recon_criterion(cycle_X, real_X)
        return cycle_loss_X
    
    def gen_loss(self, real_X, real_Y, gen_XY, gen_YX, disc_Y):
        adv_loss_XY, fake_Y = self.adv_loss(real_X, disc_Y, gen_XY)

        id_loss_Y = self.id_loss(real_Y, gen_XY)

        cycle_loss_X = self.cycle_loss(real_X, fake_Y, gen_YX)
        cycle_loss_Y = self.cycle_loss(real_Y, gen_YX(real_Y), gen_XY)
        cycle_loss = cycle_loss_X + cycle_loss_Y

        gen_loss_XY = adv_loss_XY + 0.5*self.lambda_w*id_loss_Y + self.lambda_w*cycle_loss
        return gen_loss_XY

    def disc_loss(self, real_X, fake_X, disc_X):
        disc_fake_hat = disc_X(fake_X.detach())
        disc_fake_loss = self.adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))

        disc_real_hat = disc_X(real_X)
        disc_real_loss = self.adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))

        disc_loss = (disc_fake_loss+disc_real_loss) / 2
        return disc_loss

    def configure_optimizers(self):
        params = {
            "lr": self.learning_rate,
            "betas": self.betas,
        }
        opt_gen_a = torch.optim.Adam(self.gen_a.parameters(), **params)
        opt_gen_b = torch.optim.Adam(self.gen_b.parameters(), **params)

        opt_disc_a = torch.optim.Adam(self.disc_a.parameters(), **params)
        opt_disc_b = torch.optim.Adam(self.disc_b.parameters(), **params)

        return opt_gen_a, opt_gen_b, opt_disc_a, opt_disc_b

    def training_step(self, batch, batch_idx):
        real_a, real_b = batch
        optims = self.optimizers()

        gen_loss_a = self.gen_loss(real_b, real_a, self.gen_a, self.gen_b, self.disc_a)
        self.manual_backward(gen_loss_a)
        gen_loss_b = self.gen_loss(real_a, real_b, self.gen_b, self.gen_a, self.disc_b)
        self.manual_backward(gen_loss_b)

        disc_loss_a = self.disc_loss(real_a, self.gen_a(real_b), self.disc_a)
        self.manual_backward(disc_loss_a)
        disc_loss_b = self.disc_loss(real_b, self.gen_b(real_a), self.disc_b)
        self.manual_backward(disc_loss_b)

        if batch_idx % self.accumulate_grad_batches == 0 and batch_idx != 0:
            for opt in optims:
                opt.step()
                opt.zero_grad()

        self.log("loss/generator_A", gen_loss_a)
        self.log("loss/generator_B", gen_loss_b)
        self.log("loss/discriminator_A", disc_loss_a)
        self.log("loss/discriminator_B", disc_loss_b)
        self.log(
            "learning_rate",
            self.optimizers()[0].param_groups[0]["lr"],
        )

        if (batch_idx // self.accumulate_grad_batches) %\
                self.log_image_every_n_steps == 0:
            self.logger.experiment.add_images(
                "A/real",
                real_a[0][0].detach().cpu().type(torch.float32),
                self.current_epoch,
                dataformats="WH"
            )
            self.logger.experiment.add_images(
                "A/fake",
                self.gen_b(real_b)[0][0].detach().cpu().type(torch.float32),
                self.current_epoch,
                dataformats="WH"
            )
            self.logger.experiment.add_images(
                "B/real",
                real_b[0][0].detach().cpu().type(torch.float32),
                self.current_epoch,
                dataformats="WH"
            )
            self.logger.experiment.add_images(
                "B/fake",
                self.gen_a(real_a)[0][0].detach().cpu().type(torch.float32),
                self.current_epoch,
                dataformats="WH"
            )
