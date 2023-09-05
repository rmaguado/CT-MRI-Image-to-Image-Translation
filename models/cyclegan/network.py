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
        return ((y_hat - y)**2).mean() #F.binary_cross_entropy_with_logits(y_hat, y)

    def recon_criterion(self, y_hat, y):
        return F.l1_loss(y_hat, y)

    def adv_loss(self, real_x, disc_y, gen_y):
        fake_y = gen_y(real_x)
        disc_fake_y_hat = disc_y(fake_y)
        adv_loss_y = self.adv_criterion(disc_fake_y_hat, torch.ones_like(disc_fake_y_hat))
        return adv_loss_y, fake_y

    def id_loss(self, real_x, gen_x):
        id_x = gen_x(real_x)
        id_loss_x = self.recon_criterion(id_x, real_x)
        return id_loss_x

    def cycle_loss(self, real_x, fake_y, gen_x):
        cycle_x = gen_x(fake_y)
        cycle_loss_x = self.recon_criterion(cycle_x, real_x)
        return cycle_loss_x

    def gen_loss(self, real_x, real_y, gen_y, gen_x, disc_y):
        adv_loss_y, fake_y = self.adv_loss(real_x, disc_y, gen_y)

        id_loss_y = self.id_loss(real_y, gen_y)

        cycle_loss_x = self.cycle_loss(real_x, fake_y, gen_x)
        cycle_loss_y = self.cycle_loss(real_y, gen_x(real_y), gen_y)
        cycle_loss = cycle_loss_x + cycle_loss_y

        gen_loss_y = adv_loss_y + 0.5*self.lambda_w*id_loss_y + self.lambda_w*cycle_loss
        return gen_loss_y

    def disc_loss(self, real_x, fake_x, disc_x):
        disc_fake_hat = disc_x(fake_x.detach())
        disc_fake_loss = self.adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))

        disc_real_hat = disc_x(real_x)
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

        self.disc_a.disable_grad()
        self.disc_b.disable_grad()
        gen_loss_a = self.gen_loss(real_b, real_a, self.gen_a, self.gen_b, self.disc_a)
        self.manual_backward(gen_loss_a)
        gen_loss_b = self.gen_loss(real_a, real_b, self.gen_b, self.gen_a, self.disc_b)
        self.manual_backward(gen_loss_b)

        self.disc_a.enable_grad()
        self.disc_b.enable_grad()
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

        if (batch_idx // self.accumulate_grad_batches) %\
                self.accumulate_grad_batches == 0:
            self.logger.experiment.add_image(
                "A/real",
                real_a[0][0].detach().cpu().type(torch.float32),
                batch_idx // self.accumulate_grad_batches,
                dataformats="WH"
            )
            self.logger.experiment.add_image(
                "B/fake",
                self.gen_b(real_a)[0][0].detach().cpu().type(torch.float32),
                batch_idx // self.accumulate_grad_batches,
                dataformats="WH"
            )
            self.logger.experiment.add_image(
                "B/real",
                real_b[0][0].detach().cpu().type(torch.float32),
                batch_idx // self.accumulate_grad_batches,
                dataformats="WH"
            )
            self.logger.experiment.add_image(
                "A/fake",
                self.gen_a(real_b)[0][0].detach().cpu().type(torch.float32),
                batch_idx // self.accumulate_grad_batches,
                dataformats="WH"
            )
