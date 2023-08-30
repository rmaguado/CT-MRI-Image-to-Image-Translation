import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F

from models.cyclegan_light.networks import Generator, Discriminator


class CycleGAN(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        hid_channels,
        lr,
        betas,
        lambda_w,
        accumulate_grad_batches,
        log_image_every_n_steps
    ):
        super().__init__()
        self.learning_rate = lr
        self.betas = betas
        self.lambda_w = lambda_w
        self.accumulate_grad_batches = accumulate_grad_batches
        self.log_image_every_n_steps = log_image_every_n_steps

        self.gen_a = Generator(in_channels, out_channels, hid_channels).apply(self.weights_init)
        self.gen_b = Generator(in_channels, out_channels, hid_channels).apply(self.weights_init)
        self.disc_a = Discriminator(in_channels, hid_channels).apply(self.weights_init)
        self.disc_b = Discriminator(in_channels, hid_channels).apply(self.weights_init)

        self.automatic_optimization = False

    def forward(self, z):
        return self.gen_a(z)

    def weights_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight, 0., 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def adv_criterion(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

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

        self.log("gen_loss_PM", gen_loss_a)
        self.log("gen_loss_MP", gen_loss_b)
        self.log("disc_loss_M", disc_loss_a)
        self.log("disc_loss_P", disc_loss_b)

        if batch_idx % self.log_image_every_n_steps == 0:
            self.logger.experiment.add_images(
                "fake_A",
                self.gen_b(real_b)[0].detach().cpu().type(torch.float32),
                self.current_epoch
            )
            self.logger.experiment.add_images(
                "fake_B",
                self.gen_a(real_a)[0].detach().cpu().type(torch.float32),
                self.current_epoch
            )
