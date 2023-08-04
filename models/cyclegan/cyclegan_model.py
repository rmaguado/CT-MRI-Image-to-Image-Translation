"""
Implements cycleGAN from https://arxiv.org/pdf/1703.10593.pdf
"""

import itertools
from collections import OrderedDict

import torch
from torch import nn

from models.cyclegan.image_pool import ImagePool
from models.cyclegan import networks


class CycleGANModel(nn.Module):
    """
    This class implements the CycleGAN model, for learning image-to-image
    translation without paired data.
    """
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            generator_filters: int = 64,
            discriminator_filters: int = 64,
            discriminator_layers: int = 3,
            generator_name: str = 'resnet_9blocks',
            discriminator_name: str = 'basic',
            gan_mode: str = 'lsgan',
            direction: str = 'AtoB',
            norm_type: str = 'instance',
            lambda_identity: float = 0.5,
            lambda_A: float = 10.0,
            lambda_B: float = 10.0,
            beta1: float = 0.5,
            lr: float = 0.0002,
            no_dropout: bool = False,
            init_type: str = 'normal',
            init_gain: float = 0.02,
            pool_size: int = 50,
            gpu_ids: list = [0],
            device: str = "cuda:0",
            isTrain: bool = True
    ):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.isTrain = isTrain
        self.device = device
        self.direction = direction
        self.lambda_identity = lambda_identity
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B

        self.loss_names = [
            'D_A',
            'G_A',
            'cycle_A',
            'idt_A',
            'D_B',
            'G_B',
            'cycle_B',
            'idt_B'
        ]
        visual_names_a = ['real_A', 'fake_B', 'rec_A']
        visual_names_b = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and lambda_identity > 0.0:
            visual_names_a.append('idt_B')
            visual_names_b.append('idt_A')

        self.visual_names = visual_names_a + visual_names_b
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        self.generator_a = networks.define_G(
            in_channels,
            out_channels,
            generator_filters,
            generator_name,
            norm_type,
            not no_dropout,
            init_type,
            init_gain,
            self.gpu_ids
        )
        self.generator_b = networks.define_G(
            out_channels,
            in_channels,
            generator_filters,
            generator_name,
            norm_type,
            not no_dropout,
            init_type,
            init_gain,
            self.gpu_ids
        )

        if self.isTrain:  # define discriminators
            self.discriminator_a = networks.define_D(
                out_channels,
                discriminator_filters,
                discriminator_name,
                discriminator_layers,
                norm_type,
                init_type,
                init_gain,
                self.gpu_ids
            )
            self.discriminator_b = networks.define_D(
                in_channels,
                discriminator_filters,
                discriminator_name,
                discriminator_layers,
                norm_type,
                init_type,
                init_gain,
                self.gpu_ids
            )

        if self.isTrain:
            if lambda_identity > 0.0:
                assert in_channels == out_channels
            self.fake_a_pool = ImagePool(pool_size)
            self.fake_b_pool = ImagePool(pool_size)

            self.criterion_gan = networks.GANLoss(gan_mode).to(self.device)
            self.criterion_cycle = torch.nn.L1Loss()
            self.criterion_idt = torch.nn.L1Loss()

            self.optimizers = []
            self.optimizer_g = torch.optim.Adam(
                itertools.chain(
                    self.generator_a.parameters(),
                    self.generator_b.parameters()
                ),
                lr=lr,
                betas=(beta1, 0.999)
            )
            self.optimizer_d = torch.optim.Adam(
                itertools.chain(
                    self.discriminator_a.parameters(),
                    self.discriminator_b.parameters()
                ),
                lr=lr,
                betas=(beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_g)
            self.optimizers.append(self.optimizer_d)

            self.real_a, self.real_b = None, None
            self.fake_a, self.fake_b = None, None
            self.rec_a, self.rec_b = None, None

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        
        Args:
            nets (network list): A list of networks
            requires_grad (bool): Whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary
        pre-processing steps.

        The option 'direction' can be used to swap domain A and domain B.

        Args:
            data (dict): include the data itself and its metadata information.
        """
        a_to_b = self.direction == 'AtoB'
        self.real_a = torch.tensor(
            data['A' if a_to_b else 'B']
        ).to(self.device)
        self.real_b = torch.tensor(
            data['B' if a_to_b else 'A']
        ).to(self.device)

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and
        <test>.
        """
        self.fake_b = self.generator_a(self.real_a)  # G_A(A)
        self.rec_a = self.generator_b(self.fake_b)   # G_B(G_A(A))
        self.fake_a = self.generator_b(self.real_b)  # G_B(B)
        self.rec_b = self.generator_a(self.fake_a)   # G_A(G_B(B))

    def backward_d_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator.

        We also call loss_D.backward() to calculate the gradients.

        Args:
            - netD (network): the discriminator network
            - real (tensor array): real images
            - fake (tensor array): images generated by a generator

        Returns:
            (tensor): the total loss for discriminator
        """
        # Real
        pred_real = netD(real)
        loss_d_real = self.criterion_gan(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_d_fake = self.criterion_gan(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_d_real + loss_d_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_b_pool.query(self.fake_b)
        self.loss_D_A = self.backward_d_basic(
            self.discriminator_a,
            self.real_b,
            fake_B
        )

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_a_pool.query(self.fake_a)
        self.loss_D_B = self.backward_d_basic(
            self.discriminator_b,
            self.real_a,
            fake_A
        )

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.lambda_identity
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.generator_a(self.real_b)
            self.loss_idt_A = self.criterion_idt(
                self.idt_A,
                self.real_b
            ) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.generator_b(self.real_a)
            self.loss_idt_B = self.criterion_idt(
                self.idt_B,
                self.real_a
            ) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterion_gan(self.discriminator_a(self.fake_b), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterion_gan(self.discriminator_b(self.fake_a), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterion_cycle(self.rec_a, self.real_a) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterion_cycle(self.rec_b, self.real_b) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()
    
    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.discriminator_a, self.discriminator_b], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_g.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_g.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.discriminator_a, self.discriminator_b], True)
        self.optimizer_d.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_d.step()  # update D_A and D_B's weights
