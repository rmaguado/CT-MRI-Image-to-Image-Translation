import torch
from torch import nn

import lightning.pytorch as pl

from models.mae_gan.blocks import EncoderViT, DecoderViT, DiscriminatorViT
from schedulers.schedulers import CosineAnnealingWarmupRestarts

class MaeGanOutputs:
    def __init__(
        self,
        loss,
        pred=None,
        mask=None,
        reconstruction_loss=0,
        adversarial_loss=0,
        discriminator_real_loss=0,
        discriminator_fake_loss=0,
        cycle_loss=0
    ):
        self.loss = loss
        self.pred = pred
        self.mask = mask
        self.reconstruction_loss = reconstruction_loss
        self.adversarial_loss = adversarial_loss
        self.discriminator_real_loss = discriminator_real_loss
        self.discriminator_fake_loss = discriminator_fake_loss
        self.cycle_loss = cycle_loss


class MaeGanModel(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 1,
        encoder_embed_dim: int = 1024,
        encoder_depth: int = 24,
        encoder_num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        discriminator_embed_dim: int = 512,
        discriminator_depth: int = 4,
        discriminator_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer = nn.LayerNorm,
        norm_pix_loss: bool = False,
        mask_ratio: float = 0.75,
        exclude_mask_loss: bool = False,
        discriminator_loss_weight: float = 2.0
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.exclude_mask_loss = exclude_mask_loss
        self.discriminator_loss_weight = discriminator_loss_weight
        self.crossentropy = nn.CrossEntropyLoss()

        self.mode = "mae"

        self.encoder = EncoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            encoder_embed_dim=encoder_embed_dim,
            encoder_depth=encoder_depth,
            encoder_num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )
        self.decoders = nn.ModuleDict({
            modality : DecoderViT(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    encoder_embed_dim=encoder_embed_dim,
                    decoder_embed_dim=decoder_embed_dim,
                    decoder_depth=decoder_depth,
                    decoder_num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    norm_layer=norm_layer
                ) for modality in ["CT", "MR"]
        })

        self.discriminator = DiscriminatorViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            encoder_embed_dim=encoder_embed_dim,
            discriminator_embed_dim=discriminator_embed_dim,
            discriminator_num_heads=discriminator_num_heads,
            discriminator_depth=discriminator_depth,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

        self.norm_pix_loss = norm_pix_loss

    def enable_decoder_grads(self):
        for decoder in self.decoders.values():
            decoder.enable_grad()

    def disable_decoder_grads(self):
        for decoder in self.decoders.values():
            decoder.disable_grad()

    def enable_discriminator_grads(self):
        self.discriminator.enable_grad()

    def disable_discriminator_grads(self):
        self.discriminator.disable_grad()

    def set_mode(self, mode):
        self.mode = mode
        if mode not in ["cycle", "mae"]:
            raise ValueError("Unknown mode")

    def patchify(self, imgs):
        """Converts images to patches
        
        Args:
            imgs (torch.Tensor): [N, channels, H, W]
        Returns:
            patches (torch.Tensor): [N, L, patch_size**2 *channels]
        """
        assert imgs.shape[2] == imgs.shape[3] and \
            imgs.shape[2] % self.patch_size == 0

        height = width = imgs.shape[2] // self.patch_size
        patches = imgs.reshape(
            shape=(
                imgs.shape[0],
                self.in_chans,
                height,
                self.patch_size,
                width,
                self.patch_size
            )
        )
        patches = torch.einsum('nchpwq->nhwpqc', patches)
        patches = patches.reshape(
            shape=(
                imgs.shape[0],
                height * width,
                self.patch_size**2 * self.in_chans
            )
        )
        return patches

    def unpatchify(self, patches):
        """Converts patches to images
        
        Args:
            patches (torch.Tensor): [N, L, patch_size**2 *channels]
        Returns:
            imgs (torch.Tensor): [N, channels, H, W]
        """
        height = width = int(patches.shape[1]**.5)
        assert height * width == patches.shape[1]

        patches = patches.reshape(
            shape=(
                patches.shape[0],
                height,
                width,
                self.patch_size,
                self.patch_size,
                self.in_chans
            )
        )
        patches = torch.einsum('nhwpqc->nchpwq', patches)
        imgs = patches.reshape(
            shape=(
                patches.shape[0],
                self.in_chans,
                height * self.patch_size,
                height * self.patch_size
                )
        )
        return imgs

    def reconstruction_loss(self, imgs, pred, mask=None):
        """Computes reconstruction loss of the MAE
        
        Args:
            imgs (torch.Tensor): [N, channels, H, W]
            pred (torch.Tensor): [N, L, p*p*channels]
            mask (torch.Tensor): [N, L], 0 is keep, 1 is remove
        Returns:
            loss (torch.Tensor): [1]
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2

        if self.exclude_mask_loss and mask is not None:
            # [N, L], mean loss per patch:
            loss = loss.mean(dim=-1)
            return (loss * mask).sum() / mask.sum()
        return loss.mean()

    def fake_discriminator_loss(self, pred):
        """Computes LSGAN discriminator loss for fake images.
        
        Args:
            pred (torch.Tensor): [N, L]
        Returns:
            loss (torch.Tensor): [1]
        """
        latent_discriminator, _, ids_restore_discriminator = self.encoder(
            self.unpatchify(pred), mask_ratio=0
        )
        fake_pred_discriminator = self.discriminator(
            latent_discriminator, ids_restore_discriminator
        )
        loss = fake_pred_discriminator**2
        return fake_pred_discriminator, loss.mean() #loss[mask == 1].mean() for only masked patches

    def adversarial_loss(self, pred_real, pred_fake):
        """Computes LSGAN adversarial loss
        
        Fake images should not contain unmasked patches.
        
        Args:
            pred_real (torch.Tensor): [N, L]
            pred_fake (torch.Tensor): [N, L]
        Returns:
            loss (torch.Tensor): [1]
        """
        # [N, L], mean loss per patch:
        loss = (pred_fake - 1)**2 + (pred_real)**2
        return loss.mean()

    def real_discriminator_loss(self, data):
        """Computes LSGAN real loss
        
        Args:
            data (torch.Tensor): [N, channels, H, W]
        Returns:
            loss (torch.Tensor): [1]
        """
        latent, _, ids_restore = self.encoder(data, mask_ratio=0)

        real_pred = self.discriminator(latent, ids_restore)
        loss = (real_pred**2).mean()
        return real_pred, loss

    def forward_cycle(self, data, input_type):
        latent, _, ids_restore = self.encoder(data)

        first_decoder = [
            x for x in self.decoders.keys() if x != input_type
        ][0]
        # [N, L, p*p*channels] :
        transfer_pred = self.decoders[first_decoder](latent, ids_restore)
        latent, _, ids_restore = self.encoder(
            self.unpatchify(transfer_pred)
        )
        # [N, L, p*p*channels] :
        pred = self.decoders[input_type](latent, ids_restore)

        loss = self.reconstruction_loss(data, pred)

        return pred, loss

    def forward_generate(self, data, input_type):
        # generator step
        latent, mask, ids_restore = self.encoder(data, self.mask_ratio)
        pred = self.decoders[input_type](latent, ids_restore)

        reconstruction_loss = self.reconstruction_loss(data, pred, mask)
        # real
        real_discriminator_pred, real_discriminator_loss = \
            self.real_discriminator_loss(data)

        # fake
        fake_discriminator_pred, fake_discriminator_loss =\
            self.fake_discriminator_loss(pred)
        adversarial_loss = self.adversarial_loss(
            real_discriminator_pred, fake_discriminator_pred
        )

        return (
            pred,
            reconstruction_loss,
            adversarial_loss,
            real_discriminator_loss * self.discriminator_loss_weight,
            fake_discriminator_loss * self.discriminator_loss_weight
        )

    def forward(self, data, input_type):
        if self.mode == "mae":
            return self.forward_generate(data, input_type)
        elif self.mode == "cycle":
            return self.forward_cycle(data, input_type)
        raise ValueError("Unknown mode")


class MaeGanLM(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = MaeGanModel(**config["model"])
        self.mode_counter = 0
        self.log_image_every_n_steps = config["log_image_every_n_steps"]
        self.gradient_clip_val = config["gradient_clip_val"]
        self.accumulate_grad_batches = config["accumulate_grad_batches"]
        self.automatic_optimization = False

    def log_images(self, pred, batch_idx):
        self.logger.experiment.add_image(
            self.model.mode + "/" + ["A", "B"][batch_idx % 2],
            self.model.unpatchify(
                pred
            )[0].detach().cpu().type(torch.float32),
            batch_idx // self.accumulate_grad_batches,
            dataformats="WH"
        )

    def mae_step(self, batch):
        pred, reconstruction_loss, adversarial_loss, \
            real_discriminator_loss, fake_discriminator_loss = \
                self.model(*batch)
        self.model.enable_decoder_grads()
        self.model.disable_discriminator_grads()
        self.manual_backward(reconstruction_loss, retain_graph=True)
        self.manual_backward(adversarial_loss, retain_graph=True)

        self.model.disable_decoder_grads()
        self.model.enable_discriminator_grads()
        self.manual_backward(real_discriminator_loss, retain_graph=True)
        self.manual_backward(fake_discriminator_loss, retain_graph=True)

        self.log("loss/reconstruction", reconstruction_loss.item())
        self.log("loss/adversarial", adversarial_loss.item())
        self.log("loss/discriminator_real", real_discriminator_loss.item())
        self.log("loss/discriminator_fake", fake_discriminator_loss.item())
        return pred

    def cycle_step(self, batch):
        pred, loss = self.model(*batch)
        self.model.enable_decoder_grads()
        self.manual_backward(loss)

        self.log("loss/cycle", loss.item())
        return pred

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        if self.mode_counter == self.config["mode_repetitions"]:
            next_mode = "cycle" if self.model.mode == "mae" else "mae"
            self.model.set_mode(next_mode)
            self.mode_counter = 0
        #outputs = self.model(*batch)
        # break up optimization to multiple steps.

        if self.model.mode == "mae":
            pred = self.mae_step(batch)

        elif self.model.mode == "cycle":
            pred = self.cycle_step(batch)

        if batch_idx % self.accumulate_grad_batches == 0:
            self.log(
                "learning_rate",
                self.trainer.optimizers[0].param_groups[0]["lr"]
            )
            if batch_idx > 0:
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.gradient_clip_val,
                    gradient_clip_algorithm="norm"
                )
                opt.step()
                sch.step()
                opt.zero_grad()

        if batch_idx % self.log_image_every_n_steps in [0, 1, 2, 3]:
            self.log_images(pred, batch_idx)
        self.mode_counter += 1

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["scheduler"]["min_lr"]
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer, **self.config["scheduler"]
        )
        return [optimizer], [{"scheduler":lr_scheduler, "interval": "step"}]
