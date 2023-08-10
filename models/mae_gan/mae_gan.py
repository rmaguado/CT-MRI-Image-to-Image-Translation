import torch
import torch.nn as nn

from models.mae_gan.blocks import EncoderViT, DecoderViT, DiscriminatorViT

class MaeGanOutputs:
    def __init__(self, loss, pred, mask, reconstruction_loss):
        self.loss = loss
        self.pred = pred
        self.mask = mask
        self.reconstruction_loss = reconstruction_loss

class MaeGan(nn.Module):
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

        self.mode = "masked_modeling"

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

    def patchify(self, imgs):
        """Converts images to patches
        
        Args:
            imgs (torch.Tensor): [N, channels, H, W]
        Returns:
            x (torch.Tensor): [N, L, patch_size**2 *channels]
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """Converts patches to images
        
        Args:
            x (torch.Tensor): [N, L, patch_size**2 *channels]
        Returns:
            imgs (torch.Tensor): [N, channels, H, W]
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs
    
    def reconstruction_loss(self, imgs, pred, mask):
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

        if self.exclude_mask_loss:
            # [N, L], mean loss per patch: 
            loss = loss.mean(dim=-1)
            return (loss * mask).sum() / mask.sum()
        return loss.mean()
    
    def real_discriminator_loss(self, pred):
        """Computes LSGAN discriminator loss for real images.
        
        Args:
            pred (torch.Tensor): [N, L]
        Returns:
            loss (torch.Tensor): [1]
        """
        return (pred**2).mean()
    
    def fake_discriminator_loss(self, pred, mask):
        """Computes LSGAN discriminator loss for fake images.
        
        Args:
            pred (torch.Tensor): [N, L]
            mask (torch.Tensor): [N, L], 0 is keep, 1 is remove
        Returns:
            loss (torch.Tensor): [1]
        """
        loss = pred**2
        return loss[mask == 1].mean()
    
    def adversarial_loss(self, discriminator_pred):
        """Computes LSGAN adversarial loss
        
        Fake images should not contain unmasked patches.
        
        Args:
            discriminator_pred (torch.Tensor): [N, L]
            mask (torch.Tensor): [N, L], 0 is keep, 1 is remove
        Returns:
            loss (torch.Tensor): [1]
        """
        # [N, L], mean loss per patch: 
        loss = (discriminator_pred - 1) ** 2
        return loss.mean()
    
    def real_loss(self, x):
        """Computes LSGAN real loss
        
        Args:
            x (torch.Tensor): [N, channels, H, W]
        Returns:
            loss (torch.Tensor): [1]
        """
        latent, mask, ids_restore = self.encoder(x, mask_ratio=0)
        
        real_pred = self.discriminator(latent, ids_restore)
        return self.real_discriminator_loss(real_pred)

    def fake_loss(self, pred, mask):
        """Computes LSGAN fake loss
        
        Args:
            pred (torch.Tensor): [N, L, p*p*channels]
        Returns:
            discriminator_loss (torch.Tensor): [1]
        """
        latent, _, ids_restore = self.encoder(
            self.unpatchify(pred.detach()), mask_ratio=0
        )
        
        fake_pred = self.discriminator(latent, ids_restore)
        
        discriminator_loss = self.fake_discriminator_loss(
            fake_pred, mask
        )
        
        adversarial_loss = self.adversarial_loss(fake_pred)
        
        return discriminator_loss, adversarial_loss
    
    def forward_masked_modeling(self, x, input_type):
        # generator step
        latent, mask, ids_restore = self.encoder(x, self.mask_ratio)
        pred = self.decoders[input_type](latent, ids_restore)
        
        reconstruction_loss = self.reconstruction_loss(x, pred, mask)

        # gan step
        # real
        real_discriminator_loss = self.real_loss(x)

        # fake
        fake_discriminator_loss, adversarial_loss = self.fake_loss(pred, mask)
        
        loss = reconstruction_loss + adversarial_loss + \
            self.discriminator_loss_weight * (
                real_discriminator_loss + fake_discriminator_loss
            )

        return MaeGanOutputs(loss, pred, mask, reconstruction_loss)
    
    def forward_translation(self, x, input_type):
        latent, mask, ids_restore = self.encoder(x, 0)

        first_decoder = [
            x for x in self.decoders.keys() if x != input_type
        ][0]
        # [N, L, p*p*channels] :
        transfer_pred = self.decoders[first_decoder](latent, ids_restore)
        latent, mask, ids_restore = self.encoder(
            self.unpatchify(transfer_pred), mask_ratio=0
        )
        # [N, L, p*p*channels] :
        pred = self.decoders[input_type](latent, ids_restore) 

        loss = self.reconstruction_loss(x, pred, mask)
        
        return MaeGanOutputs(loss, pred, mask)

    def forward(self, x, input_type="CT"):
        if self.mode == "masked_modeling":
            return self.forward_masked_modeling(x, input_type)
        elif self.mode == "translation":
            return self.forward_translation(x, input_type)

if __name__ == "__main__":
    in_chans = 1
    device = torch.device("cuda:0")

    test_image = torch.rand(32, in_chans, 512, 512).to(device)
    
    model = MaeGan(img_size=512, in_chans=in_chans)

    model.to(device)
    model.train()
    output = model(test_image)

    reconstructed = model.unpatchify(output.pred)
    print(reconstructed.shape)
    