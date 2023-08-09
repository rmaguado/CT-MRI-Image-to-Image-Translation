import torch
import torch.nn as nn

from models.mae_gan.blocks import EncoderViT, DecoderViT, DiscriminatorViT

class MaeGanOutputs:
    def __init__(self, loss, pred, mask):
        self.loss = loss
        self.pred = pred
        self.mask = mask

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
        discriminator_depth: int = 4,
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
            encoder_embed_dim=encoder_embed_dim,
            encoder_num_heads=encoder_num_heads,
            discriminator_depth=discriminator_depth,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

        self.norm_pix_loss = norm_pix_loss

    def patchify(self, imgs):
        """
        imgs: (N, channels, H, W)
        x: (N, L, patch_size**2 *channels)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *channels)
        imgs: (N, channels, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs
    
    def reconstruction_loss(self, imgs, pred, mask):
        """
        imgs: [N, channels, H, W]
        pred: [N, channels, H, W]
        mask: [N, L], 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)
        pred = self.patchify(pred)
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
    
    def discriminator_loss(self, pred, mask, truth):
        """Computes LSGAN discriminator loss
        pred: [N, L, 2]
        mask: [N, L], 0 is keep, 1 is remove
        truth: [N, L], 0 is real, 1 is fake
        """
        # [N, L], mean loss per patch: 
        loss = (pred - truth) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum()
    
    def adversarial_loss(self, discriminator_pred, mask):
        """Computes LSGAN adversarial loss
        discriminator_pred: [N, L, 2]
        mask: [N, L], 0 is keep, 1 is remove
        """
        # [N, L], mean loss per patch: 
        loss = discriminator_pred[:, :, 0] ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum()
    
    def real_loss(self, latent):
        """Computes LSGAN real loss
        latent: [N, L, E]
        """
        real_pred = self.discriminator(latent)
        return self.discriminator_loss(
            real_pred, torch.ones_like(real_pred)
        )
        
    def fake_loss(self, pred):
        """
        pred: [N, L, p*p*channels]
        """
        latent, mask, ids_restore = self.encoder(
            self.unpatchify(pred.detach()), mask_ratio=0
        )
        fake_pred = self.discriminator(latent)
        
        discriminator_loss = self.discriminator_loss(
            fake_pred, torch.zeros_like(fake_pred)
        )
        adversarial_loss = self.adversarial_loss(fake_pred, mask)
        return discriminator_loss, adversarial_loss
    
    def forward_masked_modeling(self, x, input_type):
        # generator step
        latent, mask, ids_restore = self.encoder(x, self.mask_ratio)
        pred = self.decoders[input_type](latent, ids_restore)
        
        generator_loss = self.reconstruction_loss(x, pred, mask)
        
        # gan step
        # real
        real_discriminator_loss = self.real_loss(latent)
        # fake
        fake_discriminator_loss, adversarial_loss = self.fake_loss(pred)
        
        loss = generator_loss + self.discriminator_loss_weight * (
            real_discriminator_loss + fake_discriminator_loss
        )
        
        return MaeGanOutputs(loss, pred, mask)
    
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
    