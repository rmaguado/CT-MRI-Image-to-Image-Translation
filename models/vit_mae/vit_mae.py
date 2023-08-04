import torch
import torch.nn as nn

from models.vit_mae.vit_blocks import EncoderViT, DecoderViT

class ViTOutputs:
    def __init__(self, loss, pred, mask):
        self.loss = loss
        self.pred = pred
        self.mask = mask

class VitMae(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(
        self,
        img_size=512,
        patch_size=16,
        in_chans=1,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_chans = in_chans

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
        self.decoder = DecoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
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

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, channels, H, W]
        pred: [N, L, p*p*channels]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x, mask_ratio=0.75):
        latent, mask, ids_restore = self.encoder(x, mask_ratio)
        pred = self.decoder(latent, ids_restore)
        
        loss = self.forward_loss(x, pred, mask)
        return ViTOutputs(loss, pred, mask)

if __name__ == "__main__":
    in_chans = 1
    device = torch.device("cuda:0")

    test_image = torch.rand(32, in_chans, 512, 512).to(device)
    
    model = VitMae(img_size=512, in_chans=in_chans)

    model.to(device)
    model.train()
    output = model(test_image)

    reconstructed = model.unpatchify(output.pred)
    print(reconstructed.shape)
    