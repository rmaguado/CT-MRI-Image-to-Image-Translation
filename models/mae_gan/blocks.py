import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from models.utils.pos_embed import get_2d_sincos_pos_embed
from models.utils.vit_blocks import _init_weights

def random_masking(x, mask_ratio):
    """Perform per-sample random masking by per-sample shuffling.
    
    Per-sample shuffling is done by argsort random noise.
    
    Args:
        x (torch.tensor): [N, L, D] sequence
        mask_ratio (float): ratio of masking
    Returns:
        x_masked (torch.tensor): [N, L*(1-mask_ratio), D] masked sequence.
            0 is keep, 1 is remove
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore

class EncoderViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, encoder_embed_dim)
        self.num_patches = ( img_size // patch_size ) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, encoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(encoder_depth)])
        self.norm = norm_layer(encoder_embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(_init_weights)

    def forward(self, x, mask_ratio: float = 0.):
        """
        Args:
            x (torch.tensor): [N, C, H, W] image
            mask_ratio (float): ratio of masking
        Returns:
            x (torch.tensor): [N, L, D] sequence
        """
        # embed patches
        x = self.patch_embed(x)
        
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore


class DecoderViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        encoder_embed_dim=1024,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_patches = ( img_size // patch_size ) ** 2

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        
         # decoder to patch

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(_init_weights)

    def forward(self, x, ids_restore):
        """
        Args:
            x (torch.tensor): [N, L, D] sequence
            ids_restore (torch.tensor): [N, L] ids to restore the original order
        Returns:
            x (torch.tensor): [N, L, D] sequence
        """
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

class DiscriminatorViT(nn.Module):
    def __init__(
        self,
        img_size=512,
        patch_size=16,
        in_chans=1,
        encoder_embed_dim=1024,
        discriminator_embed_dim=512,
        discriminator_num_heads=16,
        discriminator_depth=4,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        
        self.backbone = DecoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=discriminator_embed_dim,
            decoder_depth=discriminator_depth,
            decoder_num_heads=discriminator_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )
        
        self.backbone.decoder_pred = nn.Linear(
            discriminator_embed_dim, 1, bias=True
        )
        
        self.initialize_weights()
        
    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(_init_weights)
        
    def forward(self, x, ids_restore):
        return self.backbone(x, ids_restore)
        