from torch import nn
import torch
from timm.models.vision_transformer import PatchEmbed, Block

from utils.pos_embed import get_2d_sincos_pos_embed
from utils.params import _init_weights


class ViT(nn.Module):
    def __init__(
        self,
        img_size=512,
        patch_size=16,
        in_chans=1,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(
            img_size,
            patch_size,
            in_chans,
            embed_dim
        )
        self.num_patches = ( img_size // patch_size ) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim),
            requires_grad=False
        )

        self.blocks = nn.ModuleList([
            Block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        self.pred = nn.Linear(
            embed_dim,
            patch_size**2 * in_chans,
            bias=True
        )

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches**.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(_init_weights)

    def disable_grad(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

    def enable_grad(self):
        for name, parameter in self.named_parameters():
            if 'pos_embed' not in name:
                parameter.requires_grad = True

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

    def forward(self, x):
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

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        # predictor projection
        x = self.pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return self.unpatchify(x)


class Downsampling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm=True,
        kernel_size=4,
        stride=2,
        padding=1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False)
        )
        if norm:
            self.block.append(nn.InstanceNorm2d(out_channels, affine=True))
        self.block.append(nn.LeakyReLU(0.3))

    def forward(self, x):
        return self.block(x)

class Upsampling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=False,
        kernel_size=4,
        stride=2,
        padding=1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
        )
        if dropout:
            self.block.append(nn.Dropout(0.5))
        self.block.append(nn.ReLU())

    def forward(self, x):
        return self.block(x)

class GeneratorUNet(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            hid_channels,
            tanh_activation=False,
        ):
        super().__init__()
        self.downsampling_blocks = nn.Sequential(
            Downsampling(in_channels, hid_channels, norm=False), #64x128x128
            Downsampling(hid_channels, hid_channels*2), #128x64x64
            Downsampling(hid_channels*2, hid_channels*4), #256x32x32
            Downsampling(hid_channels*4, hid_channels*8), #512x16x16
            Downsampling(hid_channels*8, hid_channels*8), #512x8x8
            Downsampling(hid_channels*8, hid_channels*8), #512x4x4
            Downsampling(hid_channels*8, hid_channels*8), #512x2x2
            Downsampling(hid_channels*8, hid_channels*8, norm=False), #512x1x1
        )
        self.upsampling_blocks = nn.Sequential(
            Upsampling(hid_channels*8, hid_channels*8, dropout=True), #(512+512)x2x2
            Upsampling(hid_channels*16, hid_channels*8, dropout=True), #(512+512)x4x4
            Upsampling(hid_channels*16, hid_channels*8, dropout=True), #(512+512)x8x8
            Upsampling(hid_channels*16, hid_channels*8), #(512+512)x16x16
            Upsampling(hid_channels*16, hid_channels*4), #(256+256)x32x32
            Upsampling(hid_channels*8, hid_channels*2), #(128+128)x64x64
            Upsampling(hid_channels*4, hid_channels), #(64+64)x128x128
        )
        out_layer = nn.ConvTranspose2d(
            hid_channels*2,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        ) #3x256x256

        self.feature_block = nn.Sequential(
            out_layer,
            nn.Tanh()
        ) if tanh_activation else out_layer
        
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(_init_weights)

    def forward(self, x):
        skips = []
        for down in self.downsampling_blocks:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        for up, skip in zip(self.upsampling_blocks, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)

        return self.feature_block(x)

class DiscriminatorUNet(nn.Module):
    def __init__(self, in_channels=1, hid_channels=32):
        super().__init__()
        self.block = nn.Sequential(
            Downsampling(in_channels, hid_channels, norm=False), #64x128x128
            Downsampling(hid_channels, hid_channels*2), #128x64x64
            Downsampling(hid_channels*2, hid_channels*4), #256x32x32
            Downsampling(hid_channels*4, hid_channels*8, stride=1), #512x31x31
            nn.Conv2d(hid_channels*8, 1, kernel_size=4, padding=1), #1x30x30
        )
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(_init_weights)

    def forward(self, x):
        return self.block(x)


class GeneratorViT(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16
    ):
        super().__init__()
        self.block = ViT(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads
        )

    def forward(self, x):
        return self.block(x)


class DiscriminatorViT(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16
    ):
        super().__init__()
        self.block = ViT(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads
        )

    def forward(self, x):
        return self.block(x)
