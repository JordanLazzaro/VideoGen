import torch
from torch import nn
from itertools import zip_longest
from einops import rearrange
from torch.autograd import grad as torch_grad
from torch.nn import functional as F

from config import Config

from modules import ( 
    CausalConv3d,
    ResBlock3d,
    FSQ,
    ResBlockDown2d,
    ResBlockDown3d,
    AdaptiveGroupNorm,
    Upsample3d
)
from videogen.models.tokenizers.tokenizer import Tokenizer


class EncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            num_res_blocks,
            kernel_size=(3,3,3),
            space_only=False,
            time_only=False
        ):
        super().__init__()
        if space_only:
            stride = (1, 2, 2)
        elif time_only:
            stride = (2, 1, 1)
        else:
            stride = (2, 2, 2)
        
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride
            ),
            nn.Sequential(*[
                ResBlock3d(
                    in_channels  = out_channels,
                    out_channels = out_channels
                )
                for _ in range(num_res_blocks)
            ])
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            num_res_blocks,
            kernel_size=(3,3,3),
            space_only=False,
            time_only=False
        ):
        super().__init__()
        if space_only:
            stride = (1, 2, 2)
        elif time_only:
            stride = (2, 1, 1)
        else:
            stride = (2, 2, 2)
        
        self.block = nn.Sequential(
            nn.Sequential(*[
                ResBlock3d(
                    in_channels  = in_channels,
                    out_channels = in_channels
                )
                for _ in range(num_res_blocks)
            ]),
            nn.ConvTranspose3d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                output_padding = (stride[0]-1, stride[1]-1, stride[2]-1)
            )
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            init_channels,
            channel_multipliers,
            num_space_downsamples,
            num_time_downsamples,
            num_res_blocks
        ):
        super().__init__()
        channels = [init_channels * cm for cm in channel_multipliers]
        self.in_conv = nn.Conv3d(
            in_channels  = in_channels,
            out_channels = channels[0],
            kernel_size  = (3,3,3)
        )
        self.space_downsample_blocks = nn.Sequential(*[
            EncoderBlock(
                in_channels   = channels[i],
                out_channels  = channels[i+1],
                nblocks       = num_res_blocks,
                kernel_size   = (3,3,3),
                space_only    = True
            )
            for i in range(num_space_downsamples)
        ])
        time_start_channel = num_space_downsamples
        self.time_downsample_blocks = nn.ModuleList([
            EncoderBlock(
                in_channels   = channels[i+time_start_channel],
                out_channels  = channels[i+time_start_channel+1],
                nblocks       = num_res_blocks,
                kernel_size   = (3,3,3),
                time_only     = True
            )
            for i in range(num_time_downsamples)
        ])
        self.out_conv = nn.Sequential(
            AdaptiveGroupNorm(out_channels),
            nn.SiLU(),
            nn.Conv3d(
                in_channels  = channels[-1],
                out_channels = out_channels,
                kernel_size  = (3,3,3)
            )
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.space_downsample_blocks(x)
        x = self.time_downsample_blocks(x)
        x = self.out_conv(x)
        return x
    

class Decoder(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            init_channels,
            channel_multipliers,
            num_space_upsamples,
            num_time_upsamples,
            num_res_blocks
        ):
        super().__init__()
        channels = [init_channels * cm for cm in channel_multipliers][::-1]
        self.in_conv = nn.Conv3d(
            in_channels  = in_channels,
            out_channels = channels[0],
            kernel_size  = (3,3,3)
        )
        self.time_upsample_blocks = nn.Sequential(*[
            DecoderBlock(
                in_channels   = channels[i],
                out_channels  = channels[i+1],
                nblocks       = num_res_blocks,
                kernel_size   = (3,3,3),
                time_only     = True
            )
            for i in range(num_time_upsamples)
        ])
        space_start_channel = num_time_upsamples
        self.space_upsample_blocks = nn.Sequential(*[
            DecoderBlock(
                in_channels   = channels[i+space_start_channel],
                out_channels  = channels[i+space_start_channel+1],
                nblocks       = num_res_blocks,
                kernel_size   = (3,3,3),
                space_only    = True
            )
            for i in range(num_space_upsamples)
        ])
        self.out_conv = nn.Conv3d(
            in_channels  = channels[-1],
            out_channels = out_channels,
            kernel_size  = (3,3,3)
        )
    
    def forward(self, x):
        x = self.in_conv(x)
        x = self.time_upsample_blocks(x)
        x = self.space_upsample_blocks(x)
        x = self.out_conv(x)
        return x


class FSQVAE(Tokenizer):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = Encoder(
            in_channels           = config.encoder.in_channels,
            out_channels          = config.encoder.out_channels,
            init_channels         = config.encoder.init_channels,
            channel_multipliers   = config.encoder.channel_multipliers,
            num_space_downsamples = config.encoder.num_space_downsamples,
            num_time_downsamples  = config.encoder.num_time_downsamples,
            num_res_blocks        = config.encoder.num_res_blocks
        )

        self.fsq = FSQ(levels = config.fsq.levels)

        self.decoder = Decoder(
            in_channels         = config.decoder.in_channels,
            out_channels        = config.decoder.out_channels,
            init_channels       = config.decoder.init_channels,
            channel_multipliers = config.decoder.channel_multipliers,
            num_space_upsamples = config.decoder.num_space_upsamples,
            num_time_upsamples  = config.decoder.num_time_upsamples,
            num_res_blocks      = config.decoder.num_res_blocks
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        return self.fsq(z)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)
    
    def reconstruction_loss(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.config.loss.recon_loss_type == 'mae':
            return F.l1_loss(x_hat, x)
        elif self.config.loss.recon_loss_type == 'mse':
            return F.mse_loss(x_hat, x)
        else:
            raise ValueError('invalid reconstruction loss type')

    def forward(self, x):
        z = self.encode(x)
        z_q = self.quantize(z)
        x_hat = self.decode(z_q)
        return { 'z': z, 'z_q': z_q, 'x_hat': x_hat }