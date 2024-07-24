import torch
from torch import nn
from itertools import zip_longest
from einops import rearrange

from modules import ResBlockDown2d


class PatchDiscriminator(nn.Module):
    ''' bunch of downsampling resblocks + leaky relu '''
    def __init__(self, config):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(
                in_channels  = config.in_channels,
                out_channels = config.init_channels,
                kernel_size  = (3, 3),
                padding      = 'same'
            ),
            nn.LeakyReLU()
        )

        channels = [config.init_channels * cm for cm in config.channel_multipliers]
        self.downsample = nn.Sequential(*[
            ResBlockDown2d(
                in_channels  = channels[i],
                out_channels = channels[i+1]
            )
            for i in range(config.num_space_downsamples)
        ])

        self.out_conv = nn.Conv2d(
            in_channels  = channels[-1],
            out_channels = config.out_channels,
            kernel_size  = (1, 1)
        )

    def forward(self, x):
        assert len(x.shape) == 4, 'input shape must be (B, C, H, W)'
        x = self.in_conv(x)
        x = self.downsample(x)
        x = self.out_conv(x)
        return x