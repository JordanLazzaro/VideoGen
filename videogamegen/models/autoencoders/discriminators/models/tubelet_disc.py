import torch
from torch import nn
from itertools import zip_longest
from einops import rearrange

from videogamegen.models.modules import ResBlockDown3d 


class TubeletDiscriminator(nn.Module):
    ''' bunch of downsampling resblocks + leaky relu '''
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels  = config.in_channels,
                out_channels = config.init_channels,
                kernel_size  = (3, 3, 3),
                padding      = 'same'
            ),
            nn.LeakyReLU()
        )

        channels = [config.init_channels * cm for cm in config.channel_multipliers]
        space_downsample = [
            ResBlockDown3d(
                in_channels  = channels[i],
                out_channels = channels[i+1],
                space_only   = True
            )
            for i in range(config.num_space_downsamples)
        ]
        time_downsample = [
            ResBlockDown3d(
                in_channels  = channels[i+1] if i < config.num_space_downsamples else channels[i],
                out_channels = channels[i+1],
                time_only    = True
            )
            for i in range(config.num_time_downsamples)
        ]

        self.downsample = nn.Sequential(*[
            block
            for pair in zip_longest(space_downsample, time_downsample)
            for block in pair if block is not None
        ])

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels  = channels[-1],
                out_channels = channels[-1],
                kernel_size  = (3, 3, 3),
                padding      = 'same'
            ),
            nn.LeakyReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(4 * 4 * channels[-1], channels[-1]), # 4x4 after downsampling
            nn.LeakyReLU(),
            nn.Linear(channels[-1], 1)
        )

    def forward(self, x):
        assert len(x.shape) == 5, 'input shape must be (B, C, T, H, W)'
        x = self.conv1(x)
        x = self.downsample(x)
        x = self.conv2(x)
        x = rearrange(x, 'b ... -> b (...)')
        x = self.mlp(x)
        return x