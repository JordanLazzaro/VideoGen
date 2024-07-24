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
            nblocks               = config.encoder.nblocks
        )

        self.fsq = FSQ(levels = config.fsq.levels)

        self.decoder = Decoder(
            in_channels         = config.decoder.in_channels,
            out_channels        = config.decoder.out_channels,
            init_channels       = config.decoder.init_channels,
            channel_multipliers = config.encoder.channel_multipliers,
            num_space_upsamples = config.decoder.num_space_upsamples,
            num_time_upsamples  = config.decoder.num_time_upsamples,
            nblocks             = config.decoder.nblocks
        )

    def encode(self, x):
        return self.encoder(x)

    def quantize(self, z):
        return self.fsq(z)

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x):
        z = self.encode(x)
        z_q = self.quantize(z)
        x_hat = self.decode(z_q)
        return { 'z': z, 'z_q': z_q, 'x_hat': x_hat }