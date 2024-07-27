import torch
from torch import nn
from itertools import zip_longest
from einops import rearrange
from torch.autograd import grad as torch_grad
from torch.nn import functional as F

from config import Config

from modules import (
    FSQ,
    Encoder,
    Decoder
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
            num_res_blocks        = config.encoder.num_res_blocks,
            causal                = config.encoder.causal
        )

        self.fsq = FSQ(levels = config.quantization.levels)

        self.decoder = Decoder(
            in_channels         = config.decoder.in_channels,
            out_channels        = config.decoder.out_channels,
            init_channels       = config.decoder.init_channels,
            channel_multipliers = config.decoder.channel_multipliers,
            num_space_upsamples = config.decoder.num_space_upsamples,
            num_time_upsamples  = config.decoder.num_time_upsamples,
            num_res_blocks      = config.decoder.num_res_blocks,
            causal              = config.decoder.causal
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