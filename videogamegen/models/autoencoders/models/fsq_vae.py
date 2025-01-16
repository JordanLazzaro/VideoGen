import torch
from torch import nn
from typing import Dict
from itertools import zip_longest
from einops import rearrange
from torch.autograd import grad as torch_grad
from torch.nn import functional as F

from videogamegen.config import Config

from videogamegen.models.modules import (
    FSQ,
    Encoder,
    Decoder
)
from videogamegen.models.autoencoders.autoencoder import Autoencoder


class FSQVAE(Autoencoder):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = Encoder(
            in_channels           = config.autoencoder.encoder.in_channels,
            out_channels          = config.autoencoder.encoder.out_channels,
            init_channels         = config.autoencoder.encoder.init_channels,
            channel_multipliers   = config.autoencoder.encoder.channel_multipliers,
            num_space_downsamples = config.autoencoder.encoder.num_space_downsamples,
            num_time_downsamples  = config.autoencoder.encoder.num_time_downsamples,
            num_res_blocks        = config.autoencoder.encoder.num_res_blocks,
            causal                = config.autoencoder.encoder.causal
        )

        self.fsq = FSQ(levels = config.autoencoder.quantization.levels)

        self.decoder = Decoder(
            in_channels         = config.autoencoder.decoder.in_channels,
            out_channels        = config.autoencoder.decoder.out_channels,
            init_channels       = config.autoencoder.decoder.init_channels,
            channel_multipliers = config.autoencoder.decoder.channel_multipliers,
            num_space_upsamples = config.autoencoder.decoder.num_space_upsamples,
            num_time_upsamples  = config.autoencoder.decoder.num_time_upsamples,
            num_res_blocks      = config.autoencoder.decoder.num_res_blocks,
            causal              = config.autoencoder.decoder.causal
        )

        self.config = config

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        return self.fsq(z)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)
    
    def loss(self, output: Dict) -> torch.Tensor:
        if self.config.autoencoder.loss.recon_loss_type == 'mae':
            return F.l1_loss(x_hat, x)
        elif self.config.autoencoder.loss.recon_loss_type == 'mse':
            return F.mse_loss(x_hat, x)
        else:
            raise ValueError('invalid reconstruction loss type')

    def forward(self, x):
        z = self.encode(x)
        z_q = self.quantize(z)
        x_hat = self.decode(z_q)
        return { 'z': z, 'z_q': z_q, 'x_hat': x_hat }

    def add_discriminator(self):
        pass

    def add_aux_loss(self):
        pass