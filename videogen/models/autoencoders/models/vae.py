import torch
from torch import nn
from typing import Dict
from videogen.config import Config
from videogen import Autoencoder
from videogen.models.modules import Encoder2d, Decoder2d

class VAE(Autoencoder):
    def __init__(self, config: Config):
        super().__init__()
        self.config
        self.discriminator = None
        self.aux_losses = []

        # TODO: add encoder and decoder
        self.encoder = Encoder2d()
        # TODO: add code for managing auxiliary losses
        self.decoder = Decoder2d()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x) -> Dict:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return { 'mu': mu, 'logvar': logvar, 'z': z, 'x_hat': x_hat }