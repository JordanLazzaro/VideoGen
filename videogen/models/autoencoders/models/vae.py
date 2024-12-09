import torch
from torch import nn
from typing import Dict
from videogen.config import Config
from videogen import Autoencoder
#from videogen.models.modules import ()

class VAE(Autoencoder):
    def __init__(self, config: Config):
        super().__init__()
        self.config
        self.discriminator = None
        self.aux_losses = []

        # TODO: add encoder and decoder
        # TODO: add code for managing auxiliary losses

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def kld(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]), dim=0)

    def loss(self, output: Dict) -> Dict:
        # TODO: take in dict
        if self.config.autoencoder.loss.recon_loss_type == 'mae':
            recon_loss = F.l1_loss(x_hat, x)
        elif self.config.autoencoder.loss.recon_loss_type == 'mse':
            recon_loss = F.mse_loss(x_hat, x)
        else:
            raise ValueError('invalid reconstruction loss type')

        kld_loss = self.kld(output['mu'], output['logvar'])
        loss = recon_loss + self.config.kld_weight * MKLD
        
        return { 'total_loss': loss, 'recon_loss': recon_loss, 'kld_loss': kld_loss }

    def forward(self, x) -> Dict:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return { 'mu': mu, 'logvar': logvar, 'z': z, 'x_hat': x_hat }