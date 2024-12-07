import torch
from torch import nn
from videogen.config import Config
from videogen import Autoencoder
#from videogen.models.modules import ()

class VAE(Autoencoder):
    def __init__(self, config: Config):
        super().__init__()
        self.config

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss(self, x_hat, x, mu, logvar):
        # TODO: take in dict
        if self.config.autoencoder.loss.recon_loss_type == 'mae':
            recon_loss = F.l1_loss(x_hat, x)
        elif self.config.autoencoder.loss.recon_loss_type == 'mse':
            recon_loss = F.mse_loss(x_hat, x)
        else:
            raise ValueError('invalid reconstruction loss type')

        MKLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]), dim=0)
        loss = recon_loss + self.config.kld_weight * MKLD
        
        return loss, recon_loss, MKLD

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar