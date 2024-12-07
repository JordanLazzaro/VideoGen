from abc import ABC, abstractmethod
from typing import Dict
from videogen.config import Config
from torch import nn
import torch


class Autoencoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def reconstruction_loss(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pass # TODO: take in dict

    @staticmethod
    def get_autoencoder(config: Config):
        from videogen.models.autoencoders.models.magvit2 import MAGVIT2
        from videogen.models.autoencoders.models.fsq_vae import FSQVAE
        
        if config.tokenizer.name == 'vanilla-fsq-vae':
            return FSQVAE(config)
        if config.tokenizer.name == 'magvit2':
            return MAGVIT2(config)