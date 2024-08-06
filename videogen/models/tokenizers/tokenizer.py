from abc import ABC, abstractmethod
from typing import Dict
from videogen.config import Config
from torch import nn
import torch

from videogen.models.tokenizers.models.magvit2 import MAGVIT2
from videogen.models.tokenizers.models.fsq_vae import FSQVAE


class Tokenizer(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def reconstruction_loss(self, x_hat: torch.Tensor, x: torch.Tensor):
        pass

    @staticmethod
    def get_tokenizer(config: Config):
        if config.tokenizer.name == 'vanilla-fsq-vae':
            return FSQVAE(config)
        if config.tokenizer.name == 'magvit2':
            return MAGVIT2(config)