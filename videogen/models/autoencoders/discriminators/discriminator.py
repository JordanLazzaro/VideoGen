from abc import ABC, abstractmethod
from typing import Dict
from videogen.config import Config
from torch import nn
import torch


class Discriminator(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def get_discriminator(config: Config):
        if config.discriminator.name == 'patch':
            return None
        if config.discriminator.name == 'tubelet':
            return None