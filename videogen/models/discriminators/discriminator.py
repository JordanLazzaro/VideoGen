from abc import ABC, abstractmethod
from typing import Dict
from config import Config
from torch import nn
import torch

class Discriminator(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def discriminator_loss(self, logits_real: torch.Tensor, logits_fake: torch.Tensor):
        pass

    @abstractmethod
    def generator_loss(self, logits_fake: torch.Tensor):
        pass

    @abstractmethod
    def regularization_loss(self, logits_real: torch.Tensor, logits_fake: torch.Tensor):
        pass

    @abstractmethod
    def gradient_penalty(self, x: torch.Tensor, logits_real: torch.Tensor):
        pass

    @staticmethod
    def get_discriminator(config: Config):
        if config.discriminator_name == 'patch':
            return None
        if config.discriminator_name == 'tubelet':
            return None