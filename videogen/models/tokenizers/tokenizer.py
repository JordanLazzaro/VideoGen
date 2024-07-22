from abc import ABC, abstractmethod
from typing import Dict
from torch import nn
import torch


class Tokenizer(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

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