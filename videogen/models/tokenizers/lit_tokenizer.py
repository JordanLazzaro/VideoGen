import torch
import pytorch_lightning as pl
from collections import OrderedDict
from typing import Dict
from base import Tokenizer
from config import Config

from utils import (
    pick_random_frames,
    pick_random_tubelets,
    adopt_weight,
    log_disc_patches,
    log_val_clips
)

from modules import LeCAM

class LitTokenizer(pl.LightningModule):
    def __init__(self, model: Tokenizer, config: Config):
        self.model = model
        self.config = config

    def configure_model(self):
        if self.config.compile:
            self.model = torch.compile(self.model)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)
    
    def training_step(self, batch):
        pass

    def validation_step(self, batch):
        pass

    def configure_optimizers(self):
        pass