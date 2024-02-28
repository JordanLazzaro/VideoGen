import torch
import pytorch_lightning as pl
from torch.utils.data import random_split

from .dataset import SteamboatWillieDataset


class SteamboatWillieDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        self.config = config

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = SteamboatWillieDataset(self.config, mode='train')
            self.val_dataset = SteamboatWillieDataset(self.config, mode='val')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.config.num_workers)