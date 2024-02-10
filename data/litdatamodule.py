import torch
import pytorch_lightning as pl
from torch.utils.data import random_split

from .dataset import SteamboatWillieDataset


class SteamboatWillieDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        self.config = config

    def prepare_data(self):
        self.full_dataset = SteamboatWillieDataset(self.config)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_len = int(len(self.full_dataset) * self.config.train_split)
            val_len = len(self.full_dataset) - train_len
            self.train_dataset, self.val_dataset = random_split(self.full_dataset, [train_len, val_len])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.config.num_workers)