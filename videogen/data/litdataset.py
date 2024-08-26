import torch
import pytorch_lightning as pl
from videogen.data.utils import get_dataset


class LitDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = get_dataset(self.config, mode='train')
            self.val_dataset = get_dataset(self.config, mode='val')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.tokenizer.training.batch_size,
            shuffle=True,
            num_workers=self.config.tokenizer.training.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.tokenizer.training.batch_size,
            shuffle=False,
            num_workers=self.config.tokenizer.training.num_workers
        )