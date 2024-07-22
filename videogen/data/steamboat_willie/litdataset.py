import torch
import pytorch_lightning as pl
from videogen.data.steamboat_willie.dataset import SteamboatWillieDataset


class SteamboatWillieDataModule(pl.LightningDataModule):
    def __init__(self, config, batch_size, new_img_size=None):
        super().__init__()
        self.batch_size = batch_size
        self.config = config
        self.new_img_size = new_img_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = SteamboatWillieDataset(self.config, mode='train', new_img_size=self.new_img_size)
            self.val_dataset = SteamboatWillieDataset(self.config, mode='val', new_img_size=self.new_img_size)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.config.num_workers)