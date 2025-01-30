import torch
import pytorch_lightning as pl
import webdataset as wds
from videogamegen.data.utils import process_npz


class LitDataModule(pl.LightningDataModule):
    def __init__(self, model_config: Config, data_config: Config):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.dataset = (
                wds.WebDataset(shard_paths)
                .decode()
                .to_tuple("npz")
                .map(process_npz)
        )

    def train_dataloader(self):
        return wds.WebLoader(
            self.train_dataset,
            batch_size=self.model_config.autoencoder.training.batch_size,
            shuffle=True,
            num_workers=self.model_config.autoencoder.training.num_workers
        )

    def val_dataloader(self):
        return wds.WebLoader(
            self.val_dataset,
            batch_size=self.model_config.autoencoder.training.batch_size,
            shuffle=False,
            num_workers=self.model_config.autoencoder.training.num_workers
        )