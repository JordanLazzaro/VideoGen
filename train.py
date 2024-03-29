
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from .config import VideoVAEConfig
from .model import VQVAE, FSQVAE
from .litmodel import LitVQVAE, LitFSQVAE
from .data.litdatamodule import SteamboatWillieDataModule


def main():
    config = VideoVAEConfig()

    if config.quant_mode == 'fsq':
        model = FSQVAE(config)
        lit_model = LitFSQVAE(model, config)
    elif config.quant_mode == 'vq':
        model = VQVAE(config)
        lit_model = LitVQVAE(model, config)

    steamboat_willie_data = SteamboatWillieDataModule(config)

    wandb.init(project=config.project_name, config=config.to_dict())
    wandb_logger = WandbLogger(project=config.project_name, log_model=False)
    wandb_logger.watch(lit_model, log="all")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_path,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        every_n_epochs=5,
        save_top_k=config.save_top_k,
        monitor='val/loss',
        mode='min',
        save_last=True
    )

    # Define the EarlyStopping callback
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        min_delta=0.0000,
        patience=15,
        verbose=True,
        check_finite=True
    )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        devices=1,
        accelerator="gpu",
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[
            lr_monitor,
            early_stop_callback,
            # checkpoint_callback
        ],
        log_every_n_steps=1,
        # overfit_batches=1,
    )

    trainer.fit(lit_model, steamboat_willie_data)
    wandb.finish()

if __name__ == '__main__':
    main()