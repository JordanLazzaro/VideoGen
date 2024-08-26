#!/usr/bin/env python3

import argparse
import yaml
import wandb
from typing import Dict, Any
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl

from videogen.config import Config
from videogen.data.litdataset import LitDataModule
from videogen.models.tokenizers.discriminators.discriminator import Discriminator
from videogen.models.tokenizers.lit_tokenizer import LitTokenizer
from videogen.models.tokenizers.tokenizer import Tokenizer

def train(config: Dict[str, Any], kwargs: Dict[str, Any]):
    config = Config(config)

    tokenizer = Tokenizer.get_tokenizer(config)
    lit_tokenizer = LitTokenizer(tokenizer, config)
    
    if config.discriminator is not None:
        discriminator = Discriminator.get_discriminator(config)
        lit_tokenizer.add_discriminator(discriminator)

    data = LitDataModule(config)

    logger = None
    if kwargs['logging']:
        wandb.init(
            project=config.project.wandb_project,
            config=config,
            resume=kwargs['resume']
        )
        logger = WandbLogger(
            project=config.wandb_project,
            log_model=True
        )
        logger.watch(lit_tokenizer, log="all")

    callbacks = []
    if kwargs['monitor_lr']:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    if kwargs['save_checkpoint']:
        callbacks.append(ModelCheckpoint(
            dirpath        = config.tokenizer.training.checkpoint_dir,
            filename       = 'magvit2-',
            every_n_epochs = 5,
            save_top_k     = config.tokenizer.training.save_top_k,
            monitor        = 'val/rec_loss',
            mode           = 'min'
        ))
    if kwargs['early_stopping']:
        callbacks.append(EarlyStopping(
            monitor      = 'val/rec_loss',
            min_delta    = 0.000001,
            patience     = 100,
            verbose      = True, 
            check_finite = True
        ))

    trainer = pl.Trainer(
        max_epochs        = config.tokenizer.training.max_epochs,
        devices           = config.tokenizer.training.num_gpus,
        accelerator       = "gpu",
        precision         = config.tokenizer.training.precision,
        logger            = logger,
        callbacks         = callbacks,
        log_every_n_steps = 2
    )

    # trainer.fit(lit_tokenizer, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train your Tokenizers here!")
    parser.add_argument('--config', type=str, required=True, help="Path to configuration file")
    parser.add_argument('--logging', type=bool, default=True, help="Log to wandb")
    parser.add_argument('--monitor_lr', type=bool, default=True, help="Log Learning Rate to wandb")
    parser.add_argument('--save_checkpoint', type=bool, default=True, help="Save training state of lowest val loss")
    parser.add_argument('--early_stopping', type=bool, default=True, help="Stop training after too little progress")
    parser.add_argument('--resume', type=bool, default=False, help="Resume training from a previous checkpoint")
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train(config, vars(args))