#!/usr/bin/env python3

import argparse
import yaml
import wandb
from typing import Dict, Any
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl

from videogamegen.config import Config
from videogamegen.data.litdataset import LitDataModule
from videogamegen.models.tokenizers.discriminators.discriminator import Discriminator
from videogamegen.models.tokenizers.lit_tokenizer import LitTokenizer
from videogamegen.models.tokenizers.tokenizer import Tokenizer


def train(model_config: Dict[str, Any], data_config: Dict[str, Any], kwargs: Dict[str, Any]):
    model_config = Config(model_config)
    data_config = Config(data_config)

    tokenizer = Tokenizer.get_tokenizer(model_config)
    lit_tokenizer = LitTokenizer(tokenizer, model_config)
    
    if model_config.discriminator is not None:
        discriminator = Discriminator.get_discriminator(model_config)
        lit_tokenizer.add_discriminator(discriminator)

    data = LitDataModule(model_config, data_config)

    logger = None
    if kwargs['logging']:
        wandb.init(
            project=model_config.project.wandb_project,
            config=model_config,
            resume=kwargs['resume']
        )
        logger = WandbLogger(
            project=model_config.project.wandb_project,
            log_model=True
        )
        logger.watch(lit_tokenizer, log="all")

    callbacks = []
    if kwargs['monitor_lr']:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    if kwargs['save_checkpoint']:
        callbacks.append(
            ModelCheckpoint(
                dirpath        = model_config.tokenizer.training.checkpoint_dir,
                filename       = 'vanilla-fsq-vae-',
                every_n_epochs = 5,
                save_top_k     = model_config.tokenizer.training.save_top_k,
                monitor        = 'val/rec_loss',
                mode           = 'min'
            )
        )
    if kwargs['early_stopping']:
        callbacks.append(
            EarlyStopping(
                monitor      = 'val/rec_loss',
                min_delta    = 0.000001,
                patience     = 100,
                verbose      = True, 
                check_finite = True
            )
        )

    trainer = pl.Trainer(
        max_epochs        = model_config.tokenizer.training.max_epochs,
        devices           = model_config.tokenizer.training.num_gpus,
        accelerator       = "gpu",
        precision         = model_config.tokenizer.training.precision,
        logger            = logger,
        callbacks         = callbacks,
        log_every_n_steps = 2
    )

    trainer.fit(lit_tokenizer, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train your Tokenizers here!")
    parser.add_argument('--model-config', type=str, required=True, help="Path to model config file")
    parser.add_argument('--data-config', type=str, required=True, help="Path to data config file")
    parser.add_argument('--logging', type=bool, default=True, help="Log to wandb")
    parser.add_argument('--monitor_lr', type=bool, default=True, help="Monitor Learning Rate (on wandb)")
    parser.add_argument('--save_checkpoint', type=bool, default=True, help="Save training state of lowest val loss")
    parser.add_argument('--early_stopping', type=bool, default=True, help="Stop training after too little progress")
    parser.add_argument('--resume', type=bool, default=False, help="Resume training from a previous checkpoint")
    
    args = parser.parse_args()

    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)

    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f)

    train(model_config, data_config, vars(args))