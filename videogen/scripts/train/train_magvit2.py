import yaml
import argparse
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from config import Config
from videogen.models.tokenizers.magvit2 import MAGVIT2
from videogen.models.lightning.magvit2 import LitMAGVIT2
from videogen.data.steamboat_willie.litdataset import SteamboatWillieDataModule


resume_training = False

config = yaml.safe_load('../../../config/magvit2_config.yaml')
project_config = Config(config['project'])
magvit2_config = Config(config['magvit2'])
data_config    = Config(config['data'])

model     = MAGVIT2(magvit2_config)
lit_model = LitMAGVIT2(model, magvit2_config)

data = SteamboatWillieDataModule(data_config, magvit2_config.training.batch_size, (128, 128))

if resume_training:
    run = wandb.init(
        project = project_config.magvit2.wandb_project,
        config  = config['magvit2'], resume=True
    )
    # artifact = run.use_artifact('[ARTIFACT NAME]', type='model')
    # artifact_dir = artifact.download()
else:
    run = wandb.init(
        project = project_config.magvit2.wandb_project,
        config  = config['magvit2']
    )

wandb_logger = WandbLogger(
    project=project_config.magvit2.project_name,
    log_model=True
)
wandb_logger.watch(lit_model, log="all")

lr_monitor = LearningRateMonitor(logging_interval='step')

checkpoint_callback = ModelCheckpoint(
    dirpath        = magvit2_config.checkpoint_dir,
    filename       = f'magvit2-{run.name}',
    every_n_epochs = 5,
    save_top_k     = magvit2_config.training.save_top_k,
    monitor        = 'val/rec_loss',
    mode           = 'min'
)

early_stop_callback = EarlyStopping(
    monitor      = 'val/rec_loss',
    min_delta    = 0.000001,
    patience     = 100,
    verbose      = True,
    check_finite = True
)

trainer = pl.Trainer(
    max_epochs  = magvit2_config.training.max_epochs,
    devices     = 1,
    accelerator = "gpu",
    precision   = "16-mixed",
    logger      = wandb_logger,
    callbacks=[
        lr_monitor,
        early_stop_callback,
        checkpoint_callback
    ],
    log_every_n_steps = 2
)

# tuner = Tuner(trainer)
# lr_result = tuner.lr_find(lit_model, attr_name='fsq_vae_lr', datamodule=data, max_lr=1)
# lr_result.plot(show=True, suggest=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument('--resume', action='store_true', help='A flag for resuming training from checkpoint')
    args = parser.parse_args()
    
    trainer.fit(lit_model, data)
    wandb.finish()