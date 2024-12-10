import torch
import imageio
import wandb
import numpy as np
import pytorch_lightning as pl
from typing import Dict

from videogen.config import Config
from videogen.models.autoencoders.discriminators.discriminator import Discriminator
from videogen.models.autoencoders.autoencoder import Autoencoder
from videogen.models.autoencoders.utils import adopt_weight, log_disc_patches, pick_random_frames, pick_random_tubelets


class LitAutoencoder(pl.LightningModule):
    def __init__(self, config: Config, autoencoder: Autoencoder, discriminator: Discriminator = None):
        super().__init__()
        self.automatic_optimization = False
        
        self.config = config
        self.autoencoder = autoencoder
        self.discriminator = discriminator
        self.losses = self.build_losses(config)

        self.autoencoder_lr = config.autoencoder.training.autoencoder_lr
        
        if discriminator is not None:
            self.disc_lr = config.autoencoder.training.disc_lr

    def configure_model(self) -> None:
        if self.config.compile:
            self.autoencoder = torch.compile(self.autoencoder)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.autoencoder(x)
    
    def training_step(self, batch):
        x = batch
        out = self(x)

        self.autoencoder_train_step(out, x)

        if self.discriminator is not None:
            self.discriminator_train_step(out, x)

    def validation_step(self, batch):
        x = batch
        out = self(x)

        loss = {}
        for loss_fn in self.losses.values():
            loss.update(loss_fn(out, x))
        
        for name, value in loss:
            self.log(f'val/{name}', value)

        # TODO: make logging outputs more generic

    def build_losses(self, config: Config):
        losses = {}
        losses.update({ 'reconstruction': ReconstructionLoss(
            weight=config.reconstruction_loss_weight,
            recon_loss_type=config.reconstruction_loss_type) 
        })
        if config.autoencoder.loss.kld is not None:
            losses.update({ 'kld': KLDLoss(weight=config.autoencoder.loss.kld.weight) })
        if config.autoencoder.loss.perceptual is not None:
            losses.update({ 'perceptual': PerceptualLoss(weight=config.autoencoder.loss.perceptual.weight)})
        if config.autoencoder.loss.adversarial is not None:
            losses.update({ 'adversarial': AdversarialLoss(
                weight=config.autoencoder.loss.adversarial.weight,
                discriminator=self.discriminator,
                regularization_loss=None, # TODO: add factory for this
                regularization_loss_weight=config.autoencoder.loss.adversarial.regularization_loss_weight,
                grad_penalty_weight=config.autoencoder.loss.adversarial.grad_penalty_weight
            )})
    
    def configure_optimizers(self):
        # TODO: make optimizer and lr scheduler configurable
        gen_optimizer = torch.optim.AdamW(
            self.autoencoder.parameters(),
            lr=self.tokenizer_lr,
            betas=self.config.autoencoder.training.betas,
            weight_decay=self.config.autoencoder.training.weight_decay
        )

        disc_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.disc_lr,
            betas=self.config.autoencoder.training.betas,
            weight_decay=self.config.autoencoder.training.weight_decay
        ) if self.discriminator is not None else None

        if self.config.autoencoder.training.use_lr_schedule:
            gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                gen_optimizer, T_max=self.config.autoencoder.training_steps)

            disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                disc_optimizer, T_max=self.config.autoencoder.training_steps) if disc_optimizer is not None else None
            
            if self.discriminator is not None:
                return (
                    { 'optimizer': gen_optimizer, 'lr_scheduler': gen_scheduler },
                    { 'optimizer': disc_optimizer, 'lr_scheduler': disc_scheduler }
                )
            else:
                return { 'optimizer': gen_optimizer, 'lr_scheduler': gen_scheduler }
        if self.discriminator is not None:
            return (
                { 'optimizer': gen_optimizer },
                { 'optimizer': disc_optimizer }
            )
        else:
            return { 'optimizer': gen_optimizer }
    
    def autoencoder_train_step(self, out, x) -> None:
        if self.discriminator is not None:
            opt_ae = self.optimizers()[0]
        else:
            opt_ae = self.optimizers()

        loss = {}
        for loss_fn in self.losses.values():
            loss.update(loss_fn(out, x))

        for name, value in loss:
            self.log(f'train/{name}', value)

        self.toggle_optimizer(opt_ae)
        self.manual_backward(total_loss)
        opt_ae.step()
        opt_ae.zero_grad()
        self.untoggle_optimizer(opt_ae)

    def discriminator_train_step(self, out, x) -> None:
        if self.global_step <= self.config.training.discriminator_delay: return
        
        opt_d = self.optimizers()[1]

        loss = self.losses['adversarial'](out, x, mode='discriminator')

        for name, value in loss:
            self.log(f'train/{name}', value)

        self.toggle_optimizer(opt_d)
        self.manual_backward(disc_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)