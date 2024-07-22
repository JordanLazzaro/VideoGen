import torch
import pytorch_lightning as pl
from collections import OrderedDict
from typing import Dict

from videogen.models.discriminators.discriminator import Discriminator
from videogen.models.tokenizers.tokenizer import Tokenizer
from config import Config

from utils import (
    pick_random_frames,
    pick_random_tubelets,
    adopt_weight,
    log_disc_patches,
    log_val_clips
)


class LitTokenizer(pl.LightningModule):
    def __init__(self, config: Config, tokenizer: Tokenizer, discriminator: Discriminator=None):
        super().__init__()
        self.automatic_optimization = False

        self.config = config
        self.tokenizer = tokenizer
        self.discriminator = discriminator

        self.tokenizer_lr = config.training.tokenizer_lr
        self.disc_lr = config.training.disc_lr

    def configure_model(self):
        if self.config.compile:
            self.tokenizer = torch.compile(self.tokenizer)
            if self.discriminator is not None:
                self.discriminator = torch.compile(self.discriminator)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.tokenizer(x)
    
    def training_step(self, batch):
        x = batch
        out = self(x)

        self.tokenizer_train_step(x, out)
        
        if self.discriminator is not None:
            self.discriminator_train_step(x, out)

    def validation_step(self, batch):
        x = batch
        out = self(x)

        rec_loss = self.tokenizer.reconstruction_loss(out['x_hat'], x)
        self.log('val/rec_loss', rec_loss, prog_bar=True)

        log_val_clips(x, out)

        return rec_loss

    def configure_optimizers(self):
        gen_optimizer = torch.optim.AdamW(
            self.tokenizer.parameters(),
            lr=self.tokenizer_lr,
            betas=self.config.training.betas,
            weight_decay=self.config.training.weight_decay
        )

        disc_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.disc_lr,
            betas=self.config.training.betas,
            weight_decay=self.config.training.weight_decay
        ) if self.discriminator is not None else None

        if self.config.use_lr_schedule:
            gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                gen_optimizer, T_max=self.config.training_steps)

            disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                disc_optimizer, T_max=self.config.training_steps) if disc_optimizer is not None else None

            return (
                { 'optimizer': gen_optimizer, 'lr_scheduler': gen_scheduler },
                { 'optimizer': disc_optimizer, 'lr_scheduler': disc_scheduler }
            )

        return (
            { 'optimizer': gen_optimizer },
            { 'optimizer': disc_optimizer }
        )
    
    def tokenizer_train_step(self, x, out):
        opt_g, _ = self.optimizers()

        rec_loss = self.config.recon_loss_weight * self.tokenizer.reconstruction_loss(out['x_hat'], x)
        self.log('train/rec_loss', rec_loss)

        if self.discriminator is not None:
            gen_loss_weight = adopt_weight(
                self.config.gen_loss_weight,
                self.current_epoch,
                threshold=self.config.discriminator.gen_loss_delay_epochs
            )

            if gen_loss_weight is not None:
                if self.config.discriminator.disc_type == 'tubelet':
                    real, fake = pick_random_tubelets(
                        x, out['x_hat'], num_tubelets=self.config.discriminator.num_tubelets)
                elif self.config.discriminator.disc_type == 'patch':
                    real, fake = pick_random_frames(
                        x, out['x_hat'], num_frames=self.config.discriminator.num_frames)
                else:
                    raise ValueError('invalid discriminator type')

                logits_fake = self.discriminator.discriminate(fake)
                generator_loss = self.discriminator.generator_loss(logits_fake)
                self.log('train/generator_loss', generator_loss)
                rec_loss = rec_loss + gen_loss_weight * generator_loss

        total_loss = rec_loss
        self.log('train/total_loss', total_loss)

        self.toggle_optimizer(opt_g)
        self.manual_backward(total_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

    def discriminator_train_step(self, x, out):
        _, opt_d = self.optimizers()
        
        disc_loss_weight = adopt_weight(
            self.config.disc_loss_weight,
            self.current_epoch,
            threshold=self.config.discriminator.disc_loss_delay_epochs
        )

        if disc_loss_weight is not None:
            if self.config.discriminator.disc_type == 'tubelet':
                real, fake = pick_random_tubelets(
                    x, out['x_hat'], num_tubelets=self.config.discriminator.num_tubelets)

            elif self.config.discriminator.disc_type == 'patch':
                real, fake = pick_random_frames(
                    x, out['x_hat'], num_frames=self.config.discriminator.num_frames)
                log_disc_patches(real, fake)
            else:
                raise ValueError('invalid discriminator type')

            if self.config.grad_penalty_weight is not None:
                real = real.requires_grad_()

            logits_real = self.discriminator.discriminate(real)
            logits_fake = self.discriminator.discriminate(fake.detach())

            self.log('train/real_logits', logits_real.mean())
            self.log('train/fake_logits', logits_fake.mean())

            disc_loss = disc_loss_weight * self.discriminator.discriminator_loss(logits_real, logits_fake)
            self.log('train/disc_loss', disc_loss)

            if self.config.grad_penalty_weight is not None:
                grad_penalty = self.discriminator.gradient_penalty(real, logits_real)
                self.log('train/grad_penalty', grad_penalty)
                disc_loss = disc_loss + self.config.grad_penalty_weight * grad_penalty
                self.log('train/disc_loss+grad_penalty', disc_loss)

            if self.config.reg_loss_weight is not None:
                reg_loss = self.discriminator.regularization_loss(logits_real, logits_fake)
                self.log('train/reg_loss', reg_loss)
                disc_loss = disc_loss + self.config.reg_loss_weight * reg_loss.detach()

            self.toggle_optimizer(opt_d)
            self.manual_backward(disc_loss)
            opt_d.step()
            opt_d.zero_grad()
            self.untoggle_optimizer(opt_d)