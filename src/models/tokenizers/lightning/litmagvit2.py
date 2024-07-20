import torch
import pytorch_lightning as pl
from collections import OrderedDict

from utils import (
    pick_random_frames,
    pick_random_tubelets,
    adopt_weight,
    log_disc_patches,
    log_val_clips
)

from modules import LeCAM


class LitMAGVIT2(pl.LightningModule):
    def __init__(self, magvit2, config):
        super().__init__()
        self.automatic_optimization = False

        self.magvit2 = magvit2

        if config.lecam_loss_weight is not None:
            self.lecam = LeCAM(config)

        self.config = config
        self.fsq_vae_lr = config.training.fsq_vae_lr
        self.disc_lr = config.training.disc_lr

    def configure_model(self):
        if self.config.compile:
            self.magvit2 = torch.compile(self.magvit2)

    def forward(self, x):
        return self.magvit2(x)

    def training_step(self, batch):
        opt_g, opt_d = self.optimizers()

        x = batch
        out = self(x)

        # train generator/FSQ-VAE
        rec_loss = self.config.recon_loss_weight * self.magvit2.reconstruction_loss(out['x_hat'], x)
        self.log('train/rec_loss', rec_loss)

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

            logits_fake = self.magvit2.discriminate(fake)
            generator_loss = self.magvit2.generator_loss(logits_fake)
            self.log('train/generator_loss', generator_loss)
            rec_loss = rec_loss + gen_loss_weight * generator_loss

        total_loss = rec_loss
        self.log('train/total_loss', total_loss)

        self.toggle_optimizer(opt_g)
        self.manual_backward(total_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # train discriminator
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
            else:
                raise ValueError('invalid discriminator type')

            log_disc_patches(real, fake)

            if self.config.grad_penalty_weight is not None:
                real = real.requires_grad_()

            logits_real = self.magvit2.discriminate(real)
            logits_fake = self.magvit2.discriminate(fake.detach())

            self.log('train/real_logits', logits_real.mean())
            self.log('train/fake_logits', logits_fake.mean())

            disc_loss = disc_loss_weight * self.magvit2.discriminator_loss(logits_real, logits_fake)
            self.log('train/disc_loss', disc_loss)

            if self.config.grad_penalty_weight is not None:
                grad_penalty = self.magvit2.gradient_penalty(real, logits_real)
                self.log('train/grad_penalty', grad_penalty)
                disc_loss = disc_loss + self.config.grad_penalty_weight * grad_penalty
                self.log('train/disc_loss+grad_penalty', disc_loss)

            if self.config.lecam_loss_weight is not None:
                self.lecam.update(logits_real, logits_fake)
                lecam_loss = self.lecam(logits_real, logits_fake)
                self.log('train/lecam_loss', lecam_loss)
                disc_loss = disc_loss + self.config.lecam_loss_weight * lecam_loss.detach()

            self.toggle_optimizer(opt_d)
            self.manual_backward(disc_loss)
            torch.nn.utils.clip_grad_norm_(self.magvit2.discriminator.parameters(), max_norm=1.0)
            opt_d.step()
            opt_d.zero_grad()
            self.untoggle_optimizer(opt_d)

    def validation_step(self, batch):
        x = batch
        out = self(x)

        rec_loss = self.magvit2.reconstruction_loss(out['x_hat'], x)
        self.log('val/rec_loss', rec_loss, prog_bar=True)

        log_val_clips(x, out)

        return rec_loss

    def configure_optimizers(self):
        fsqvae_optimizer = torch.optim.AdamW(
            self.magvit2.fsqvae.parameters(),
            lr=self.fsq_vae_lr,
            betas=self.config.training.betas,
            weight_decay=self.config.training.weight_decay
        )

        disc_optimizer = torch.optim.AdamW(
            self.magvit2.discriminator.parameters(),
            lr=self.disc_lr,
            betas=self.config.training.betas,
            weight_decay=self.config.training.weight_decay
        )

        if self.config.use_lr_schedule:
            fsqvae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                fsqvae_optimizer, T_max=self.config.training_steps)

            disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                disc_optimizer, T_max=self.config.training_steps)

            return (
                { 'optimizer': fsqvae_optimizer, 'lr_scheduler': fsqvae_scheduler },
                { 'optimizer': disc_optimizer, 'lr_scheduler': disc_scheduler }
            )

        return (
            { 'optimizer': fsqvae_optimizer },
            { 'optimizer': disc_optimizer }
        )

    def load_partial_weights(self, checkpoint_path, load_fsqvae=True, load_discriminator=True):
        # TODO: ensure other training checkpoint info is preserved
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_state_dict = checkpoint['state_dict']
        if load_fsqvae:
            fsqvae_dict = OrderedDict((k.replace('magvit2.fsqvae.', ''), v)
                                      for (k, v) in model_state_dict.items()
                                      if k.startswith('magvit2.fsqvae.'))
            self.magvit2.fsqvae.load_state_dict(fsqvae_dict)
            print("FSQVAE loaded from checkpoint.")

        if load_discriminator:
            disc_dict = OrderedDict((k.replace('magvit2.discriminator.', ''), v)
                                    for (k, v) in model_state_dict.items()
                                    if k.startswith('magvit2.discriminator.'))
            self.magvit2.discriminator.load_state_dict(disc_dict)
            print("Discriminator loaded from checkpoint.")

    def on_fit_start(self):
        if self.config.checkpoint_path:
            self.load_partial_weights(
                self.config.checkpoint_path,
                load_fsqvae=self.config.load_fsqvae,
                load_discriminator=self.config.load_discriminator
            )