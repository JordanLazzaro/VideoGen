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
    def __init__(self, autoencoder: Autoencoder, config: Config):
        super().__init__()
        self.automatic_optimization = False
        
        self.config = config
        self.autoencoder = autoencoder
        self.discriminator = None
        self.aux_losses = []

        self.autoencoder_lr = config.autoencoder.training.autoencoder_lr
        self.disc_lr = config.autoencoder.training.disc_lr

    def add_discriminator(self, discriminator: Discriminator) -> None:
        if self.config.compile:
            self.discriminator = torch.compile(discriminator)
        else:
            self.discriminator = discriminator

    def add_aux_loss(self, loss: nn.Module) -> None:
        self.aux_losses.append(loss)

    def configure_model(self) -> None:
        if self.config.compile:
            self.autoencoder = torch.compile(self.autoencoder)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.autoencoder(x)
    
    def training_step(self, batch):
        x = batch
        out = self(x)

        self.autoencoder_train_step(x, out)
        
        if self.discriminator is not None:
            self.discriminator_train_step(x, out)

    def validation_step(self, batch):
        x = batch
        out = self(x)

        # rec_loss = self.autoencoder.reconstruction_loss(out['x_hat'], x)
        rec_loss = self.autoencoder.loss(out)
        self.log('val/rec_loss', rec_loss, prog_bar=True)

        self.log_val_clips(x, out)

        return rec_loss

    def configure_optimizers(self):
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
    
    def autoencoder_train_step(self, x, out) -> None:
        if self.discriminator is not None:
            opt_g = self.optimizers()[0]
        else:
            opt_g = self.optimizers()

        loss = self.autoencoder.loss(out) # TODO: take in dict + change to '.loss(...)'
        self.log('train/rec_loss', rec_loss)

        if self.config.autoencoder.loss.recon_loss_weight is not None:
            rec_loss = rec_loss * self.config.autoencoder.loss.recon_loss_weight

        total_loss = rec_loss
        for aux_loss in self.aux_losses:
            total_loss += aux_loss(out, x) # TODO: modify to account for aux_loss weight in config
        
        if self.discriminator is not None:
            gen_loss_weight = adopt_weight(
                self.config.autoencoder.discriminator.loss.gen_loss_weight,
                self.current_epoch,
                threshold=self.config.autoencoder.discriminator.loss.gen_loss_delay_epochs
            )

            if gen_loss_weight is not None:
                if self.config.autoencoder.discriminator.disc_type == 'tubelet':
                    real, fake = pick_random_tubelets(
                        x, out['x_hat'], num_tubelets=self.config.discriminator.num_tubelets)
                elif self.config.autoencoder.discriminator.disc_type == 'patch':
                    real, fake = pick_random_frames(
                        x, out['x_hat'], num_frames=self.config.autoencoder.discriminator.num_frames)
                else:
                    raise ValueError('invalid discriminator type')

                logits_fake = self.discriminator.discriminate(fake)
                generator_loss = self.discriminator.generator_loss(logits_fake)
                self.log('train/generator_loss', generator_loss)
                total_loss = total_loss + gen_loss_weight * generator_loss

        self.log('train/total_loss', total_loss)

        self.toggle_optimizer(opt_g)
        self.manual_backward(total_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

    def discriminator_train_step(self, x, out) -> None:
        opt_d = self.optimizers()[1]
        
        disc_loss_weight = adopt_weight(
            self.config.autoencoder.discriminator.loss.disc_loss_weight,
            self.current_epoch,
            threshold=self.config.autoencoder.discriminator.loss.disc_loss_delay_epochs
        )

        if disc_loss_weight is not None:
            if self.config.autoencoder.discriminator.disc_type == 'tubelet':
                real, fake = pick_random_tubelets(
                    x, out['x_hat'], num_tubelets=self.config.discriminator.num_tubelets)

            elif self.config.autoencoder.discriminator.disc_type == 'patch':
                real, fake = pick_random_frames(
                    x, out['x_hat'], num_frames=self.config.autoencoder.discriminator.num_frames)
                log_disc_patches(real, fake)
            else:
                raise ValueError('invalid discriminator type')

            if self.config.autoencoder.discriminator.grad_penalty_weight is not None:
                real = real.requires_grad_()

            logits_real = self.discriminator.discriminate(real)
            logits_fake = self.discriminator.discriminate(fake.detach())

            self.log('train/real_logits', logits_real.mean())
            self.log('train/fake_logits', logits_fake.mean())

            disc_loss = disc_loss_weight * self.discriminator.discriminator_loss(logits_real, logits_fake)
            self.log('train/disc_loss', disc_loss)

            if self.config.autoencoder.discriminator.grad_penalty_weight is not None:
                grad_penalty = self.discriminator.gradient_penalty(real, logits_real)
                self.log('train/grad_penalty', grad_penalty)
                disc_loss = disc_loss + self.config.discriminator.loss.grad_penalty_weight * grad_penalty
                self.log('train/disc_loss+grad_penalty', disc_loss)

            if self.config.autoencoder.discriminator.reg_loss_weight is not None:
                reg_loss = self.discriminator.regularization_loss(logits_real, logits_fake)
                self.log('train/reg_loss', reg_loss)
                disc_loss = disc_loss + self.config.discriminator.loss.reg_loss_weight * reg_loss.detach()

            self.toggle_optimizer(opt_d)
            self.manual_backward(disc_loss)
            opt_d.step()
            opt_d.zero_grad()
            self.untoggle_optimizer(opt_d)

    def log_val_clips(self, x, out, num_clips=2):
        n_clips = min(x.size(0), num_clips)

        for i in range(n_clips):
            # extract the ith original and reconstructed clip
            original_clip = x[i]  # (C, T, H, W)
            reconstructed_clip = out['x_hat'][i]  # (C, T, H, W)

            # convert tensors to numpy arrays and transpose to (T, H, W, C) for GIF creation
            original_clip_np = original_clip.permute(1, 2, 3, 0).cpu().numpy()
            reconstructed_clip_np = reconstructed_clip.permute(1, 2, 3, 0).cpu().numpy()

            original_clip_np = (original_clip_np - original_clip_np.min()) / (original_clip_np.max() - original_clip_np.min())
            reconstructed_clip_np = (reconstructed_clip_np - reconstructed_clip_np.min()) / (reconstructed_clip_np.max() - reconstructed_clip_np.min())

            original_clip_np = (original_clip_np * 255).astype(np.uint8)
            reconstructed_clip_np = (reconstructed_clip_np * 255).astype(np.uint8)

            # grayscale videos need to be of shape (T, H, W)
            if original_clip_np.shape[-1] == 1:
                original_clip_np = original_clip_np.squeeze(-1)

            if reconstructed_clip_np.shape[-1] == 1:
                reconstructed_clip_np = reconstructed_clip_np.squeeze(-1)

            # create GIFs for the original and reconstructed clips
            original_gif_path = f'/tmp/original_clip_{i}.gif'
            reconstructed_gif_path = f'/tmp/reconstructed_clip_{i}.gif'
            imageio.mimsave(original_gif_path, original_clip_np, fps=10)
            imageio.mimsave(reconstructed_gif_path, reconstructed_clip_np, fps=10)

            # log the GIFs to wandb
            self.logger.experiment.log({
                f"val/original_clip_{i}": wandb.Video(original_gif_path, fps=10, format="gif", caption="Original"),
                f"val/reconstructed_clip_{i}": wandb.Video(reconstructed_gif_path, fps=10, format="gif", caption="Reconstructed")
            })