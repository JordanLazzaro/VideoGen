import wandb
import imageio
import torch
import torchvision
import numpy as np
import pytorch_lightning as pl
from einops import rearrange
from collections import OrderedDict

from modules import LeCAM


def adopt_weight(weight, epoch, threshold=0, value=None):
    '''
    https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/losses/vqperceptual.py#L14
    '''
    if epoch < threshold:
        weight = value
    return weight

def pick_random_patches(real_clips, fake_clips, num_patches=16, patch_size=(32, 32)):
    assert real_clips.shape == fake_clips.shape, 'real and fake videos must have the same shape'

    B, C, T, H, W = real_clips.shape
    ph, pw = patch_size

    assert H % ph == 0, 'frame height must be divisible by patch height'
    assert W % pw == 0, 'frame width be divisible by patch width'

    real_clips = rearrange(real_clips, 'b c t h w -> b t h w c')
    fake_clips = rearrange(fake_clips, 'b c t h w -> b t h w c')

    # real_patches = rearrange(real_clips, 'b t (h ph) (w pw) c -> b (t h w) c ph pw', ph=ph, pw=pw)
    # fake_patches = rearrange(fake_clips, 'b t (h ph) (w pw) c -> b (t h w) c ph pw', ph=ph, pw=pw)
    real_patches = rearrange(real_clips, 'b t (h ph) (w pw) c -> (b t h w) c ph pw', ph=ph, pw=pw)
    fake_patches = rearrange(fake_clips, 'b t (h ph) (w pw) c -> (b t h w) c ph pw', ph=ph, pw=pw)

    # rand_batch = torch.randperm(real_patches.shape[0])[0]
    # rand_idxs = torch.randperm(real_patches.shape[1])[:num_patches]
    rand_idxs = torch.randperm(real_patches.shape[0])[:num_patches]

    # real_patches = real_patches[rand_batch, rand_idxs, :, :, :]
    # fake_patches = fake_patches[rand_batch, rand_idxs, :, :, :]
    real_patches = real_patches[rand_idxs, :, :, :]
    fake_patches = fake_patches[rand_idxs, :, :, :]

    return real_patches, fake_patches


def pick_random_tubelets(real_clips, fake_clips, num_tubelets=16, tubelet_size=(4, 32, 32)):
    assert real_clips.shape == fake_clips.shape, 'real and fake videos must have the same shape'

    B, C, T, H, W = real_clips.shape
    tt, th, tw = tubelet_size

    assert T % tt == 0, 'clip time length must be divisible by tubelet time length'
    assert H % th == 0, 'frame height must be divisible by tubelet height'
    assert W % tw == 0, 'frame width be divisible by tubelet width'

    real_clips = rearrange(real_clips, 'b c t h w -> b t h w c')
    fake_clips = rearrange(fake_clips, 'b c t h w -> b t h w c')

    real_tubelets = rearrange(real_clips, 'b (t tt) (h th) (w tw) c -> b (t h w) c tt th tw', tt=tt, th=th, tw=tw)
    fake_tubelets = rearrange(fake_clips, 'b (t tt) (h th) (w tw) c -> b (t h w) c tt th tw', tt=tt, th=th, tw=tw)

    rand_batch = torch.randperm(real_tubelets.shape[0])[0]
    rand_idxs = torch.randperm(real_tubelets.shape[1])[:num_tubelets]

    real_tubelets = real_tubelets[rand_batch, rand_idxs, :, :, :]
    fake_tubelets = fake_tubelets[rand_batch, rand_idxs, :, :, :]

    return real_tubelets, fake_tubelets

def pick_random_frames(real_clips, fake_clips, num_frames=16):
    assert real_clips.shape == fake_clips.shape, 'real and fake videos must have the same shape'

    B, C, T, H, W = real_clips.shape

    real_clips = rearrange(real_clips, 'b c t h w -> b t h w c')
    fake_clips = rearrange(fake_clips, 'b c t h w -> b t h w c')

    rand_batch = torch.randperm(real_clips.shape[0])[0]
    rand_idxs = torch.randperm(real_clips.shape[1])[:num_frames]

    real_frames = real_clips[rand_batch, rand_idxs, :, :, :]
    fake_frames = fake_clips[rand_batch, rand_idxs, :, :, :]

    real_frames = rearrange(real_frames, 't h w c -> t c h w')
    fake_frames = rearrange(fake_frames, 't h w c -> t c h w')

    return real_frames, fake_frames


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
                # real, fake = pick_random_patches(
                #     x, out['x_hat'], num_patches=self.config.discriminator.num_patches)
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
        # out = {"x_hat": torch.rand_like(x)}
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
                # real, fake = pick_random_patches(
                #     x, out['x_hat'], num_patches=self.config.discriminator.num_patches)
                real, fake = pick_random_frames(
                    x, out['x_hat'], num_frames=self.config.discriminator.num_frames)
            else:
                raise ValueError('invalid discriminator type')

            self.log_disc_patches(real, fake)

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

        self.log_val_clips(x, out)

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

    def log_disc_patches(self, real_patches, fake_patches):
        n_images = min(real_patches.size(0), 8)
        comparison = torch.cat([real_patches[:n_images], fake_patches[:n_images]])
        grid = torchvision.utils.make_grid(comparison)
        self.logger.experiment.log({"train/disc_patches": [wandb.Image(grid, caption="Top: Real Patches, Bottom: Fake Patches")]})