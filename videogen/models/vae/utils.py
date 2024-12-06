import wandb
import imageio
import torch
import torchvision
import numpy as np
from einops import rearrange


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

def log_disc_patches(self, real_patches, fake_patches):
    n_images = min(real_patches.size(0), 8)
    comparison = torch.cat([real_patches[:n_images], fake_patches[:n_images]])
    grid = torchvision.utils.make_grid(comparison)
    self.logger.experiment.log({"train/disc_patches": [wandb.Image(grid, caption="Top: Real Patches, Bottom: Fake Patches")]})