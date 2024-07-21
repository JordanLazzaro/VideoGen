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