import torch
import wandb
import imageio
import numpy as np
import pytorch_lightning as pl


class LitVQVAE(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.lr = config.lr

        if self.logger:
            self.logger.experiment.config.update(config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch
        out = self(x)

        self.log('train/loss',            out['loss'],            prog_bar=True)
        self.log('train/MSE',             out['MSE'],             prog_bar=True)
        self.log('train/vq_loss',         out['vq_loss'],         prog_bar=True)
        self.log('train/commitment_loss', out['commitment_loss'], prog_bar=True)
        self.log('train/perplexity',      out['perplexity'],      prog_bar=True)

        return out['loss']

    def validation_step(self, batch, batch_idx):
        x = batch
        out = self(x)

        self.log('val/loss',            out['loss'],            prog_bar=True)
        self.log('val/MSE',             out['MSE'],             prog_bar=True)
        self.log('val/vq_loss',         out['vq_loss'],         prog_bar=True)
        self.log('val/commitment_loss', out['commitment_loss'], prog_bar=True)

        if batch_idx == 0:
            self.log_val_clips(x, out)

        return out['loss']

    def on_epoch_end(self):
        # tracking perplexity per epoch
        self.model.quantizer.reset_usage_stats()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )

        if self.config.use_lr_schedule:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
            return [optimizer], [scheduler]

        return optimizer

    def log_val_clips(self, x, out, num_clips=2):
        n_clips = min(x.size(0), num_clips)

        for i in range(n_clips):
            # Extract the ith original and reconstructed clip
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
            imageio.mimsave(original_gif_path, original_clip_np, fps=5)
            imageio.mimsave(reconstructed_gif_path, reconstructed_clip_np, fps=5)

            # log the GIFs to wandb
            self.logger.experiment.log({
                f"val/original_clip_{i}": wandb.Video(original_gif_path, fps=5, format="gif", caption="Original"),
                f"val/reconstructed_clip_{i}": wandb.Video(reconstructed_gif_path, fps=5, format="gif", caption="Reconstructed")
            })

class LitFSQVAE(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.lr = config.lr

        if self.logger:
            self.logger.experiment.config.update(config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch
        out = self(x)

        self.log('train/loss', out['loss'], prog_bar=True)

        return out['loss']

    def validation_step(self, batch, batch_idx):
        x = batch
        out = self(x)

        self.log('val/loss', out['loss'], prog_bar=True)

        if batch_idx == 0:
            self.log_val_clips(x, out)

        return out['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )

        if self.config.use_lr_schedule:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
            return [optimizer], [scheduler]

        return optimizer

    def log_val_clips(self, x, out, num_clips=2):
        n_clips = min(x.size(0), num_clips)

        for i in range(n_clips):
            # Extract the ith original and reconstructed clip
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