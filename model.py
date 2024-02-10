import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels
        ):
        super().__init__()
        if in_channels != out_channels:
            self.identity = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            )
        else:
            self.identity = nn.Identity()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            ),
            nn.BatchNorm3d(out_channels)
        )
        self.res_act = nn.ReLU()

    def forward(self, x):
        out = self.block(x) + self.identity(x)
        return self.res_act(out)


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            nlayers,
            nblocks
        ):
        super().__init__()
        self.downsample_blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channels if i==0 else hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=(4, 4, 4),
                    stride=(2, 2, 2),
                    padding=(1, 1, 1)
                ),
                nn.BatchNorm3d(hidden_channels),
                nn.ReLU()
            ) for i in range(nlayers)
        ])

        self.res_blocks = nn.Sequential(*[
            ResBlock(
                in_channels=hidden_channels,
                out_channels=out_channels if i==nblocks-1 else hidden_channels
            ) for i in range(nblocks)
        ])

    def forward(self, x):
        x = self.downsample_blocks(x)
        x = self.res_blocks(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            nlayers,
            nblocks
        ):
        super().__init__()
        self.res_blocks = nn.Sequential(*[
            ResBlock(
                in_channels=in_channels if i==0 else hidden_channels,
                out_channels=hidden_channels,
            )
            for i in range(nblocks)
        ])

        self.upsample_blocks = nn.Sequential(*[
            nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=(4, 4, 4),
                    stride=(2, 2, 2),
                    padding=(1, 1, 1)),
                nn.BatchNorm3d(hidden_channels),
                nn.ReLU()
            ) for i in range(nlayers-1)
        ])

        self.out_layer = nn.ConvTranspose3d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=(4, 4, 4),
            stride=(2, 2, 2),
            padding=(1, 1, 1)
        )

        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.res_blocks(x)
        x = self.upsample_blocks(x)
        x = self.out_layer(x)
        x = self.out_act(x)
        return x


class QuantizerEMA(nn.Module):
    def __init__(
            self,
            codebook_size,
            latent_channels,
            ema_gamma,
            commit_loss_beta,
            track_codebook
        ):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_channels = latent_channels
        self.ema_gamma = ema_gamma
        self.commit_loss_beta = commit_loss_beta
        self.track_codebook = track_codebook

        self.codebook = nn.Embedding(codebook_size, latent_channels)
        nn.init.uniform_(self.codebook.weight, -1/codebook_size, 1/codebook_size)

        self.register_buffer('N', torch.zeros(codebook_size) + 1e-6)
        self.register_buffer('m', torch.zeros(codebook_size, latent_channels))
        nn.init.uniform_(self.m, -1/codebook_size, 1/codebook_size)

        if track_codebook:
            self.register_buffer('codebook_usage', torch.zeros(codebook_size, dtype=torch.float))
            self.register_buffer('total_usage', torch.tensor(0, dtype=torch.float))

    def ema_update(self, code_idxs, flat_inputs):
        # we don't want to track grads for ops in EMA calculation
        code_idxs, flat_inputs = code_idxs.detach(), flat_inputs.detach()

        # number of vectors for each i which quantize to e_i
        n = torch.bincount(code_idxs, minlength=self.codebook_size)

        # sum of vectors for each i which quantize to code e_i
        one_hot_indices = F.one_hot(code_idxs, num_classes=self.codebook_size).type(flat_inputs.dtype)
        embed_sums = one_hot_indices.T @ flat_inputs

        # update EMA of code usage and sum of codes
        self.N = self.N * self.ema_gamma + n * (1 - self.ema_gamma)
        self.m = self.m * self.ema_gamma + embed_sums * (1 - self.ema_gamma)

        self.codebook.weight.data.copy_(self.m / self.N.unsqueeze(-1))

    def reset_usage_stats(self):
        self.codebook_usage.zero_()
        self.total_usage.zero_()

    def calculate_perplexity(self, enc_idxs):
        unique_indices, counts = torch.unique(enc_idxs, return_counts=True)
        self.codebook_usage.index_add_(0, unique_indices, counts.float())
        self.total_usage += torch.sum(counts)

        if self.total_usage > 0:
            probs = self.codebook_usage / self.total_usage
            perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
            return perplexity
        else:
            return torch.tensor([0.0])

    def forward(self, inputs):
        B, C, T, H, W = inputs.shape

        # (B, C, T, H, W) --> (B, T, H, W, C)
        inputs_permuted = inputs.permute(0, 2, 3, 4, 1).contiguous()
        # (B, T, H, W, C) --> (BTHW, C)
        flat_inputs = inputs_permuted.reshape(-1, self.latent_channels)

        # (BTHW, Codebook Size)
        # Σ(x-y)^2 = Σx^2 - 2xy + Σy^2
        dists = (
            torch.sum(flat_inputs ** 2, dim=1, keepdim=True) - # Σx^2
            2 * (flat_inputs @ self.codebook.weight.t()) +     # 2*xy
            torch.sum(self.codebook.weight ** 2, dim=1)        # Σy^2
        )

        # (BTHW, 1)
        code_idxs = torch.argmin(dists, dim=1)
        # (BTHW, C)
        codes = self.codebook(code_idxs)
        # (BTHW, C) --> (B, T, H, W, C)
        quantized_inputs = codes.reshape(B, T, H, W, C)
        # (B, T, H, W, C) --> (B, C, T, H, W)
        quantized_inputs = quantized_inputs.permute(0, 4, 1, 2, 3).contiguous()

        if self.training:
            # perform exponential moving average update for codebook
            self.ema_update(code_idxs, flat_inputs)

        # "since the volume of the embedding space is dimensionless, it can grow
        # arbitrarily if the embeddings e_i do not train as fast as the encoder
        # parameters. To make sure the encoder commits to an embedding and its
        # output does not grow, we add a commitment loss"
        commitment_loss = F.mse_loss(quantized_inputs.detach(), inputs)

        # part 3 of full loss (ie. not including reconstruciton loss or embedding loss)
        vq_loss = commitment_loss * self.commit_loss_beta

        # sets the output to be the input plus the residual value between the
        # quantized latents and the inputs like a resnet for Straight Through
        # Estimation (STE)
        quantized_inputs = inputs + (quantized_inputs - inputs).detach()

        if self.track_codebook:
            perplexity = self.calculate_perplexity(code_idxs)

        return {
            'q_z':              quantized_inputs,
            'vq_loss':          vq_loss,
            'commitment_loss':  commitment_loss,
            'perplexity':       perplexity if self.track_codebook else torch.tensor([0.0])
        }


class VQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(
            in_channels     = config.in_channels,
            hidden_channels = config.hidden_channels,
            out_channels    = config.latent_channels,
            nlayers         = config.nlayers,
            nblocks         = config.nblocks
        )
        self.decoder = Decoder(
            in_channels     = config.latent_channels,
            hidden_channels = config.hidden_channels,
            out_channels    = config.in_channels,
            nlayers         = config.nlayers,
            nblocks         = config.nblocks
        )
        self.quantizer = QuantizerEMA(
            codebook_size    = config.codebook_size,
            latent_channels  = config.latent_channels,
            ema_gamma        = config.ema_gamma,
            commit_loss_beta = config.commit_loss_beta,
            track_codebook   = config.track_codebook,
        )

    def encode(self, inputs):
        return self.encoder(inputs)

    def quantize(self, z):
        return self.quantizer(z)

    def decode(self, q_z):
        return self.decoder(q_z)

    def loss(self, x_hat, x, quantized):
        MSE = F.mse_loss(x_hat, x)
        loss = MSE + quantized['vq_loss']

        return {
            'MSE':  MSE,
            'loss': loss,
            **quantized
        }

    def forward(self, x):
        z = self.encode(x)
        quantized = self.quantize(z)
        x_hat = self.decode(quantized['q_z'])
        losses = self.loss(x_hat, x, quantized)

        return {'x_hat': x_hat, **losses}