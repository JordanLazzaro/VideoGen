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

        hidden_channels = in_channels // 4

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=hidden_channels,
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
                    kernel_size=(3, 4, 4),
                    stride=(2, 2, 2) if i%2==0 else (1, 2, 2),
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
                   kernel_size=(3, 4, 4),
                    stride=(1, 2, 2) if i%2==0 else (2, 2, 2),
                    padding=(1, 1, 1),
                    output_padding=(0, 0, 0) if i%2==0 else (1, 0, 0)
                    ),
                nn.BatchNorm3d(hidden_channels),
                nn.ReLU()
            ) for i in range(nlayers-1)
        ])

        self.out_layer = nn.ConvTranspose3d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=(3, 4, 4),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
            output_padding=(1, 0, 0)
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
    

class FSQ(nn.Module):
    def __init__(self, levels, eps=1e-3):
        super().__init__()
        self.register_buffer('levels', torch.tensor(levels))
        self.register_buffer(
            'basis',
            torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        )

        self.eps = eps
        self.codebook_size = torch.prod(self.levels)

        self.register_buffer('implicit_codebook', self.idxs_to_code(torch.arange(self.codebook_size)))

    def round_ste(self, z):
        z_q = torch.round(z)
        return z + (z_q - z).detach()

    def quantize(self, z):
        # half_l is used to determine how to scale tanh; we
        # subtract 1 from the number of levels to account for 0
        # being a quantization bin and tanh being symmetric around 0
        half_l = (self.levels - 1) * (1 - self.eps) / 2

        # if a given level is even, it will result in a scale for tanh
        # which is halfway between integer values, so we offset
        # the tanh output down by 0.5 to line it with whole integers
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)

        # if our level is even, we want to shift the tanh input to
        # ensure the 0 quantization bin is centered
        shift = torch.tan(offset / half_l)

        # once we have our shift and offset (in the case of an even level)
        # we can round to the nearest integer bin and allow for STE
        z_q = self.round_ste(torch.tanh(z + shift) * half_l - offset)

        # after quantization, we want to renormalize the quantized
        # values to be within the range expected by the model (ie. [-1, 1])
        half_width = self.levels // 2
        return z_q / half_width

    def scale_and_shift(self, z_q_normalized):
        half_width = self.levels // 2
        return (z_q_normalized * half_width) + half_width

    def scale_and_shift_inverse(self, z_q):
        half_width = self.levels // 2
        return (z_q - half_width) / half_width

    def code_to_idxs(self, z_q):
        z_q = self.scale_and_shift(z_q)
        return (z_q * self.basis).sum(dim=-1).to(torch.int32)

    def idxs_to_code(self, idxs):
        idxs = idxs.unsqueeze(-1)
        codes_not_centered = (idxs // self.basis) % self.levels
        return self.scale_and_shift_inverse(codes_not_centered)

    def forward(self, z):
        # TODO: make this work for generic tensor sizes
        # TODO: use einops to clean up
        B, C, T, H, W = z.shape

        # (B, C, T, H, W) -> (B, T, H, W, C)
        z_c_last = z.permute(0, 2, 3, 4, 1).contiguous()
        
        # (B, T, H, W, C) -> (BTHW, C)
        z_flatten = z_c_last.reshape(-1, C)
        
        z_flatten_q = self.quantize(z_flatten)
        
        # (BTHW, C) -> (B, T, H, W, C) -> (B, C, T, H, W)
        z_q = z_flatten_q.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        
        return {'z_q': z_q}


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
    

class FSQVAE(nn.Module):
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
        
        self.quantizer = FSQ(config.levels)

        self.config = config

    def encode(self, inputs):
        return self.encoder(inputs)

    def quantize(self, z):
        return self.quantizer(z)

    def decode(self, z_q):
        return self.decoder(z_q)

    def loss(self, x_hat, x, quantized):
        MSE = F.mse_loss(x_hat, x)

        return {
            'loss': MSE,
            **quantized
        }

    def forward(self, x):
        z = self.encode(x)
        quantized = self.quantize(z)
        x_hat = self.decode(quantized['z_q'])
        losses = self.loss(x_hat, x, quantized)

        return {'x_hat': x_hat, **losses}