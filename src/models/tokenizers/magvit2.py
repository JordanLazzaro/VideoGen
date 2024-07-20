import torch
from torch import nn
from itertools import zip_longest
from einops import rearrange
from torch.autograd import grad as torch_grad
from torch.nn import functional as F

from modules import ( 
    Encoder,
    Decoder,
    FSQ,
    ResBlockDown2d,
    ResBlockDown3d
)


class FSQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(
            in_channels           = config.encoder.in_channels,
            out_channels          = config.encoder.out_channels,
            init_channels         = config.encoder.init_channels,
            channel_multipliers   = config.encoder.channel_multipliers,
            num_space_downsamples = config.encoder.num_space_downsamples,
            num_time_downsamples  = config.encoder.num_time_downsamples,
            nblocks               = config.encoder.nblocks
        )

        self.fsq = FSQ(levels = config.fsq.levels)

        self.decoder = Decoder(
            in_channels         = config.decoder.in_channels,
            out_channels        = config.decoder.out_channels,
            init_channels       = config.decoder.init_channels,
            channel_multipliers = config.encoder.channel_multipliers,
            num_space_upsamples = config.decoder.num_space_upsamples,
            num_time_upsamples  = config.decoder.num_time_upsamples,
            nblocks             = config.decoder.nblocks
        )

    def encode(self, x):
        return self.encoder(x)

    def quantize(self, z):
        return self.fsq(z)

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x):
        z = self.encode(x)
        z_q = self.quantize(z)
        x_hat = self.decode(z_q)
        return { 'z': z, 'z_q': z_q, 'x_hat': x_hat }
    

class PatchDiscriminator(nn.Module):
    ''' bunch of downsampling resblocks + leaky relu '''
    def __init__(self, config):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(
                in_channels  = config.in_channels,
                out_channels = config.init_channels,
                kernel_size  = (3, 3),
                padding      = 'same'
            ),
            nn.LeakyReLU()
        )

        channels = [config.init_channels * cm for cm in config.channel_multipliers]
        self.downsample = nn.Sequential(*[
            ResBlockDown2d(
                in_channels  = channels[i],
                out_channels = channels[i+1]
            )
            for i in range(config.num_space_downsamples)
        ])

        self.out_conv = nn.Conv2d(
            in_channels  = channels[-1],
            out_channels = config.out_channels,
            kernel_size  = (1, 1)
        )

    def forward(self, x):
        assert len(x.shape) == 4, 'input shape must be (B, C, H, W)'
        x = self.in_conv(x)
        x = self.downsample(x)
        x = self.out_conv(x)
        return x
    

class TubeletDiscriminator(nn.Module):
    ''' bunch of downsampling resblocks + leaky relu '''
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels  = config.in_channels,
                out_channels = config.init_channels,
                kernel_size  = (3, 3, 3),
                padding      = 'same'
            ),
            nn.LeakyReLU()
        )

        channels = [config.init_channels * cm for cm in config.channel_multipliers]
        space_downsample = [
            ResBlockDown3d(
                in_channels  = channels[i],
                out_channels = channels[i+1],
                space_only   = True
            )
            for i in range(config.num_space_downsamples)
        ]
        time_downsample = [
            ResBlockDown3d(
                in_channels  = channels[i+1] if i < config.num_space_downsamples else channels[i],
                out_channels = channels[i+1],
                time_only    = True
            )
            for i in range(config.num_time_downsamples)
        ]

        self.downsample = nn.Sequential(*[
            block
            for pair in zip_longest(space_downsample, time_downsample)
            for block in pair if block is not None
        ])

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels  = channels[-1],
                out_channels = channels[-1],
                kernel_size  = (3, 3, 3),
                padding      = 'same'
            ),
            nn.LeakyReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(4 * 4 * channels[-1], channels[-1]), # 4x4 after downsampling
            nn.LeakyReLU(),
            nn.Linear(channels[-1], 1)
        )

    def forward(self, x):
        assert len(x.shape) == 5, 'input shape must be (B, C, T, H, W)'
        x = self.conv1(x)
        x = self.downsample(x)
        x = self.conv2(x)
        x = rearrange(x, 'b ... -> b (...)')
        x = self.mlp(x)
        return x
    

class MAGVIT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fsqvae = FSQVAE(config.fsqvae)

        if config.discriminator.disc_type == 'tubelet':
            self.discriminator = TubeletDiscriminator(config.discriminator)
        elif config.discriminator.disc_type == 'patch':
            self.discriminator = PatchDiscriminator(config.discriminator)
        else:
            self.discriminator = None

    def forward(self, x):
        return self.fsqvae(x)

    def tokenize(self, x):
        z_q = self.fsqvae.quantize(self.fsqvae.encode(x))
        z_q = rearrange(z_q, 'b c d h w -> b (d h w) c')
        idxs = self.fsqvae.fsq.codes_to_idxs(z_q)
        return idxs

    def decode(self, idxs):
        c, d, h, w = self.config.latent_shape
        codes = self.fsqvae.fsq.idxs_to_codes(idxs)
        codes = rearrange(codes, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)
        return self.fsqvae.decode(codes)

    def discriminate(self, x):
        if self.discriminator is None:
            raise ValueError('discriminator not initialized')
        return self.discriminator(x)

    def reconstruction_loss(self, x_hat, x):
        if self.config.recon_loss_type == 'mae':
            return F.l1_loss(x_hat, x)
        elif self.config.recon_loss_type == 'mse':
            return F.mse_loss(x_hat, x)
        else:
            raise ValueError('invalid reconstruction loss type')

    def generator_loss(self, logits_fake):
        ''' non-saturating generator loss (NLL) '''
        return -torch.mean(logits_fake)

    def discriminator_loss(self, logits_real, logits_fake):
        '''
        smooth version hinge loss from:
        https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/losses/vqperceptual.py#L20C1-L24C18
        '''
        loss_real = torch.mean(F.softplus(1.0 - logits_real))
        loss_fake = torch.mean(F.softplus(1.0 + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake).mean()
        return d_loss

    def gradient_penalty(self, x, logits_real):
        '''
        inspired by:
        https://github.com/lucidrains/magvit2-pytorch/blob/9f49074179c912736e617d61b32be367eb5f993a/magvit2_pytorch/magvit2_pytorch.py#L99
        '''
        gradients = torch_grad(
            outputs = logits_real,
            inputs = x,
            grad_outputs = torch.ones(logits_real.size(), device = x.device),
            create_graph = True,
            retain_graph = True,
            only_inputs = True
        )[0]
        gradients = rearrange(gradients, 'b ... -> b (...)')
        return ((gradients.norm(2, dim = 1) - 1) ** 2).mean()