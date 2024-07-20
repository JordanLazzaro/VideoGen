import torch
from torch import nn, einsum
from torch.nn import functional as F
from einops import rearrange, repeat

def get_blur_filter(kernel_size):
    if kernel_size == 1:
        return torch.tensor([1.,])
    elif kernel_size == 2:
        return torch.tensor([1., 1.])
    elif kernel_size == 3:
        return torch.tensor([1., 2., 1.])
    elif kernel_size == 4:
        return torch.tensor([1., 3., 3., 1.])
    elif kernel_size == 5:
        return torch.tensor([1., 4., 6., 4., 1.])
    elif kernel_size == 6:
        return torch.tensor([1., 5., 10., 10., 5., 1.])
    elif kernel_size == 7:
        return torch.tensor([1., 6., 15., 20., 15., 6., 1.])
    else:
        raise ValueError(f'Blur kernel size {kernel_size} not supported')


class CausalConv3d(nn.Module):
    '''
    enforces causality in the time dimension (https://paperswithcode.com/method/causal-convolution)
    inspired by:
    https://github.com/lucidrains/magvit2-pytorch/blob/9f49074179c912736e617d61b32be367eb5f993a/magvit2_pytorch/magvit2_pytorch.py#L889
    '''
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(3,3,3),
            stride=(1, 1, 1),
            dilation=(1, 1, 1),
            pad_mode='constant'
        ):
        super().__init__()
        k_t, k_h, k_w = kernel_size
        pad_w, pad_h, pad_t = k_w//2, k_h//2, dilation[0] * (k_t - 1) + (1 - stride[0])

        # pad: (left, right, top, bottom, front, back)
        self.t_causal_pad = (pad_w, pad_w, pad_h, pad_h, pad_t, 0)
        self.pad_mode = pad_mode

        self.conv = nn.Conv3d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            dilation     = dilation
        )

    def forward(self, x):
        x = F.pad(x, self.t_causal_pad, mode=self.pad_mode)
        x = self.conv(x)

        return x
    

class BlurPool2d(nn.Module):
    '''
    https://arxiv.org/abs/1904.11486
    inspired by:
    https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
    https://github.com/lucidrains/magvit2-pytorch/blob/9f49074179c912736e617d61b32be367eb5f993a/magvit2_pytorch/magvit2_pytorch.py#L509
    '''
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels

        self.stride = (2, 2)
        self.padding = ((kernel_size - 1) // 2, (kernel_size - 1) // 2)

        self.register_buffer('blur_filter', self.get_blur_filter2d(kernel_size))

    def forward(self, x):
        assert len(x.shape) == 4, 'BlurPool2d only supports rank 4 tensors'
        return F.conv2d(x, self.blur_filter, stride=self.stride, padding=self.padding)

    def get_blur_filter2d(self, kernel_size):
        filter = get_blur_filter(kernel_size)        

        filter = einsum('i, j -> i j', filter, filter)

        filter = repeat(filter, 'h w -> oc ic h w', oc=self.in_channels, ic=self.in_channels)

        return filter / torch.sum(filter)
    

class BlurPool3d(nn.Module):
    '''
    https://arxiv.org/abs/1904.11486
    inspired by:
    https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
    https://github.com/lucidrains/magvit2-pytorch/blob/9f49074179c912736e617d61b32be367eb5f993a/magvit2_pytorch/magvit2_pytorch.py#L509
    '''
    def __init__(self, in_channels, kernel_size=3, space_only=False, time_only=False):
        super().__init__()
        self.in_channels = in_channels
        self.space_only = space_only
        self.time_only = time_only

        if space_only:
            self.stride = (1, 2, 2)
            self.padding = (0, (kernel_size - 1) // 2, (kernel_size - 1) // 2)
        elif time_only:
            self.stride = (2, 1, 1)
            self.padding = ((kernel_size - 1) // 2, 0, 0)
        else:
            self.stride = (2, 2, 2)
            self.padding = ((kernel_size - 1) // 2, (kernel_size - 1) // 2, (kernel_size - 1) // 2)

        self.register_buffer('blur_filter', self.get_blur_filter3d(kernel_size))

    def forward(self, x):
        assert len(x.shape) == 5, 'BlurPool3d only supports rank 5 tensors'
        return F.conv3d(x, self.blur_filter, stride=self.stride, padding=self.padding)

    def get_blur_filter3d(self, kernel_size):
        filter = get_blur_filter(kernel_size)

        if self.space_only:
            filter = einsum('i, j -> i j', filter, filter)
            filter = rearrange(filter, '... -> 1 ...')
        elif self.time_only:
            filter = rearrange(filter, 'f -> f 1 1')
        else:
            filter = einsum('i, j, k -> i j k', filter, filter, filter)

        filter = repeat(filter, 'd h w -> oc ic d h w', oc=self.in_channels, ic=self.in_channels)

        return filter / torch.sum(filter)
    

class ResBlock3d(nn.Module):
    ''' handles both same and different in/out channels '''
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(3,3,3)
        ):
        super().__init__()
        if in_channels != out_channels:
            self.identity = CausalConv3d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = (1, 1, 1)
            )
        else:
            self.identity = nn.Identity()

        self.block = nn.Sequential(
            nn.GroupNorm(
                num_groups   = in_channels // 2 if in_channels >= 2 else 1,
                num_channels = in_channels
            ),
            nn.SiLU(),
            CausalConv3d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = (1,1,1),
                dilation     = (1,1,1)
            ),
            nn.GroupNorm(
                num_groups   = out_channels // 2 if out_channels >= 2 else 1,
                num_channels = out_channels
            ),
            nn.SiLU(),
            CausalConv3d(
                in_channels  = out_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = (1,1,1),
                dilation     = (1,1,1)
            ),
        )

    def forward(self, x):
        out = self.block(x) + self.identity(x)
        return out
    

class ResBlockDown2d(nn.Module):
    ''' strided conv for down-sampling + blur pooling (https://arxiv.org/abs/1904.11486) '''
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(3,3),
        ):
        super().__init__()
        self.identity = nn.Sequential(
            nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = (1, 1),
                stride       = 2
            )
        )

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                padding      = 'same'
            ),
            nn.GroupNorm(
                num_groups   = out_channels // 2 if out_channels >= 2 else 1,
                num_channels = out_channels
            ),
            nn.LeakyReLU(),
            BlurPool2d(out_channels),
            nn.Conv2d(
                in_channels  = out_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                padding      = 'same'
            ),
            nn.GroupNorm(
                num_groups   = out_channels // 2 if out_channels >= 2 else 1,
                num_channels = out_channels
            ),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x) + self.identity(x)
    

class ResBlockDown3d(nn.Module):
    ''' strided conv for down-sampling + blur pooling (https://arxiv.org/abs/1904.11486) '''
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(3,3,3),
            space_only=False,
            time_only=False
        ):
        super().__init__()
        if space_only:
            id_stride = (1, 2, 2)
        elif time_only:
            id_stride = (2, 1, 1)
        else:
            id_stride = (2, 2, 2)
       
        self.identity = nn.Sequential(
            nn.Conv3d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = (1, 1, 1),
                stride       = id_stride
            )
        )

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                padding      = 'same'
            ),
            nn.LeakyReLU(),
            BlurPool3d(out_channels, space_only=space_only, time_only=time_only),
            nn.Conv3d(
                in_channels  = out_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                padding      = 'same'
            ),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x) + self.identity(x)
    

class PixelShuffle3d(nn.Module):
    ''' https://arxiv.org/abs/1609.05158 '''
    def __init__(self, space_only=False, time_only=False):
        super().__init__()
        if space_only:
            self.r = (1, 2, 2)
        elif time_only:
            self.r = (2, 1, 1)
        else:
            self.r = (2, 2, 2)

    def forward(self, x):
        return rearrange(
            x,
            'b (c r1 r2 r3) d h w -> b c (d r1) (h r2) (w r3)',
            r1=self.r[0], r2=self.r[1], r3=self.r[2]
        )
    

class Upsample3d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            space_only=False,
            time_only=False
        ):
        super().__init__()
        if space_only:
            cm = 4
        elif time_only:
            cm = 2
        else:
            cm = 8
        self.conv = CausalConv3d(
            in_channels  = in_channels,
            out_channels = out_channels * cm,
            kernel_size  = (3,3,3)
        )
        self.pixel_shuffle = PixelShuffle3d(
            space_only=space_only,
            time_only=time_only
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x
    

class EncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            nblocks,
            kernel_size=(3,3,3),
            space_only=False,
            time_only=False
        ):
        super().__init__()
        if space_only:
            stride = (1, 2, 2)
        elif time_only:
            stride = (2, 1, 1)
        else:
            stride = (2, 2, 2)
        
        self.block = nn.Sequential(
            CausalConv3d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride
            ),
            nn.Sequential(*[
                ResBlock3d(
                    in_channels  = out_channels,
                    out_channels = out_channels
                )
                for _ in range(nblocks)
            ])
        )

    def forward(self, x):
        return self.block(x)
    

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            nblocks=2,
            space_only=False,
            time_only=False
        ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Sequential(*[
                ResBlock3d(
                    in_channels  = in_channels if i==0 else out_channels,
                    out_channels = out_channels
                )
                for i in range(nblocks)
            ]),
            nn.GroupNorm(
                num_groups   = out_channels // 2 if out_channels >= 2 else 1,
                num_channels = out_channels
            ),
            Upsample3d(
                in_channels   = out_channels,
                out_channels  = out_channels,
                space_only    = space_only,
                time_only     = time_only 
            )
        )

    def forward(self, x):
        return self.block(x)
    

class FSQ(nn.Module):
    def __init__(self, levels, eps=1e-3):
        super().__init__()
        self.register_buffer('levels', torch.tensor(levels))

        self.register_buffer('basis',
            torch.cat([
                torch.tensor([1]),
                torch.cumprod(torch.tensor(levels[:-1]), dim=0)
            ], dim=0)
        )

        self.eps = eps
        self.codebook_size = torch.prod(self.levels)

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

    def codes_to_idxs(self, z_q):
        assert z_q.shape[-1] == len(self.levels)
        z_q = self.scale_and_shift(z_q)
        return (z_q * self.basis).sum(dim=-1).to(torch.int32)

    def idxs_to_codes(self, idxs):
        idxs = idxs.unsqueeze(-1)
        codes_not_centered = (idxs // self.basis) % self.levels
        return self.scale_and_shift_inverse(codes_not_centered)

    @autocast(enabled = False)
    def forward(self, z):
        if len(z.shape) == 5: # video
            B, C, T, H, W = z.shape
            z = rearrange(z, 'b c t h w -> (b t h w) c')
            z_q = self.quantize(z)
            z_q = rearrange(z_q, '(b t h w) c -> b c t h w', b=B, t=T, h=H, w=W)

        elif len(z.shape) == 4: # image
            B, C, H, W = z.shape
            z = rearrange(z, 'b c h w -> (b h w) c')
            z_q = self.quantize(z)
            z_q = rearrange(z_q, '(b h w) c -> b c h w', b=B, h=H, w=W)

        elif len(z.shape) == 2: # vector sequence
            z_q = self.quantize(z)
        return z_q
    

class Quantizer(nn.Module):
    def __init__(
            self,
            codebook_size,
            latent_channels,
            ema_gamma,
            commit_loss_beta,
            use_ema,
            track_codebook
        ):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, latent_channels)
        torch.nn.init.uniform_(self.codebook.weight, -1/codebook_size, 1/codebook_size)

        if use_ema:
            self.register_buffer('N', torch.zeros(codebook_size) + 1e-6)
            self.register_buffer('m', torch.zeros(codebook_size, latent_channels))
            torch.nn.init.uniform_(self.m, -1/codebook_size, 1/codebook_size)

        if track_codebook:
            self.register_buffer('codebook_usage', torch.zeros(codebook_size, dtype=torch.float))
            self.register_buffer('total_usage', torch.tensor(0, dtype=torch.float))

        self.commit_loss_beta = commit_loss_beta
        self.ema_gamma = ema_gamma
        self.latent_channels = latent_channels
        self.codebook_size = codebook_size
        self.use_ema = use_ema
        self.track_codebook = track_codebook

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
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flat_inputs = inputs.reshape(-1, self.latent_channels)

        # Σ(x-y)^2 = Σx^2 - 2xy + Σy^2
        dists = (
            torch.sum(flat_inputs ** 2, dim=1, keepdim=True) - # Σx^2
            2 * (flat_inputs @ self.codebook.weight.t()) +     # 2*xy
            torch.sum(self.codebook.weight ** 2, dim=1)        # Σy^2
        )

        code_idxs = torch.argmin(dists, dim=1)
        quantized_inputs = self.codebook(code_idxs).reshape(inputs.shape)

        if not self.use_ema:
            # "The VQ objective uses the l2 error to move the embedding vectors
            # e_i towards the encoder outputs z_e(x)"
            embedding_loss = F.mse_loss(quantized_inputs, inputs.detach())
        
        if self.training and self.use_ema:
            # perform exponential moving average update for codebook
            self.ema_update(code_idxs, flat_inputs)

        # "since the volume of the embedding space is dimensionless, it can grow
        # arbitrarily if the embeddings e_i do not train as fast as the encoder
        # parameters. To make sure the encoder commits to an embedding and its
        # output does not grow, we add a commitment loss"
        commitment_loss = F.mse_loss(quantized_inputs.detach(), inputs)

        # parts 2 & 3 of full loss (ie. not including reconstruciton loss)
        vq_loss = commitment_loss * self.commit_loss_beta
        
        if not self.use_ema:
            vq_loss += embedding_loss

        # sets the output to be the input plus the residual value between the
        # quantized latents and the inputs like a resnet for Straight Through
        # Estimation (STE)
        quantized_inputs = inputs + (quantized_inputs - inputs).detach()
        quantized_inputs = quantized_inputs.permute(0, 3, 1, 2).contiguous()

        if self.track_codebook:
            perplexity = self.calculate_perplexity(code_idxs)

        return {
            'quantized_inputs': quantized_inputs,
            'vq_loss':          vq_loss,
            'embedding_loss':   embedding_loss if not self.use_ema else None,
            'commitment_loss':  commitment_loss,
            'perplexity':       perplexity if self.track_codebook else None
        }
    

class Encoder(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            init_channels,
            channel_multipliers,
            num_space_downsamples,
            num_time_downsamples,
            nblocks
        ):
        super().__init__()
        channels = [init_channels * cm for cm in channel_multipliers]
        self.in_conv = CausalConv3d(
            in_channels  = in_channels,
            out_channels = channels[0],
            kernel_size  = (3,3,3)
        )
        space_downsample = [
            EncoderBlock(
                in_channels   = channels[i],
                out_channels  = channels[i+1],
                nblocks       = nblocks,
                kernel_size   = (3,3,3),
                space_only=True
            )
            for i in range(num_space_downsamples)
        ]
        time_downsample = [
            EncoderBlock(
                in_channels   = channels[i+1] if i < num_space_downsamples else channels[i],
                out_channels  = channels[i+1],
                nblocks       = nblocks,
                kernel_size   = (3,3,3),
                time_only=True
            )
            for i in range(num_time_downsamples)
        ]

        self.enc_blocks = nn.Sequential(*[
            block
            for pair in zip_longest(space_downsample, time_downsample)
            for block in pair if block is not None
        ])

        self.out_block = nn.Sequential(
            nn.GroupNorm(
                num_groups   = channels[-1] // 2,
                num_channels = channels[-1]
            ),
            nn.SiLU(),
            CausalConv3d(
                in_channels  = channels[-1],
                out_channels = out_channels,
                kernel_size  = (1,1,1)
            ),
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.enc_blocks(x)
        x = self.out_block(x)
        return x
    

class Decoder(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            init_channels,
            channel_multipliers,
            num_space_upsamples,
            num_time_upsamples,
            nblocks
        ):
        super().__init__()
        channels = [init_channels * cm for cm in channel_multipliers][::-1]
        self.in_block = nn.Sequential(
            CausalConv3d(
                in_channels  = in_channels,
                out_channels = channels[0],
                kernel_size  = (3,3,3)
            ),
            nn.Sequential(*[
                ResBlock3d(
                    in_channels  = channels[0],
                    out_channels = channels[0]
                )
                for i in range(nblocks)
            ])
        )
        
        space_upsample = [
            DecoderBlock(
                in_channels  = channels[i],
                out_channels = channels[i+1],
                nblocks      = nblocks,
                space_only   = True,
            )
            for i in range(num_space_upsamples)]
        
        time_upsample = [
            DecoderBlock(
                in_channels  = channels[i+1] if i < num_space_upsamples else channels[i],
                out_channels = channels[i+1],
                nblocks      = nblocks,
                time_only    = True,
            )
            for i in range(num_time_upsamples)]

        self.dec_blocks = nn.Sequential(*[
            block
            for pair in zip_longest(space_upsample, time_upsample)
            for block in pair if block is not None
        ])
        
        self.out_block = nn.Sequential(
            nn.SiLU(),
            CausalConv3d(
                in_channels  = channels[-1],
                out_channels = out_channels,
                kernel_size  = (3,3,3)
            )
        )

    def forward(self, x):
        x = self.in_block(x)
        x = self.dec_blocks(x)
        x = self.out_block(x)
        return x
    

class LeCAM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer('logits_real_ema', torch.tensor(config.lecam_ema_init))
        self.register_buffer('logits_fake_ema', torch.tensor(config.lecam_ema_init))
        self.lecam_decay = config.lecam_decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.lecam_decay * self.logits_real_ema + (1 - self.lecam_decay) * torch.mean(logits_real)
        self.logits_fake_ema = self.lecam_decay * self.logits_fake_ema + (1 - self.lecam_decay) * torch.mean(logits_fake)

    def forward(self, real_pred, fake_pred):
        return torch.mean(F.relu(real_pred - self.logits_fake_ema) ** 2) + torch.mean(F.relu(self.logits_real_ema - fake_pred) ** 2)