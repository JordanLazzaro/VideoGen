from torch import nn
from torch.nn import functional as F

from modules import Upsample3d,ResBlock3d


class SuperResolution(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.upsample_blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Sequential(*[
                    ResBlock3d(
                        in_channels  = config.in_channels if i==0 and j==0 else config.hidden_channels,
                        out_channels = config.hidden_channels,
                        space_only   = True
                    )
                    for j in range(config.nblocks)
                ]),
                nn.GroupNorm(
                    num_groups   = config.hidden_channels // 2 if config.hidden_channels >= 2 else 1,
                    num_channels = config.hidden_channels
                ),
                nn.SiLU(),
                Upsample3d(
                    config.in_channels,
                    out_channels=config.out_channels if i==config.num_upsamples-1 else config.hidden_channels,
                    space_only=True
                ),
                nn.SiLU() if i<config.num_upsamples-1 else nn.Identity()
            ) for i in range(config.num_upsamples)
        ])
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        return self.out_act(self.upsample_blocks(x))

    def loss(self, y_hat, y):
        if self.config.loss_type == 'mae':
            return F.l1_loss(y_hat, y)
        elif self.config.loss_type == 'mse':
            return F.mse_loss(y_hat, y)
        else:
            raise ValueError('invalid reconstruction loss type')