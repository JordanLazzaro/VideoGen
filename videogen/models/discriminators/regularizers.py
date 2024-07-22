import torch
from torch import nn
from torch.nn import functional as F


class LeCAM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer('logits_real_ema', torch.tensor(config.lecam_ema_init))
        self.register_buffer('logits_fake_ema', torch.tensor(config.lecam_ema_init))
        self.lecam_decay = config.lecam_decay

    def _update(self, logits_real, logits_fake):
        self.logits_real_ema = self.lecam_decay * self.logits_real_ema + (1 - self.lecam_decay) * torch.mean(logits_real)
        self.logits_fake_ema = self.lecam_decay * self.logits_fake_ema + (1 - self.lecam_decay) * torch.mean(logits_fake)

    def forward(self, logits_real, logits_fake):
        self._update(logits_real, logits_fake)
        return torch.mean(F.relu(logits_real - self.logits_fake_ema) ** 2) + torch.mean(F.relu(self.logits_real_ema - logits_fake) ** 2)