import torch
import torch.nn.functional as F
from torch import nn


class BaseLoss(nn.Module):
    def __init__(self, name, weight: float = 1.0):
        super().__init__()
        self.name = name
        self.weight = weight

    def forward(self, predictions, targets, **kwargs):
        raise NotImplementedError


class ReconstructionLoss(BaseLoss):
    def __init__(self, weight: float = 1.0, recon_loss_type: str = 'mse'):
        super().__init__(name='reconstruction', weight=weight)
        self.recon_loss_type = recon_loss_type

    def forward(self, out, x):
        if recon_self.recon_loss_typeloss_type == 'mae':
            loss = F.l1_loss(out['x_hat'], x)
        elif self.recon_loss_type == 'mse':
            loss = F.mse_loss(out['x_hat'], x)
        
        return self.weight * loss


class KLDLoss(BaseLoss):
    def __init__(self, weight: float = 1.0):
        ''' currently calculates KL between two Gaussian distributions '''
        super().__init__(name='kld', weight=weight)

    def forward(self, out, x):
        assert out['mu'] is not None and out['logvar'] is not None, 'mu and logvar must be a part of out'
        assert out['mu'].shape == out['logvar'].shape, 'mu must be same shape as logvar'
        assert len(out['mu'].shape) == 4 and len(out['logvar']) == 4, 'mu and logvar must be of shape: (B, C, H, W)'

        return torch.mean(-0.5 * torch.sum(1 + out['logvar'] - out['mu'].pow(2) - out['logvar'].exp(), dim=[1, 2, 3]), dim=0)


class PerceptualLoss(BaseLoss):
    """
    Perceptual loss using VGG16 feature maps
    """
    def __init__(self, weight: float = 1.0, model_name: str = "vgg16"):
        super().__init__(name='perceptual', weight=weight)
        
        vgg = models.vgg16(pretrained=True).features
        blocks = []
        blocks_info = [(0, 4), (4, 9), (9, 16), (16, 23), (23, 30)]
        
        for start, end in blocks_info:
            block = vgg[start:end]
            for param in block.parameters():
                param.requires_grad = False
            blocks.append(block)
            
        self.blocks = nn.ModuleList(blocks)
        self.register_buffer("mean", rearrange(torch.tensor([0.485, 0.456, 0.406]), 'c -> 1 c 1 1'))
        self.register_buffer("std", rearrange(torch.tensor([0.229, 0.224, 0.225]), 'c -> 1 c 1 1'))
           

    def forward(self, out, x) -> torch.Tensor:
        if x.shape[1] != 3:
            x = repeat(x, 'b 1 h w -> b c h w', c=3)
            x_hat = repeat(out['x_hat'], 'b 1 h w -> b c h w', c=3)
            
        # normalize with ImageNet stats
        x = (x - self.mean) / self.std
        x_hat = (x_hat - self.mean) / self.std
        
        # resize to 224x224 as VGG expects this size
        x = F.interpolate(x, mode='bilinear', size=(224, 224), align_corners=False)
        x_hat = F.interpolate(x_hat, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            x_hat = block(x_hat)
            loss += F.l1_loss(x_hat, x)
            
        return loss


class AdversarialLoss(BaseLoss):
    def __init__(
            self,
            weight: tuple = (1.0, 1.0),
            discriminator: Discriminator = None,
            discriminator_delay: int = 0,
            regularization_loss: nn.Module = None,
            regularization_loss_weight: float = 0.0,
            grad_penalty_weight: float = 0.0,
        ):
        super().__init__(name='adversarial', weight=weight)
        assert discriminator is not None, 'Discriminator must be provided'
        
        self.discriminator = discriminator
        self.grad_penalty_weight = grad_penalty_weight
        self.regularization_loss_weight = regularization_loss_weight
        self.regularization_loss = regularization_loss

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
        d_loss = (loss_real + loss_fake).mean()
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
    
    def forward(self, out, x, mode='generator'):
        loss = {} # TODO: I wanna log unweighted individual losses and total loss
        if mode == 'generator':
            logits_fake = self.discriminator(out['x_hat'])
            loss.update({ 'generator_loss': self.weight[0] * self.generator_loss(logits_fake) })
        elif mode == 'discriminator':
            logits_real = self.discriminator(x)
            logits_fake = self.discriminator(out['x_hat'].detatch())
            
            disc_loss = self.weight[1] * self.discriminator_loss(logits_real, logits_fake)
            
            if self.grad_penalty_weight > 0.0:
                disc_loss += self.grad_penalty_weight * self.gradient_penalty(x, logits_real)
            if self.regularization_loss_weight > 0.0:
                disc_loss += self.regularization_loss_weight * self.regularization_loss(logits_real, logits_fake)
        else:
            raise ValueError('Invalid mode type.')

        return loss

        