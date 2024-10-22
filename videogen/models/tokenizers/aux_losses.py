class VGGLoss(nn.Module):
    """
    Perceptual loss using VGG16 feature maps
    """
    def __init__(self) -> None:
        super().__init__()
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
           

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # TODO: modify to handle video
        # Ensure input is RGB
        if input.shape[1] != 3:
            input = repeat(input, 'b 1 h w -> b c h w', c=3)
            target = repeat(target, 'b 1 h w -> b c h w', c=3)
            
        # Normalize with ImageNet stats
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # Always resize to 224x224 as VGG expects this size
        input = F.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
        target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss = 0.0
        x = input
        y = target
        
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
            
        return loss