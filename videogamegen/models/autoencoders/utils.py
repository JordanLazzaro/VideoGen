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

