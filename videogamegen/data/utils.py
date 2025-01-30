import torch
import numpy as np


def process_npz(self, npz_data: bytes) -> torch.Tensor:
    """Process .npz file into a tensor of frames."""
    with np.load(npz_data) as data:
        frames = data['frames']  
    
    frames = torch.from_numpy(frames).float()
    
    if frames.max() > 1.0:
        frames = frames / 255.0
        
    return frames