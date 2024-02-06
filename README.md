# VideoGPT
Implementation of VideoGPT

https://github.com/wilson1yan/VideoGPT

Steamboat Willie source: https://archive.org/download/steamboat-willie-mickey

VideoGPT utilizes a two-model, two-stage approach where batches of frames
from videos are first used to train a VQ-VAE with 3D Conv layers, and the resulting latent
codebook is used as a vocabulary for a transformer decoder to learn to model sequences of
video frames in latent space. The sequences of latents are then used by the VQ-VAE decoder
to map back to a sequence of frames in image space.

This project will require:

[X] Part 1: VAE - CIFAR10

[X] Part 2: VQ-VAE/VQ-VAE 2 (Conv2D) - CIFAR10

[] Part 3: (FSQ?) VQ-VAE (Conv3D) - Steamboat Willie

[] Part 4: Transformer Decoder - latent codes from Part 3
