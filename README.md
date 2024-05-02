# Autoregressive Video Generation Model

Heavy inspiration from the following work:

[VideoGPT](https://github.com/wilson1yan/VideoGPT)

[FSQ-VAE](https://arxiv.org/abs/2309.15505)

[MAGVIT](https://arxiv.org/abs/2212.05199)

[MAGVIT-V2](https://magvit.cs.cmu.edu/v2/)

[VideoPoet](https://research.google/blog/videopoet-a-large-language-model-for-zero-shot-video-generation/)

## Goal

**Build MAGVIT2 visual tokenizer (using FSQ in place of LFQ) and use it to tokenize Steamboat Willie so it can be modeled by a transformer sequence model**

![](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgxFblHaHRJNH7Oi2_oOTosGN9XrjgjhWmnfADchMT8WR0XAo6SxiUfpUmn5R6akciiRduaKIMdgwHZzK3xW8mErarQ_ugx41ctQAMK08O9UMVevgkk-AgFI1xYFWAomd16OcOh0R-XpyZVLQXncpk2SHf-RmPzrqBbIWZc-nUG2TH6nC2R7qyHXn8eTC-u/s2680/image21.png)

## Dataset Source
Steamboat Willie source: https://archive.org/download/steamboat-willie-mickey

## Current Highlights
Clip reconstructions from roadmap steps 3 and 4

*best 16-frame, full spatial dimension reconstructions from VQ-VAE before switching to FSQ*:

![](assets/wooing-infatuation-93-1.gif)
![](assets/wooing-infatuation-93-2.gif)

*best 16-frame, full spatial dimension reconstructions reconstructions from FSQ-VAE*:

![](assets/super_snowball_23_1.gif)
![](assets/super_snowball_23_2.gif)

*current best tubelet reconstructions from FSQ-VAE (rearranged back into clip)*:

![](assets/pious_firefly_98_1.gif)
![](assets/pious_firefly_98_2.gif)

*Better clips coming with updated approach*


## Project Roadmap

- [ ] Implement MAGVIT-V2 tokenizer
    - [X] Dialated Causal Convolution (in time dim)
    - [X] Blur Pool
    - [X] FSQ-VAE
    - [X] Descriminator
    - [X] GAN Loss
    - [ ] Perceptual Loss
- [ ] Implement the Transformer Decoder
- [ ] Implement the Super Resolution model
- [ ] Incorporate audio (extra credit)

## MAGVIT2
This is a VQVAE-style setup which adds a GAN loss to the reconstruction loss. The paper uses Lookup-Free Quantization, but I would like to try Finite Scalar Quantization since it seems to do well in other implementations of MAGVIT2 (and I've already implemented it)

## Transformer Decoder (Spatio-Temporal Latent Prediction)
For this, I will use FlashAttention2 in conjunction with the ALiBi positional encoder to efficiently model sequences while being able to extrapolate to longer sequences at inference time.

## Super Resolution
After we've mapped our generated sequence back into a 128 x 128 video, we can upsample the video frames to 256x256 (or maybe even 512 x 512?) to ensure our generated clip is tractable to learn and compute.
