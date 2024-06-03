# Autoregressive Video Generation Model

Heavy inspiration from the following work:

[VideoGPT](https://github.com/wilson1yan/VideoGPT)

[FSQ](https://arxiv.org/abs/2309.15505)

[MAGVIT](https://arxiv.org/abs/2212.05199)

[MAGVIT-V2](https://magvit.cs.cmu.edu/v2/)

[VideoPoet](https://research.google/blog/videopoet-a-large-language-model-for-zero-shot-video-generation/)

## Goal
**Build a video generation model which can generate novel scenes from Steamboat Willie**

This project was initially started because I had a naive idea for how I would build a video generation
model despite having never actually implemented even a vanilla VAE before. I had a high-level theoretical 
understanding of how VAEs worked, so I knew I could use one to learn good latent features of frames which could 
be fed into a transformer decoder for sequence modeling to get videos. After some digging, I found VideoGPT 
from 2021, which seemed to be the first video generation model to work in this way, so I knew my initial idea 
wasn't totally off from what the research community initially tried.

To familiarize myself with how to actually get VAEs to work well in practice, I implemented a VAE, VQ-VAE, and 
VQ-VAE-2 on CIFAR-10. I then modified the VQ-VAE from before to work with video clips to train it on 
Steamboat Willie. 

*best 16-frame, full spatial dimension reconstructions from VQ-VAE (without any codebook tricks)*:

![](assets/wooing-infatuation-93-1.gif)
![](assets/wooing-infatuation-93-2.gif)

I had some trouble with the VideoClips PyTorch tool taking too long to load clips, so I 
preprocessed the video clips and stored them as binary files for faster reads. While working on the VQ-VAE, I 
discovered Finite Scalar Quantization (FSQ) and liked the idea of it simplifying the whole quantization process 
and alleviating a lot of the training issues common in standard VQ-VAEs. I implemented it and used it in the now
FSQ-VAE, and trained that on the Steamboat Willie clips.

*best 16-frame, full spatial dimension reconstructions reconstructions from FSQ-VAE*:

![](assets/super_snowball_23_1.gif)
![](assets/super_snowball_23_2.gif)

Having seen the recently released Sora tech report from OpenAI, I liked the idea of operating on spatiotemporal
patches of the video, as it allowed for the model to handle variably shaped videos more easily. I could 
reassemble the patches back into video clips and get the full reconstructed clips back.

*current best tubelet reconstructions from FSQ-VAE (rearranged back into clips)*:

![](assets/pious_firefly_98_1.gif)
![](assets/pious_firefly_98_2.gif)

After finally getting some acceptable-looking recontructions, I fed the sequence of quantized encodings directly
into a transformer decoder which used FlashAttention2 with Sliding Window Attention along with ALiBi positional
encodings so I had the option of extending to longer sequences at inference time. This is where I found out I 
goofed up big time. The FSQ-VAE was simply not good enough at compressing the video clips, and the resulting
sequence was way too long. My transformer was also too small to generate coherent clips, and it quickly turned
into nonsense.

It was around this time I was recommended to check out VideoPoet from Google, which is the current
state of the art in autoregressive video generation. This model uses a video tokenizer called MAGVIT-v2, which is
the current state-of-the-art video tokenizer. So now, I have switched my approach to use this new MAGVIT-v2 
video tokenizer to get short video token sequences, and I am in the process of getting it to train effectively.

Stay tuned for more updates!

## Dataset Source
Steamboat Willie source: https://archive.org/download/steamboat-willie-mickey

## Current Project Roadmap

- [ ] Implement MAGVIT-V2 tokenizer
    - [X] Dialated Causal Convolution (in time dim)
    - [X] Blur Pool
    - [X] FSQ
    - [X] Encoder
    - [X] Decoder
    - [X] FSQ-VAE
    - [X] Discriminator
        - [X] grad penalty
    - [X] GAN Loss
    - [X] Delay GAN loss
    - [X] Replace GAN loss with hinge loss
    - [X] make Generator loss non-saturating
    - [ ] add LeCAM Regularization
    - [X] adjust discriminator to accept patches
      - [X] separate time downsample and space downsample
    - [X] add more capacity to FSQ-VAE
    - [X] increase vocabulary (level 9 -> 11)
    - [X] increase disc start channels and cap at paper value
    - [X] first disc delay for training disc, secondary disc delay for gen loss
    - [X] increase vae channels
    - [X] replace calculated channel increase with multipliers (for disc, enc, and dec)
    - [X] log real and fake logits
- [X] Implement the Transformer Decoder
- [ ] Implement the Super Resolution model
- [ ] Incorporate audio (optional/if feasible)

## MAGVIT2

![](assets/magvit2-arch-diagram.png)

This is a VQGAN style setup that uses a GAN loss to augment the FSQ-VAE reconstruction loss. The paper uses Lookup-Free Quantization, but I used Finite Scalar Quantization since 
it seemed to do well in other implementations of MAGVIT2 (and I've already implemented it). I have also been experimenting with Discriminators which take PatchGAN-style patches 
and spatiotemporal patches.

## Transformer Latent Sequence Model
For this, I will use FlashAttention2 in conjunction with the ALiBi positional encoder to efficiently model sequences while being able to extrapolate to longer sequences at inference time.

## Super Resolution Model
After we've mapped our generated sequence back into a 128 x 128 video, we can upsample the video frames to 256x256 (or maybe even 512 x 512?) to ensure our generated clip is tractable to learn and compute.
