# Video Game Generation

## Goal
**Build a neural 1989 GameBoy simulator**

## Components
This implementation follows the same rough outline as [Genie (Generative Interactive Environment)](https://arxiv.org/abs/2402.15391),
with a model for learning compressed frame representations, a model for learning the latent action resulting in the next frame given
the previous frames, and a dynamics model for learning to predict the next frame given previous frames and current action.

### VAE
A VAE trained similarly to VQGAN/MAGVIT-v2 (without quantization), with a reconstruction loss, perceptual loss, and adversarial loss.
Right now I'm using a Conv2d encoder, but I might try out a ST-Transformer later.

### Latent Action Model (LAM)
Learn which action (select, start, d-up, d-down, d-left, d-right, a, b) corresponds with the next frame given previous frames.
Still trying to think about how I want to do this, but I'm thinking of using the ST-Transformer with Finite Scalar Quantization.

### Dynamics Model
Learn which state follows previous states conditioned on an action from the LAM codebook (Diffusion Forcing + Flow Matching).
I'm going to use a causal transformer to predict the next frame using diffusion forcing with Flow Matching for stable
autoregressive frame generation.

## Longplay Dataset
WebDataset with .tar files containing a .bin file of compressed frames.

Videos are scraped from [world of longplays](https://longplays.org/infusions/longplays/longplays.php?cat_id=30) with associated metadata
for contributor attribution and so on.

Training on longplays for:
- Tetris
- Super Mario Land
- Super Mario Land 2
- Super Mario Land 3
- Pokemon Red
- Pokemon Blue
- The Legend of Zelda
- Kirby's Dream Land
- Kirby's Dream Land 2
- Donkey Kong
- Donkey Kong Land