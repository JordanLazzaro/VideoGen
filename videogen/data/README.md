# GameBoy Longplays
This dataset consists of all 489 [Nintendo GameBoy](https://en.wikipedia.org/wiki/Game_Boy) longplays (LPs) hosted on
[The World of Longplays](https://longplays.org/infusions/longplays/longplays.php?cat_id=30). Long Plays are intented to show a complete, 
thorough, and normally paced playthrough of a given game, including all or most of the in-game content. I hypothesize that this makes them a 
very information dense dataset for training [Generative Interactive Environments](https://arxiv.org/abs/2402.15391), as they capture the full 
range of environment dynamics and interactability from the player. 

One interesting caveat is that the player generally never looses the game or looses any signifacant amount of health/resources/tries in a 
longplay recording, showing a clear and important gap in their representation of game mechanics. They will at times also be composed of 
recordings from separate attempts clipped together, which could potentially impact the continuity of the underlying game mechanics as they are 
represented in the recording. Both of these are major open questions to be addressed and evaluated post training.

The GameBoy display has an aspect ratio of 10:9, with size: W=160px H=144px, and is a 2-bit green display. The longplay videos
are in a higher resolution, color shifted to grayscale, and contains black side bars, so we can crop the video, resize to a 1:1 aspect ratio, 
convert from RGB to Grayscale, and downscale to W=256px, H=256px without much loss of perceptual quality.

## Frame Dataset
This dataset is intended to train a VAE for learning latent representations, and consists of W=256px H=256px, grayscale frames. All videos are 
are stored as an HDF5 dataset.

## Latent Action Model Dataset
The Latent Action Model dataset consists of a sequence of frame latents encoded by our pretrained VAE, where all frame latents are packed and stored as a contiguous, memory-mapped binary file in float32 precision.

We will  has 6 player controls (D-Up, D-Down, D-Left, D-Right, A, B), as well as a 7th implicit NoOp action. 

## Latent-Action Dynamics Dataset
Collect a sequence of frame latents encoded by our pretrained VAE and their corresponding action codes from our pretrained Latent Action Model.