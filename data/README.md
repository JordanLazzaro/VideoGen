# Steamboat Willie

source: https://archive.org/download/steamboat-willie-mickey

Steamboat Willie was the first animated film featuring Mickie Mouse, and premiered on November 18, 1928
in New York City. This is the date celebrated as Mickie Mouse's birthday.

Given that it's a short animated cartoon from the 1920's, I figured the clips would be low entropy 
enough to keep them at a reasonable resolution (256x256) for maintaining image clarity without
requiring too large a model because I am kind of GPU poor.

The torchvision VideoClips library which I saw being used by the original VideoGPT repo was super slow and the dataloader
was crawling, so I converted all the clips to numpy arrays of dtype uint8 and stored them as individual bin files.
I could then keep a dictionary mapping clip indices to their corresponding mmapped np.arrays and convert them to float32
PyTorch Tensors in ```__getitem__(self, idx)``` 