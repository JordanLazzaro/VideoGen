import math


def get_num_downsample_layers(img_size):
    """
    Get the number of strided Conv2D layers
    required to produce a 2x2 output volume
    """
    if img_size < 2:
        raise ValueError("Image size must be at least 2x2.")

    # Calculate the minimum number of downsample layers required for 2x2 final
    num_layers = math.ceil(math.log2(img_size / 2))
    return num_layers

def build_channel_dims(start_channels, nlayers):
    """
    Construct a list of channel counts for nlayers downsample layers
    assuming the channels double as spatial dims halve
    """
    channels = []
    for _ in range(nlayers):
        channels.append(start_channels)
        start_channels *= 2
    return channels

class VideoVAEConfig:
    def __init__(self):
        self.project_name = 'FSQ-VAE Steamboat Willie'
        # dataset properties
        self.paths = ['/content/drive/My Drive/SteamboatWillie/SteamboatWillie.mp4']
        self.dest_dir = './clips/'
        # model checkpoints
        self.checkpoint_path = "./checkpoints"
        self.save_top_k = 1
        # training
        self.train_split = 0.8
        self.batch_size = 32
        self.max_epochs = 500
        self.training_steps = 100000
        self.num_workers = 2
        # optimizer
        self.lr = 6e-5
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.weight_decay = 0.0 # 1e-2
        self.use_wd_schedule = False
        self.use_lr_schedule = False
        # input properties
        self.clip_length = 16
        self.img_size = 256
        self.in_channels = 1
        # quantization
        self.quant_mode = 'fsq' # 'vq'
        self.latent_channels = 4 # 8
        self.codebook_size = 512
        self.commit_loss_beta = 0.25
        self.track_codebook = True
        self.use_ema = True
        self.ema_gamma = 0.99
        self.levels = [5, 5, 5, 5] # |C| = 625
        # encoder/decoder
        self.hidden_channels = 256
        self.start_channels = 32
        self.nblocks = 4
        self.nlayers = 3

    def update(self, updates):
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}