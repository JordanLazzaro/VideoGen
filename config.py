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
        self.lr = 6e-4
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.weight_decay = 0.0 # 1e-2
        self.use_wd_schedule = False
        self.use_lr_schedule = True
        # input properties
        self.clip_length = 16
        self.img_size = 256
        self.in_channels = 1
        # quantization
        self.quant_mode = 'fsq' # 'vq'
        self.latent_channels = 10 # 8
        self.codebook_size = 512
        self.commit_loss_beta = 0.25
        self.track_codebook = True
        self.use_ema = True
        self.ema_gamma = 0.99
        self.level = 7
        self.levels = [self.level for _ in range(self.latent_channels)]
        # encoder/decoder
        self.hidden_channels = 256
        self.start_channels = 32
        self.nblocks = 5
        self.nlayers = 3

    def update(self, updates):
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}