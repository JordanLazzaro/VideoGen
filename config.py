class VideoVQVAEConfig:
    def __init__(self):
        # dataset properties
        self.paths = ['/content/drive/My Drive/SteamboatWillie/SteamboatWillie.mp4']
        self.dest_dir = './clips/'
        # model checkpoints
        self.checkpoint_path = "./checkpoints"
        self.save_top_k = 1
        # training
        self.train_split = 0.8
        self.batch_size = 8
        self.max_epochs = 120
        self.training_steps = 100000
        self.num_workers = 2
        # optimizer
        self.lr = 3e-4
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.weight_decay = 0.0 # 1e-2
        self.use_wd_schedule = False
        self.use_lr_schedule = False
        # input properties
        self.clip_length = 16
        self.img_size = 256
        self.in_channels = 1
        # latents / quantization
        self.latent_channels = 16
        self.codebook_size = 1024
        self.commit_loss_beta = 0.25
        self.track_codebook = True
        self.use_ema = True
        self.ema_gamma = 0.99
        # encoder/decoder
        self.hidden_channels = 256
        self.nblocks = 1
        self.nlayers = 4

    def update(self, updates):
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}