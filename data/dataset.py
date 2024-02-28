import tqdm
import os
import math
import torch
import torchvision
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, Resize, ToTensor, CenterCrop, Grayscale
from torchvision.datasets.video_utils import VideoClips


class RandomHorizontalFlipVideo(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        # x shape is expected to be (C, T, H, W)
        if torch.rand(1) < self.p:
            # Flip all frames in the clip
            return x.flip(-1)
        return x


class SteamboatWillieDataset(Dataset):
    def __init__(self, config, mode='train', train_split=0.8):
        super().__init__()
        self.config = config
        self.train_split = train_split
        self.mode = mode

        self.preprocess_transforms = Compose([
                Lambda(lambda x: x.permute(0, 3, 1, 2)), # (T, H, W, C) to (T, C, H, W) for Greyscale
                Grayscale(num_output_channels=1), # Convert to grayscale
                Lambda(lambda x: x.permute(1, 0, 2, 3)), # (T, C, H, W) to (C, T, H, W) for Conv3d
                CenterCrop((480, 575)), # Center crop to remove virtical bars
                Resize((config.img_size, config.img_size))
        ])

        self.postprocess_transforms = Compose([
            Lambda(lambda x: x / 255.),
            Lambda(lambda x: x.view(self.config.in_channels, self.config.clip_length, self.config.img_size, self.config.img_size))
        ])

        if self.mode == 'train':
            self.postprocess_transforms.transforms.append(RandomHorizontalFlipVideo(p=0.5))

        if os.path.exists(config.dest_dir):
            clip_paths = self.build_existing_clip_paths(config.dest_dir)
            self.clips = self.build_clip_refs(clip_paths)
        else:
            video_clips = VideoClips(
                config.paths,
                clip_length_in_frames=config.clip_length,
                frames_between_clips=config.clip_length
            )

            self.clips = self.build_clip_refs(self.build_clip_paths(video_clips, self.preprocess_transforms, config.dest_dir))

        if mode in ['train', 'val']:
            total_clips = len(self.clips)

            indices = torch.randperm(total_clips).tolist()
            train_size = int(total_clips * train_split)

            if mode == 'train':
                self.clip_indices = indices[:train_size]
            else:
                self.clip_indices = indices[train_size:]
        else:
            self.clip_indices = list(range(len(self.clips)))

    def build_clip_paths(self, video_clips, transforms, dest_dir):
        """
        Build set of binary files to store processed video clips
        returns dict of clip_idx -> mmapped file path
        """
        clip_paths = {}

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        for idx in tqdm(range(video_clips.num_clips()), desc='Creating clip .bin files'):
            # transform clips and write to mmap file
            clip, _, _, _ = video_clips.get_clip(idx)
            clip = self.preprocess_transforms(clip)
            clip_np = clip.numpy().astype(np.uint8)

            mmapped_file_path = os.path.join(dest_dir, f'clip_{idx}.bin')
            fp = np.memmap(mmapped_file_path, dtype='uint8', mode='w+', shape=clip_np.shape)
            fp[:] = clip_np[:]
            fp.flush()
            del fp
            clip_paths[idx] = mmapped_file_path

        return clip_paths

    def build_existing_clip_paths(self, dest_dir):
        """"
        returns dict of clip_idx -> mmapped file path
        from existing .bin files
        """
        clips_paths = {}
        for filename in os.listdir(dest_dir):
            if filename.startswith('clip_') and filename.endswith('.bin'):
                idx = int(filename.split('_')[1].split('.')[0])
                file_path = os.path.join(dest_dir, filename)
                clips_paths[idx] = file_path

        return clips_paths

    def build_clip_refs(self, clip_paths):
        """
        Build mmap reference to bin files
        returns dict of clip_idx -> np.array mmapped to respective bin file
        """
        clips = {}
        for idx, path in tqdm(clip_paths.items(), desc='Building clip refs'):
            clips[idx] = np.memmap(path, dtype='uint8', mode='r')

        return clips

    def __len__(self):
        return len(self.clip_indices)

    def __getitem__(self, idx):
        clip = self.clips[self.clip_indices[idx]]
        return self.postprocess_transforms(torch.tensor(clip, dtype=torch.float32))