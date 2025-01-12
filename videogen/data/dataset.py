import h5py
import torch
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict
from intervaltree import IntervalTree
from einops import rearrange

import time


class VideoFrameDataset(Dataset):
    def __init__(self, h5_dir, selected_longplay_ids=[]):
        self.idx_range_path_map = IntervalTree()
        self.file_ptr_cache = {}
        
        total_frames = 0
        for path in sorted(Path(h5_dir).glob("*.h5")):
            longplay_id, video_id = path.stem.split('_')
            if len(selected_longplay_ids) != 0 and int(longplay_id) not in selected_longplay_ids:
                continue
            with h5py.File(path, 'r') as f:
                num_frames = f['video_frames'].attrs['total_frames']
                self.idx_range_path_map.addi(total_frames, total_frames + num_frames, str(path))
                total_frames += num_frames
        
        self.total_frames = total_frames

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        intervals = self.idx_range_path_map[idx]
        if not intervals:
            raise IndexError(f"Frame index {idx} out of bounds")
        if len(intervals) > 1:
            raise RuntimeError(f"Found overlapping video frames at index {idx}. This should never happen!")
        
        interval = intervals.pop()
        path = interval.data
        f = self.file_ptr_cache[path]
        frame_idx = idx - interval.begin
        frame = torch.from_numpy(f['video_frames'][frame_idx]).float()
        frame = rearrange(frame, 'h w c -> c h w')
        
        return frame
    
    # def __getitem__(self, idx):
    #     start = time.perf_counter()
        
    #     # Time interval lookup
    #     t0 = time.perf_counter()
    #     intervals = self.idx_range_path_map[idx]
    #     if not intervals:
    #         raise IndexError(f"Frame index {idx} out of bounds")
    #     if len(intervals) > 1:
    #         raise RuntimeError(f"Found overlapping video frames at index {idx}. This should never happen!")
    #     interval = intervals.pop()
    #     t_interval = time.perf_counter() - t0
        
    #     # Time H5 read
    #     t0 = time.perf_counter()
    #     path = interval.data
    #     f = self.file_ptr_cache[path]
    #     frame_idx = idx - interval.begin
    #     frame = torch.from_numpy(f['video_frames'][frame_idx]).float()
    #     t_read = time.perf_counter() - t0
        
    #     # Time reshape
    #     t0 = time.perf_counter()
    #     frame = rearrange(frame, 'h w c -> c h w')
    #     t_reshape = time.perf_counter() - t0
        
    #     total_time = time.perf_counter() - start
        
    #     if idx % 100 == 0:  # Log every 100th frame
    #         print(f"Frame {idx} timing:")
    #         print(f"  Interval lookup: {t_interval*1000:.2f}ms")
    #         print(f"  H5 read: {t_read*1000:.2f}ms")
    #         print(f"  Reshape: {t_reshape*1000:.2f}ms")
    #         print(f"  Total: {total_time*1000:.2f}ms")
        
    #     return frame

    def populate_worker_fp_cache(self):
        self.file_ptr_cache = {
            filepath: h5py.File(filepath, 'r')
            for filepath in {interval.data for interval in self.idx_range_path_map}
        }

#########
# utils #
#########

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.populate_worker_fp_cache()
    