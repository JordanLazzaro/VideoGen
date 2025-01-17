import torch
from torch.utils.data import Dataset
from pathlib import Path
from intervaltree import IntervalTree
from einops import rearrange
from videogamegen.data.chunkstore import ChunkStore


class VideoFrameDataset(Dataset):
    def __init__(self, chunk_store_dir, selected_longplay_ids=[]):
        self.idx_range_store_key_map = IntervalTree() # idx -> (store_key, store_path)
        self.store_ref_cache = {} # store_key -> store_ref

        total_frames = 0
        for path in sorted(Path(chunk_store_dir).iterdir()):
            if path.is_dir():
                longplay_id, video_id = path.name.split('.')
                if len(selected_longplay_ids) != 0 and int(longplay_id) not in selected_longplay_ids:
                    continue
                with ChunkStore(path, mode='r') as store:
                    num_frames = len(store)
                    store_key = f'{longplay_id}.{video_id}'
                    self.idx_range_store_key_map.addi(total_frames, total_frames + num_frames, (store_key, path))
                    total_frames += num_frames
        
        self.total_frames = total_frames

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        intervals = self.idx_range_store_key_map[idx]
        if not intervals:
            raise IndexError(f"Frame index {idx} out of bounds")
        if len(intervals) > 1:
            raise RuntimeError(f"Found overlapping video frames at index {idx}. This should never happen!")
        
        interval = intervals.pop()
        store_key, _ = interval.data
        store = self.store_ref_cache[store_key]
        frame_idx = idx - interval.begin
        frame = torch.from_numpy(store[frame_idx].copy()).float()
        frame = rearrange(frame, 'h w c -> c h w')
        
        return frame

    def populate_store_ref_cache(self):
        ''' each dataloader worker gets it's own store refs '''
        self.store_ref_cache = {
            store_key: ChunkStore(store_path, mode='r')
            for store_key, store_path in {interval.data for interval in self.idx_range_store_key_map}
        }

################
# worker utils #
################

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.populate_store_ref_cache()
    