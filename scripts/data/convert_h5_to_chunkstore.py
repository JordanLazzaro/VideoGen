import h5py
import argparse
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
from videogen.data.chunkstore import ChunkStore


def convert_h5_to_chunkstore(h5_path: str, batch_size: int = 1024) -> None:
    """
    Sequential H5 to ChunkStore converter using batched operations.
    
    Args:
        h5_path: Path to input HDF5 file
        batch_size: Number of frames to process in each batch
    """
    longplay_id, video_id = Path(h5_path).stem.split('_')
    store_path = Path('data/chunkstore') / f'{longplay_id}.{video_id}'
    store_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(h5_path, 'r') as f:
        dset = f['video_frames']
        chunk_shape = dset.shape[1:]
        dtype = dset.dtype
        total_frames = dset.shape[0]
        with ChunkStore(store_path, chunk_shape=chunk_shape, dtype=dtype, mode='w+') as store:
            with tqdm(total=total_frames, desc="Converting frames", unit="frames") as pbar:
                for start_idx in range(0, total_frames, batch_size):
                    end_idx = min(start_idx + batch_size, total_frames)
                    frames = dset[start_idx:end_idx]
                    store.append(frames)
                    pbar.update(len(frames))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, required=True, help="Path to H5 file")
    parser.add_argument("--batch_size", type=int, default=1024, help="Number of frames to process in each batch")
    args = parser.parse_args()
    
    convert_h5_to_chunkstore(args.h5_path, batch_size=args.batch_size)