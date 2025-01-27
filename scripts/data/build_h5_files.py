import av
import cv2
import numpy as np
import h5py
import os
import concurrent.futures
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import List
from pathlib import Path

mp4_files = []
for file in Path('./gameboy_longplays_mp4').glob('*.mp4'):
    mp4_files.append(f'./gameboy_longplays_mp4/{file.name}')

def decode_video_to_hdf5_chunked(
    video_path: str,
    output_dir: str,
    target_size: tuple = (256, 256),
    chunk_size: int = 16,
    dataset_name: str = "video_frames",
    compression: str = "gzip",
    compression_opts: int = 4
):
    """
    Decodes a single MP4 file in 'chunk_size' batches and writes frames to an HDF5 file.
    
    Args:
        video_path (str): Path to the input MP4 file.
        output_dir (str): Directory where the per-video HDF5 file will be created.
        chunk_size (int): Number of frames to read and write at a time (batch size).
        dataset_name (str): Name of the dataset inside the HDF5 file.
        compression (str): HDF5 compression algorithm (e.g., 'gzip', 'lzf', or None).
        compression_opts (int): Compression level (if using 'gzip', 1-9).
    """
    longplay_id, video_id, _ = os.path.basename(video_path).split('_', maxsplit=2)
    hdf5_out_path = os.path.join(output_dir, f"{longplay_id}_{video_id}.h5")
    
    def process_frame(frame):
        arr = frame.to_ndarray(format='rgb24')
        if not np.any(arr):
            return
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return cv2.resize(arr, target_size)
    
    def write_batch(f, dset, batch_frames, frame_count):
        batch_array = np.stack(batch_frames, axis=0)
        if dset is None:
            height, width = batch_array.shape[1:3]
            dset = f.create_dataset(
                dataset_name,
                shape=(0, height, width, 1),  # since it's grayscale
                maxshape=(None, height, width, 1),
                dtype=batch_array.dtype,
                chunks=(chunk_size, height, width, 1),
                compression=compression,
                compression_opts=compression_opts
            )
        
        dset.resize(dset.shape[0] + batch_array.shape[0], axis=0)
        dset[-batch_array.shape[0]:] = batch_array[..., np.newaxis]  # add channel dimension
        frame_count += batch_array.shape[0]
        return dset, frame_count

    def get_total_frames(container):
        return container.streams.video[0].frames

    with av.open(video_path) as container:
        with h5py.File(hdf5_out_path, "w") as f:
            dset = None
            frame_count = 0
            batch_frames = []
            
            for frame in container.decode(video=0):
                frame_arr = process_frame(frame)
                if frame_arr:
                    batch_frames.append(frame_arr)
                
                if len(batch_frames) == chunk_size:
                    dset, frame_count = write_batch(f, dset, batch_frames, frame_count)
                    batch_frames = []
            
            # write any remaining frames
            if batch_frames:
                dset, frame_count = write_batch(f, dset, batch_frames, frame_count)
            
            if dset is not None:
                dset.attrs["video_path"] = video_path
                
                # original video properties
                stream = container.streams.video[0]
                dset.attrs["original_width"] = stream.width
                dset.attrs["original_height"] = stream.height
                dset.attrs["fps"] = float(stream.average_rate)
                dset.attrs["duration_seconds"] = float(stream.duration * stream.time_base)
                dset.attrs["total_frames"] = stream.frames
                
                # processing parameters
                dset.attrs["target_size"] = target_size
                dset.attrs["compression"] = compression
                dset.attrs["compression_level"] = compression_opts
                dset.attrs["chunk_size"] = chunk_size
        
                

def decode_videos_in_parallel_chunked(
    video_paths: List[str],
    output_dir: str,
    target_size: tuple = (256, 256),
    chunk_size: int = 16,
    max_workers: int = 4
):
    """
    Decodes multiple MP4 files in parallel, each to its own HDF5 file using chunked writes.
    
    Args:
        video_paths (List[str]): List of MP4 paths to decode.
        output_dir (str): Directory where per-video HDF5 files will be placed.
        chunk_size (int): Number of frames to read/write at a time for each video.
        max_workers (int): Number of parallel processes.
    """
    os.makedirs(output_dir, exist_ok=True)
    total_videos = len(video_paths)
    
    with tqdm(total=total_videos, desc="Processing videos", unit="video", position=0) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(decode_video_to_hdf5_chunked, path, output_dir, target_size, chunk_size): path
                for path in video_paths
            }
            
            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    future.result()
                    pbar.update(1)
                except Exception as exc:
                    print(f"\nError decoding {path}: {exc}")
                    pbar.update(1)

if __name__ == '__main__':
    decode_videos_in_parallel_chunked(
        [f for f in mp4_files if '289_0_' in f],
        output_dir="data/longplay_h5_files",
        chunk_size=16,
        max_workers=32
    )