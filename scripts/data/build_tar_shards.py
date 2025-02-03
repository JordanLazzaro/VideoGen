import av
import cv2
import numpy as np
import os
import tarfile
import concurrent.futures
from tqdm.auto import tqdm
from typing import List, Optional, Iterator, Tuple
from pathlib import Path
import io
import json
import argparse

def read_frames(container: av.container.Container, target_fps: Optional[float] = None) -> Iterator[av.frame.Frame]:
    """Generator that yields frames at the desired frame rate.
    
    Args:
        container: PyAV container with video stream
        target_fps: Desired output frame rate. If None, keeps original fps.
                   Must be lower than original fps.
    """
    stream = container.streams.video[0]
    original_fps = float(stream.average_rate)
    
    if target_fps is None or target_fps == original_fps:
        yield from container.decode(video=0)
        return

    nth_frame = round(original_fps / target_fps)
    
    for i, frame in enumerate(container.decode(video=0)):
        if i % nth_frame == 0:
            yield frame

def process_frame(frame: av.frame.Frame, target_size: Tuple[int, ...]) -> np.ndarray:
    """Convert a frame to grayscale and resize it"""
    arr = frame.to_ndarray(format='rgb24')
    if not np.any(arr):
        return None
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)[..., np.newaxis]
    return cv2.resize(arr, target_size[:2])

def save_shard(tar: tarfile.TarFile, frames: List[np.ndarray], shard_idx: int) -> int:
    """Save a shard of frames to the tar file"""
    if not frames:
        return 0
        
    shard_array = np.stack(frames, axis=0)[..., np.newaxis]
    
    buffer = io.BytesIO()
    np.savez_compressed(buffer, frames=shard_array)
    buffer.seek(0)
    
    meta = {
        "shape": shard_array.shape,
        "dtype": str(shard_array.dtype)
    }
    
    meta_buffer = io.BytesIO(json.dumps(meta).encode('utf-8'))
    meta_info = tarfile.TarInfo(f"{shard_idx:08d}.json")
    meta_info.size = len(meta_buffer.getvalue())
    tar.addfile(meta_info, meta_buffer)
    
    return shard_array.shape[0]

def process_video(
    video_path: str,
    output_dir: str,
    target_size: tuple = (256, 256, 1),
    target_fps: Optional[float] = None,
    shard_size: int = 4096,
    dataset_name: str = 'gameboy-longplays',
    num_threads: int = 4
):
    """Process a video file and save frame shards to a tar file"""
    os.makedirs(output_dir, exist_ok=True)
    with av.open(video_path, options={'threads': str(num_threads), 'thread_type': 'frame'}) as container:
        frames = []
        shard_idx = 0
        for frame in read_frames(container, target_fps):
            frame_arr = process_frame(frame, target_size)
            if frame_arr is not None:
                frames.append(frame_arr)
                
                if len(frames) == shard_size:
                    tar_path = os.path.join(output_dir, f"{dataset_name}-{shard_idx:08d}.tar")
                    with tarfile.open(tar_path, "w") as tar:
                        save_shard(tar, frames, shard_idx)
                    frames = []
                    shard_idx += 1
        
        if frames:
            tar_path = os.path.join(output_dir, f"{dataset_name}-{shard_idx:08d}.tar")
            with tarfile.open(tar_path, "w") as tar:
                save_shard(tar, frames, shard_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video files into WebDataset format')
    parser.add_argument('--input_dir', help='Directory containing MP4 files')
    parser.add_argument('--output_dir', default='data/longplay_tar_files', 
                       help='Output directory for tar files')
    parser.add_argument('--target_fps', type=float, default=20.0,
                       help='Target FPS for processed videos')
    parser.add_argument('--shard_size', type=int, default=4096,
                       help='Number of frames per shard')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker processes')
    
    args = parser.parse_args()
    
    mp4_files = list(Path(args.input_dir).glob('*.mp4'))
    if not mp4_files:
        print(f"No MP4 files found in {args.input_dir}")
        exit(1)

    for f in tqdm(mp4_files, desc='Processing videos into tar files'):
        process_video(
            video_path=f,
            output_dir=args.output_dir,
            target_fps=args.target_fps,
            shard_size=args.shard_size
        )