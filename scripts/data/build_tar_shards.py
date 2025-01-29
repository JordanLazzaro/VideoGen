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
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return cv2.resize(arr, target_size)

def save_shard(tar: tarfile.TarFile, frames: List[np.ndarray], shard_idx: int) -> int:
    """Save a shard of frames to the tar file"""
    if not frames:
        return 0
        
    shard_array = np.stack(frames, axis=0)[..., np.newaxis]
    
    buffer = io.BytesIO()
    np.savez_compressed(buffer, frames=shard_array)
    buffer.seek(0)
    
    info = tarfile.TarInfo(f"{shard_idx:06d}.npy")
    info.size = len(buffer.getvalue())
    tar.addfile(info, buffer)
    
    return shard_array.shape[0]

def get_video_metadata(container: av.container.Container, target_fps: Optional[float] = None) -> dict:
    """Extract video metadata"""
    stream = container.streams.video[0]
    stream_time_base = float(stream.time_base)
    original_fps = float(stream.average_rate)
    
    return {
        "original_width": stream.width,
        "original_height": stream.height,
        "original_fps": original_fps,
        "target_fps": target_fps if target_fps is not None else original_fps,
        "duration_seconds": float(stream.duration * stream_time_base),
        "total_frames": stream.frames
    }

def process_video(
    video_path: str,
    output_dir: str,
    target_size: tuple = (256, 256, 1),
    target_fps: Optional[float] = None,
    shard_size: int = 4096
):
    """Process a video file and save frame shards to a tar file"""
    longplay_id, video_id, _ = os.path.basename(video_path).split('_', maxsplit=2)
    tar_path = os.path.join(output_dir, f"{longplay_id}_{video_id}_%06d.tar" % 0)
    
    with av.open(video_path) as container:
        metadata = {
            "video_path": video_path,
            "target_size": target_size,
            "shard_size": shard_size,
            **get_video_metadata(container, target_fps)
        }
        
        with tarfile.open(tar_path, "w") as tar:
            meta_buffer = io.BytesIO(json.dumps(metadata).encode('utf-8'))
            meta_info = tarfile.TarInfo("_metadata.json")
            meta_info.size = len(meta_buffer.getvalue())
            tar.addfile(meta_info, meta_buffer)
            
            frames = []
            shard_idx = 0
            
            for frame in read_frames(container, target_fps):
                frame_arr = process_frame(frame, target_size)
                if frame_arr is not None:
                    frames.append(frame_arr)
                    
                    if len(frames) == shard_size:
                        save_shard(tar, frames, shard_idx)
                        shard_idx += 1
                        frames = []
            
            if frames:
                save_shard(tar, frames, shard_idx)

def process_videos_parallel(
    video_paths: List[str],
    output_dir: str,
    target_size: tuple = (256, 256, 1),
    target_fps: Optional[float] = None,
    shard_size: int = 4096,
    max_workers: int = 4
):
    """Process multiple videos in parallel"""
    os.makedirs(output_dir, exist_ok=True)
    
    with tqdm(total=len(video_paths), desc="Processing videos", unit="video") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_video, 
                    path, 
                    output_dir, 
                    target_size, 
                    target_fps,
                    shard_size
                ): path for path in video_paths
            }
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"\nError processing {futures[future]}: {exc}")
                finally:
                    pbar.update(1)

if __name__ == '__main__':
    mp4_files = list(Path('./gameboy_longplays_mp4').glob('*.mp4'))
    
    process_videos_parallel(
        [str(f) for f in mp4_files],
        output_dir="data/longplay_tar_files",
        target_fps=30.0,
        shard_size=1024,
        max_workers=32
    )