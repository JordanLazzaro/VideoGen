import av
import cv2
import numpy as np
import os
import tarfile
import concurrent.futures
from tqdm.auto import tqdm
from typing import List, Optional
from pathlib import Path
import io
import json

def process_video(
    video_path: str,
    output_dir: str,
    target_size: tuple = (256, 256, 1),
    target_fps: Optional[float] = None,
    chunk_size: int = 16,
    compression: int = 5
):
    """
    Processes a video file and saves frame chunks as compressed numpy arrays in a tar file.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory for output tar files
        target_size (tuple): Target dimensions for frame resizing
        target_fps (float, optional): Desired output frame rate. If None, keeps original fps
        chunk_size (int): Number of frames per chunk
        compression (int): Numpy compression level (1-9)
    """
    longplay_id, video_id, _ = os.path.basename(video_path).split('_', maxsplit=2)
    tar_path = os.path.join(output_dir, f"{longplay_id}_{video_id}_%06d.tar" % 0)
    
    def process_frame(frame):
        """Convert video frame to grayscale and resize"""
        arr = frame.to_ndarray(format='rgb24')
        if not np.any(arr):
            return None
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return cv2.resize(arr, target_size)
    
    def save_chunk(tar, frames, chunk_idx):
        """Save a chunk of frames as a compressed numpy array in the tar file"""
        if not frames:
            return
            
        chunk_array = np.stack(frames, axis=0)[..., np.newaxis]
        
        buffer = io.BytesIO()
        np.savez_compressed(buffer, frames=chunk_array)
        buffer.seek(0)
        
        info = tarfile.TarInfo(f"{chunk_idx:06d}.npy")
        info.size = len(buffer.getvalue())
        tar.addfile(info, buffer)
        
        return chunk_array.shape[0]

    with av.open(video_path) as container:
        video_stream = container.streams.video[0]
        original_fps = float(video_stream.average_rate)
        
        if target_fps is not None and target_fps != original_fps:
            resampler = av.VideoResampler(rate=f"{target_fps}/1")
            base_time = video_stream.start_time or 0
            frame_iterator = resampler.resample(container.decode(video=0))
        else:
            frame_iterator = container.decode(video=0)
            target_fps = original_fps
        
        with tarfile.open(tar_path, "w") as tar:
            metadata = {
                "video_path": video_path,
                "original_width": video_stream.width,
                "original_height": video_stream.height,
                "original_fps": original_fps,
                "target_fps": target_fps,
                "duration_seconds": float(video_stream.duration * video_stream.time_base),
                "total_frames": video_stream.frames,
                "target_size": target_size,
                "chunk_size": chunk_size,
                "compression": compression
            }
            
            meta_buffer = io.BytesIO()
            meta_buffer.write(json.dumps(metadata).encode('utf-8'))
            meta_buffer.seek(0)
            
            meta_info = tarfile.TarInfo("_metadata.json")
            meta_info.size = len(meta_buffer.getvalue())
            tar.addfile(meta_info, meta_buffer)
            
            frames = []
            chunk_idx = 0
            total_saved_frames = 0
            
            for frame in frame_iterator:
                frame_arr = process_frame(frame)
                if frame_arr is not None:
                    frames.append(frame_arr)
                
                if len(frames) == chunk_size:
                    frames_saved = save_chunk(tar, frames, chunk_idx)
                    if frames_saved:
                        total_saved_frames += frames_saved
                        chunk_idx += 1
                    frames = []
            
            if frames:
                frames_saved = save_chunk(tar, frames, chunk_idx)
                if frames_saved:
                    total_saved_frames += frames_saved

def process_videos_parallel(
    video_paths: List[str],
    output_dir: str,
    target_size: tuple = (256, 256, 1),
    target_fps: Optional[float] = None,
    chunk_size: int = 1024,
    max_workers: int = 4
):
    """
    Process multiple videos in parallel, saving frame chunks to tar files.
    
    Args:
        video_paths (List[str]): List of video paths to process
        output_dir (str): Output directory for tar files
        target_size (tuple): Target dimensions for frame resizing
        target_fps (float, optional): Desired output frame rate. If None, keeps original fps
        chunk_size (int): Number of frames per chunk
        max_workers (int): Number of parallel workers
    """
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
                    chunk_size
                ): path for path in video_paths
            }
            
            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"\nError processing {path}: {exc}")
                finally:
                    pbar.update(1)

if __name__ == '__main__':
    mp4_files = list(Path('./gameboy_longplays_mp4').glob('*.mp4'))
    
    process_videos_parallel(
        [str(f) for f in mp4_files],
        output_dir="data/longplay_tar_files",
        target_fps=20.0
    )