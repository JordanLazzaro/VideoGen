import os
import argparse
import numpy as np
import h5py
import concurrent.futures
from pathlib import Path
from tqdm.auto import tqdm


def rechunk_dataset(filename, dataset_name, new_chunk_size):
    with h5py.File(filename, 'r+') as f:
        # Get original dataset
        old_dset = f[dataset_name]
        
        # Store original attributes and dtype
        attrs = dict(old_dset.attrs)
        dtype = old_dset.dtype
        shape = old_dset.shape
        
        # Create temporary dataset name
        temp_name = dataset_name + '_temp'
        
        # Create new dataset with desired chunk size
        new_dset = f.create_dataset(
            temp_name,
            shape=shape,
            dtype=dtype,
            chunks=(new_chunk_size,) + shape[1:],  # Assuming first dim is frames
            compression=old_dset.compression,
            compression_opts=old_dset.compression_opts
        )
        
        # Copy data in reasonable block sizes
        block_size = 4096  # Adjust based on memory constraints
        for i in range(0, shape[0], block_size):
            end = min(i + block_size, shape[0])
            new_dset[i:end] = old_dset[i:end]
        
        # Copy attributes
        for key, value in attrs.items():
            new_dset.attrs[key] = value
        
        # Delete old dataset
        del f[dataset_name]
        
        # Rename new dataset to original name
        f[dataset_name] = f[temp_name]
        del f[temp_name]

def rechunk_dataset_parallel(
    h5_paths,
    max_workers=32
):
    total_videos = len(h5_paths)
    
    with tqdm(total=total_videos, desc="Rechunking H5 Files", unit="file", position=0) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(rechunk_dataset, path, 'video_frames', new_chunk_size=1): path
                for path in h5_paths
            }
            
            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    future.result()
                    pbar.update(1)
                except Exception as exc:
                    print(f"\nError decoding {path}: {exc}")
                    pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_dir", type=str, default= '.', help="Directory containing H5 files")
    args = parser.parse_args()

    h5_paths = []
    for file in Path(args.h5_dir).glob('*.h5'):
        h5_paths.append(f'{args.h5_dir}/{file.name}')

    rechunk_dataset_parallel(h5_paths=h5_paths)