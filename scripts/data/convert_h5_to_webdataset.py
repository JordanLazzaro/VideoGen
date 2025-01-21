import os
import h5py
import tarfile
import numpy as np
import io
from PIL import Image

def h5_to_webdataset(in_h5_files, shard_prefix, shard_size=2e9):
    """
    - in_h5_files: list of paths to .h5 video files
    - shard_prefix: prefix for the output tar shards
    - shard_size: target size (in bytes) per shard
    """
    shard_idx = 0
    current_tar = None
    current_size = 0

    for h5_file_path in in_h5_files:
        with h5py.File(h5_file_path, 'r') as h5f:
            frames = h5f["frames"]  # shape (N, 256, 256, 1), dtype=uint8

            video_id = os.path.splitext(os.path.basename(h5_file_path))[0]
            for frame_i, frame in enumerate(frames):
                img = Image.fromarray(frame.squeeze(axis=-1), mode="L")  # "L" for 8-bit grayscale

                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes = img_bytes.getvalue()
                fsize = len(img_bytes)

                if current_tar is None or (current_size + fsize) > shard_size:
                    if current_tar is not None:
                        current_tar.close()
                    shard_name = f"{shard_prefix}-{shard_idx:05d}.tar"
                    current_tar = tarfile.open(shard_name, "w")
                    shard_idx += 1
                    current_size = 0

                arcname = f"{video_id}_{frame_i:06d}.png"

                tarinfo = tarfile.TarInfo(name=arcname)
                tarinfo.size = fsize
                current_tar.addfile(tarinfo, io.BytesIO(img_bytes))
                current_size += fsize

    if current_tar is not None:
        current_tar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_dir", type=str, default= '.', help="Directory containing H5 files")
    args = parser.parse_args()

    h5_paths = []
    for file in Path(args.h5_dir).glob('*.h5'):
        h5_paths.append(f'{args.h5_dir}/{file.name}')

    h5_to_webdataset(h5_paths)