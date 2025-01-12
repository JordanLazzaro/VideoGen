import numpy as np
import json
import lzf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import struct


class ChunkMetadata:
    idx: int
    byte_offset: int
    byte_size: int
    shape: Tuple[int, ...]
    dtype: str

    @property
    def original_byte_size(self) -> int:
        """Calculate original uncompressed size from shape and dtype"""
        return np.prod(self.shape) * np.dtype(self.dtype).itemsize


class ChunkStore:
    def __init__(self, filename: str, mode: str = 'r'):
        ''' 
        memory-mapped binary file containing a contiguous sequence of individually lzf-compressed chunks.

        Args:
            filename: Base name for the store files
            mode: Access mode
                'r': Read-only access to existing store
                'r+': Read-write access to existing store (append)
                'w+': Create new store, overwriting if exists
        '''
        if mode not in ('r', 'r+', 'w+'):
            raise ValueError("Mode must be 'r', 'r+', or 'w+'")
        self.mode = mode

        self.data_file = Path(str(filename) + '.bin')
        self.metadata_file = Path(str(filename) + '.meta.json')
        self.chunks_metadata = {}

        if mode == 'w':
            if self.data_file.exists() or self.metadata_file.exists():
                raise FileExistsError(
                    f"Files {self.data_file} or {self.metadata_file} already exist.")
            self.mmap = np.memmap(self.data_file, dtype='uint8', mode='w+', shape=(0,))
            self._save_metadata()
        else:
            if not self.data_file.exists():
                raise FileNotFoundError(f"File: {self.data_file} not found")
            if not self.metadata_file.exists():
                raise FileNotFoundError(f"File: {self.metadata_file} not found")
            self._load_metadata()
            file_size = self.data_file.stat().st_size
            self.mmap = np.memmap(self.data_file, dtype='uint8', mode=mode, shape=(file_size,))


    def append(self, chunk: np.ndarray) -> None:
        if not hasattr(self, 'mmap') or self.mmap.mode not in ('r+', 'w+'):
            raise ValueError("File not opened in write mode")

        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a numpy array")

        compressed_data = lzf.compress(frame.tobytes())
        compressed_size = len(compressed_data)
        
        current_offset = self.data_file.stat().st_size
        
        self.mmap.flush()
        self.mmap.base.resize(current_offset + compressed_size)
        
        np.copyto(self.mmap[current_offset:], np.frombuffer(compressed_data, dtype='uint8'))
        self.mmap.flush()
        
        metadata = ChunkMetadata(
            idx=self.next_index,
            offset=current_offset,
            compressed_size=compressed_size,
            shape=frame.shape,
            dtype=str(frame.dtype)
        )
        self.chunks_metadata.append(metadata)
        self.next_index += 1
        self._save_metadata()

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx not in self.chunks_metadata:
            raise IndexError(f"Chunk index {idx} not found")
        
        metadata = self.chunks_metadata[idx]
        compressed_data = self.mmap[metadata.offset:metadata.byte_offset + metadata.byte_size]
        decompressed_data = lzf.decompress(compressed_data.tobytes(), metadata.original_byte_size)
        frame = np.frombuffer(decompressed_data, dtype=metadata.dtype).reshape(metadata.shape)
        
        return frame

    def __len__(self) -> int:
        return len(self.chunks_metadata)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # don't suppress exceptions

    def close(self) -> None:
        if self.mmap is not None:
            self.mmap.flush()
            del self.mmap
            self.mmap = None
    
    def get_metadata(self, idx: int) -> Dict[str, Any]:
        if idx not in self.chunks_metadata:
            raise IndexError(f"Chunk index {idx} not found")
        return asdict(self.chunks_metadata[idx])

    def _load_metadata(self) -> None:
        with open(self.meta_file, 'r') as f:
            metadata_dict = json.load(f)
        
        self.next_index = metadata_dict.pop('__next_index__', 0)
        
        self.chunks_metadata = {
            int(idx): ChunkMetadata(
                idx=m['idx'],
                byte_offset=m['byte_offset'],
                byte_size=m['byte_size'],
                shape=tuple(m['shape']),
                dtype=m['dtype']
            )
            for idx, m in metadata_dict.items()
        }

    def _save_metadata(self) -> None:
        metadata_dict = {
            str(idx): {
                'idx': idx,
                'byte_offset': metadata.offset,
                'byte_size': metadata.compressed_size,
                'shape': metadata.shape,
                'dtype': metadata.dtype
            }
            for idx, metadata in self.chunks_metadata.items()
        }
        # save next_index to maintain continuity across sessions
        metadata_dict['__next_index__'] = self.next_index
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata_dict, f)