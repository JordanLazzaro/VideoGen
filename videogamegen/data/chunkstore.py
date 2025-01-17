import numpy as np
import json
import lzf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import struct
import time


@dataclass
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
    def __init__(
        self,
        path: Union[str, Path],
        chunk_shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[Union[np.dtype, str]] = None,
        mode: str = 'r'
    ):
        ''' 
        memory-mapped binary file containing a contiguous sequence of individually lzf-compressed chunks.

        Args:
            path: Base path for the store files
            chunk_shape: Shape of a single chunk
            dtype: np.dtype of a chunk
            mode: Access mode
                'r': Read-only access to existing store
                'r+': Read-write access to existing store (append)
                'w+': Create new store, overwriting if exists
        '''
        if mode not in ('r', 'w+'):
            raise ValueError("Mode must be 'r', 'r+', or 'w+'")

        if mode == 'w+' and (chunk_shape is None or dtype is None):
            raise ValueError("chunk_shape and dtype are required for 'w+' mode")
       
        self.chunk_shape = chunk_shape
        self.dtype = dtype
        self.mode = mode
        
        self.base_dir = Path(path) if isinstance(path, str) else path
        self.data_file = self.base_dir / 'data.bin'
        self.metadata_file = self.base_dir / 'metadata.json'
        
        self.chunks_metadata = {}

        if mode == 'w+':
            if self.base_dir.exists():
                if not self.base_dir.is_dir():
                    raise FileExistsError(f"{self.base_dir} exists but is not a directory")
                if any(self.base_dir.iterdir()):
                    raise FileExistsError(f"Directory {self.base_dir} is not empty")
            else:
                self.base_dir.mkdir(parents=True)

            self.data_file.touch()
                
            self.next_index = 0
            self._save_metadata()
        else:
            if not self.base_dir.exists() or not self.base_dir.is_dir():
                raise FileNotFoundError(f"Directory {self.base_dir} not found")
            if not self.data_file.exists():
                raise FileNotFoundError(f"File: {self.data_file} not found")
            if not self.metadata_file.exists():
                raise FileNotFoundError(f"File: {self.metadata_file} not found")

            self._load_metadata()
            self._mmap = np.memmap(self.data_file, dtype='uint8', mode=mode, shape=(self.data_file.stat().st_size,))

    def append(self, chunks: Union[np.ndarray, List[np.ndarray]]) -> None:
        """
        Append multiple chunks using a single write operation.

        Args:
            chunks: np.ndarray, batched np.ndarray, or list of np.ndarray arrays to append
        """
        if self.mode not in ('w+', 'r+'):
            raise ValueError("File not opened in write/append mode")

        self._validate_chunks(chunks)

        if isinstance(chunks, np.ndarray):
            if chunks.shape == self.chunk_shape:
                chunk_list = [chunks]
            else:
                chunk_list = [chunks[i] for i in range(chunks.shape[0])]
        else:
            chunk_list = chunks
            
        current_offset = self.data_file.stat().st_size
        
        compressed_chunks = []
        for chunk in chunk_list:
            compressed_data = lzf.compress(chunk.tobytes())
            compressed_chunks.append(compressed_data)
            
            self.chunks_metadata[self.next_index] = ChunkMetadata(
                idx=self.next_index,
                byte_offset=current_offset,
                byte_size=len(compressed_data),
                shape=chunk.shape,
                dtype=str(chunk.dtype)
            )
            
            current_offset += len(compressed_data)
            self.next_index += 1
        
        with open(self.data_file, 'ab') as f:
            f.write(b''.join(compressed_chunks))
            
        self._save_metadata()

    # def __getitem__(self, idx: int) -> np.ndarray:
    #     if not isinstance(idx, int):
    #         raise TypeError("Index must be an integer")
    #     if idx not in self.chunks_metadata:
    #         raise IndexError(f"Chunk index {idx} not found")
    #     # if self._mmap is None:
    #     #     raise RuntimeError('mmap must be initialized')

    #     metadata = self.chunks_metadata[idx]
    #     # mmap = np.memmap(self.data_file, dtype='uint8', mode='r', shape=(self.data_file.stat().st_size,))
        
    #     # compressed_data = mmap[metadata.byte_offset:metadata.byte_offset + metadata.byte_size]
    #     compressed_data = self._mmap[metadata.byte_offset:metadata.byte_offset + metadata.byte_size]
    #     decompressed_data = lzf.decompress(compressed_data.tobytes(), metadata.original_byte_size)
    #     chunk = np.frombuffer(decompressed_data, dtype=metadata.dtype).reshape(metadata.shape)
        
    #     return chunk

    def __getitem__(self, idx: int) -> np.ndarray:
        start_total = time.perf_counter()
        
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer")
        if idx not in self.chunks_metadata:
            raise IndexError(f"Chunk index {idx} not found")
        
        metadata = self.chunks_metadata[idx]
        
        t1 = time.perf_counter()
        compressed_data = self._mmap[metadata.byte_offset:metadata.byte_offset + metadata.byte_size]
        t2 = time.perf_counter()
        
        decompressed_data = lzf.decompress(compressed_data.tobytes(), metadata.original_byte_size)
        t3 = time.perf_counter()
        
        chunk = np.frombuffer(decompressed_data, dtype=metadata.dtype).reshape(metadata.shape)
        t4 = time.perf_counter()
        
        print(f"Slice: {t2-t1:.6f}s, Decompress: {t3-t2:.6f}s, Reshape: {t4-t3:.6f}s, Total: {t4-start_total:.6f}s")
        
        return chunk

    def __len__(self) -> int:
        return len(self.chunks_metadata)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # don't suppress exceptions

    def close(self) -> None:
        pass  # no persistent resources to clean up
    
    def get_metadata(self, idx: int) -> Dict[str, Any]:
        if idx not in self.chunks_metadata:
            raise IndexError(f"Chunk index {idx} not found")
        return asdict(self.chunks_metadata[idx])

    def _load_metadata(self) -> None:
        with open(self.metadata_file, 'r') as f:
            metadata_dict = json.load(f)
        
        self.next_index = metadata_dict.pop('__next_index__', 0)
        self.chunk_shape = tuple(metadata_dict.pop('__chunk_shape__'))
        self.dtype = metadata_dict.pop('__dtype__')
        
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
            idx: {
                'idx': idx,
                'byte_offset': metadata.byte_offset,
                'byte_size': metadata.byte_size,
                'shape': metadata.shape,
                'dtype': metadata.dtype
            }
            for idx, metadata in self.chunks_metadata.items() if idx != '__next_index__'
        }
        # save next_index, chunk_shape, and dtype to maintain continuity across sessions
        metadata_dict['__next_index__'] = self.next_index
        metadata_dict['__chunk_shape__'] = self.chunk_shape
        metadata_dict['__dtype__'] = str(self.dtype)
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata_dict, f)

    def _validate_chunks(self, chunks: Union[np.ndarray, List[np.ndarray]]) -> None:
        if chunks is None:
            raise ValueError("chunks cannot be None")
        
        if not isinstance(chunks, (np.ndarray, list)):
            raise TypeError(
                "chunks must be of type np.ndarray or List[np.ndarray], "
                f"got {type(chunks).__name__}"
            )
        if isinstance(chunks, np.ndarray):
            is_single = chunks.shape == self.chunk_shape
            is_batch = (len(chunks.shape) == len(self.chunk_shape) + 1 and 
                    chunks.shape[1:] == self.chunk_shape)
                    
            if not (is_single or is_batch):
                raise ValueError(
                    f"Invalid shape {chunks.shape}. Expected {self.chunk_shape} "
                    f"or (N,{','.join(str(d) for d in self.chunk_shape)})"
                )
                
            if is_batch and chunks.shape[0] == 0:
                raise ValueError("Cannot append empty batch")
                
            if chunks.dtype != self.dtype:
                raise TypeError(f"Expected dtype {self.dtype}, got {chunks.dtype}")
                
        else:
            if not chunks:
                raise ValueError("Cannot append empty list of chunks")
                
            if not all(isinstance(chunk, np.ndarray) for chunk in chunks):
                raise TypeError("All chunks must be of type np.ndarray")
                
            if any(chunk.shape != self.chunk_shape for chunk in chunks):
                raise ValueError(f"All chunks must have shape {self.chunk_shape}")
                
            if any(chunk.dtype != self.dtype for chunk in chunks):
                raise TypeError(f"All chunks must have dtype {self.dtype}")