import numpy as np
from pyspark import SparkFiles, SparkContext
from tempfile import mkstemp
from pathlib import Path
import os
from delex.utils.traits import SparkDistributable
from delex.utils import size_in_bytes
from typing import List

class PackedMemmapArrays(SparkDistributable):
    """
    a container for many MemmapArrays. used to store many MemmapArrays in a 
    single file
    """

    def __init__(self, arrs):
        self._shapes = [a.shape for a in arrs]
        self._dtypes = [a.dtype for a in arrs]
        self._offsets = np.cumsum([0] + [a.itemsize * a.size for a in arrs])
        self._total_bytes = self._offsets[-1]
        self._local_mmap_file = Path(mkstemp(suffix='.mmap_arr')[1])
        with self._local_mmap_file.open('wb') as ofs:
            for arr in arrs:
                ofs.write(memoryview(arr))

        self._mmap_arr = None
        self._on_spark = False

    def size_in_bytes(self) -> int:
        return size_in_bytes(self._local_mmap_file)

    def init(self):
        if self._mmap_arr is None:
            f = self._local_mmap_file
            if self._on_spark:
                f = SparkFiles.get(f.name)
                if not os.path.exists(f):
                    raise RuntimeError('cannot find database file at {f}')
            self._mmap_arr = np.memmap(f, mode='r', shape=self._total_bytes, dtype=np.uint8)

    def deinit(self):
        self._mmap_arr = None
        
    def unpack(self) -> List[np.ndarray]:
        """
        read all of the memmap arrays and return as a list
        """
        self.init()
        arrs = []
        f = self._local_mmap_file
        if self._on_spark:
            f = SparkFiles.get(f.name)

        for offset, shape, dtype in zip(self._offsets, self._shapes, self._dtypes):
            sz = dtype.itemsize
            for dim in shape:
                sz *= dim
            
            arr = np.memmap(f, offset=offset, mode='r', shape=shape, dtype=dtype)
            arrs.append(arr)

        return arrs

    def delete(self):
        if self._local_mmap_file.exists():
            self._local_mmap_file.unlink()

    def to_spark(self):
        if not self._on_spark:
            SparkContext.getOrCreate().addFile(str(self._local_mmap_file))
            self._on_spark = True
