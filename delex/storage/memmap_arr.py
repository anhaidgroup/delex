import numpy as np
from pyspark import SparkFiles, SparkContext
from tempfile import mkstemp
from pathlib import Path
import os
from delex.utils.traits import SparkDistributable
from delex.utils import size_in_bytes



class MemmapArray(SparkDistributable):

    def __init__(self, arr):
        self._dtype = arr.dtype
        self._shape = arr.shape
        
        if isinstance(arr, np.memmap):
            self._local_mmap_file = Path(arr.filename)
            arr.flush()
            self._mmap_arr = arr
        else:
            self._local_mmap_file = Path(mkstemp(suffix='.mmap_arr')[1])
            self._mmap_arr = np.memmap(self._local_mmap_file, shape=self._shape, dtype=self._dtype, mode='w+')
            self._mmap_arr[:] = arr[:]
            self._mmap_arr.flush()

        self._on_spark = False
    
    def size_in_bytes(self):
        return size_in_bytes(self._local_mmap_file)

    def __reduce__(self):
        self._mmap_arr = None
        return super().__reduce__()

    @property
    def values(self):
        return self._mmap_arr

    @property
    def shape(self):
        return self._shape
    
    def __len__(self):
        return self._shape[0]

    def init(self):
        if self._mmap_arr is None:
            f = self._local_mmap_file
            if self._on_spark:
                f = SparkFiles.get(f.name)
                if not os.path.exists(f):
                    raise RuntimeError('cannot find database file at {f}')
            self._mmap_arr = np.memmap(f, mode='r', shape=self._shape, dtype=self._dtype)
    
    def deinit(self):
        self._mmap_arr = None

    def delete(self):
        if self._local_mmap_file.exists():
            self._local_mmap_file.unlink()

    def to_spark(self):
        if not self._on_spark:
            SparkContext.getOrCreate().addFile(str(self._local_mmap_file))
            self._on_spark = True
