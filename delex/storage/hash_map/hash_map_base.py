import numpy as np
from delex.storage import MemmapArray
from delex.utils.traits import SparkDistributable



class DistributableHashMap(SparkDistributable):

    def __init__(self, arr):
        self._memmap_arr = MemmapArray(arr)
    
    def size_in_bytes(self) -> int:
        return self._memmap_arr.size_in_bytes()

    @property
    def _arr(self):
        return self._memmap_arr.values

    @property
    def on_spark(self):
        return self._memmap_arr.on_spark

    def init(self):
        self._memmap_arr.init()

    def deinit(self):
        self._memmap_arr.deinit()

    def to_spark(self):
        self._memmap_arr.to_spark()

    @staticmethod
    def _allocate_map(nkeys, load_factor, dtype):
        map_size = int(nkeys / load_factor)
        if map_size % 2 == 0:
            map_size += 1
        arr = np.zeros(map_size, dtype=dtype)
        return arr


