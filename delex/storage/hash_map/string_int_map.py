import numpy as np
import numba as nb
from delex.utils import HashFunction
from .hash_map_base import DistributableHashMap

map_entry_128_t = np.dtype([('hash1', np.uint64), ('hash2', np.uint64), ('val', np.int32)])
numba_map_entry_128_t = nb.from_dtype(map_entry_128_t)

njit_kwargs = {
        'cache' : False, 
        'parallel' : False
}

@nb.njit(**njit_kwargs)
def _get_probe_stride(val):
    # for now just do linear probing
    return 1

@nb.njit(**njit_kwargs)
def _hash_map_insert_key_128(arr, hash1, hash2, val):
    if hash1 == 0 and hash2 == 0:
        raise ValueError('keys must be non zero')
    
    i = hash1 % len(arr)
    inc = _get_probe_stride(hash1)
    for tries in range(len(arr)):
        if (arr[i].hash1 == 0 and arr[i].hash2 == 0): 
            arr[i].hash1 = hash1
            arr[i].hash2 = hash2
            arr[i].val = val
            return True
        elif (arr[i].hash1 == hash1 and arr[i].hash2 == hash2):
            arr[i].val = val
            return False
        else:
            i = (i + inc) % len(arr)
    else:
        raise RuntimeError('no spots found for key')

@nb.njit(**njit_kwargs)
def _hash_map_insert_keys_128(arr, keys, vals):
    for i in range(len(keys)):
        is_new = _hash_map_insert_key_128(arr, keys[i, 0], keys[i, 1], vals[i])
        if not is_new:
            raise ValueError('keys not unique')



@nb.njit(**njit_kwargs)
def _hash_map_get_key_128(arr, hash1, hash2):
    if hash1 == 0:
        raise ValueError('keys must be non zero')
    
    i = hash1 % len(arr)
    inc = _get_probe_stride(hash1)
    for tries in range(len(arr)):
        if (arr[i].hash1 == 0 and arr[i].hash2 == 0):
            return -1
        elif (arr[i].hash1 == hash1 and arr[i].hash2 == hash2):
            return arr[i].val 
        else:
            i = (i + inc) % len(arr)
    else:
        raise RuntimeError('key not found but max number of tries exceeded')


@nb.njit(**njit_kwargs)
def _hash_map_get_keys_128(arr, keys):
    out = np.empty(len(keys), dtype=np.int32)
    for i in range(len(keys)):
        out[i] = _hash_map_get_key_128(arr, keys[i, 0], keys[i, 1])
    return out

class StringIntHashMap(DistributableHashMap):

    def __init__(self, arr, hash_func):
        self._hash_func = hash_func
        super().__init__(arr)

    @classmethod
    def build(cls, strings, ints, load_factor=.5):
        for i in range(10):
            hash_func = HashFunction()
            hashes = np.fromiter(map(hash_func.hash, strings), dtype=np.void(16), count=len(strings))\
                        .view(np.uint64)\
                        .reshape(len(strings), 2)

            arr = cls._allocate_map(len(strings), load_factor, map_entry_128_t)
            try:
                _hash_map_insert_keys_128(arr, hashes, ints)
            except ValueError:
                continue
            else:
                break
        else:
            raise RuntimeError('unable to find hash function')

        return cls(arr, hash_func)

    def __getitem__(self, keys):
        if isinstance(keys, str):
            h1, h2 = self._hash_func.hash_split(keys)
            i = _hash_map_get_key_128(self._arr, np.uint64(h1), np.uint64(h2))
            return i
        else:
            hashes = np.fromiter(map(self._hash_func.hash, keys), count=len(keys), dtype=np.void(16))\
                            .view(np.uint64)\
                            .reshape(-1, 2)
            return _hash_map_get_keys_128(self._arr, hashes)
