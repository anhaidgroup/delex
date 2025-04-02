import numpy as np
import numba as nb
from .hash_map_base import DistributableHashMap

map_entry_t = np.dtype([('hash', np.uint64), ('val', np.int32)])
numba_map_entry_t = nb.from_dtype(map_entry_t)

njit_kwargs = {
        'cache' : False, 
        'parallel' : False
}

# adapted from this blog post 
# https://bitsquid.blogspot.com/2011/08/code-snippet-murmur-hash-inverse-pre.html
@nb.njit(**njit_kwargs)
def _murmurhash_64(val, seed=0):
    val = np.uint64(val)
    m = np.uint64(0xc6a4a7935bd1e995)
    r = 47;

    h = seed ^ (8 * m);

    k = val

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
 
    h ^= h >> r
    h *= m
    h ^= h >> r

    return h

@nb.njit(**njit_kwargs)
def _get_probe_stride(val):
    # for now just do linear probing
    return 1

@nb.njit(**njit_kwargs)
def hash_map_insert_key(arr, key, val):
    if key == 0:
        raise ValueError('keys must be non zero')
    
    i = _murmurhash_64(key) % len(arr)
    inc = _get_probe_stride(key)
    for tries in range(len(arr)):
        if arr[i].hash == 0:
            arr[i].hash = key
            arr[i].val = val
            return True
        elif arr[i].hash == key:
            arr[i].val = val
            return False
        else:
            i = (i + inc) % len(arr)
    else:
        raise RuntimeError('no spots found for key')

@nb.njit(**njit_kwargs)
def hash_map_insert_keys(arr, keys, vals):
    for i in range(len(keys)):
        is_new = hash_map_insert_key(arr, keys[i], vals[i])
        if not is_new:
            raise ValueError('keys not unique')

@nb.njit(**njit_kwargs)
def hash_map_get_key(arr, key):
    i = _murmurhash_64(key) % len(arr)
    inc = _get_probe_stride(key)

    for tries in range(len(arr)):
        if arr[i].hash == key:
            # hash found, return value at position
            return arr[i].val 
        elif arr[i].hash == 0:
            # hash not found, return -1
            return -1
        else:
            i = (i + inc) % len(arr)
    else:
        raise RuntimeError('key not found but max number of tries exceeded')


@nb.njit(**njit_kwargs)
def hash_map_get_keys(arr, keys):
    out = np.empty(len(keys), dtype=np.int32)
    for i in range(len(keys)):
        out[i] = hash_map_get_key(arr, keys[i])
    return out

class IdOffsetHashMap(DistributableHashMap):

    def __init__(self, arr):
        super().__init__(arr)
    
    # +1 to allow for 0 keys since this is usually used to map and id to an int
    @classmethod
    def build(cls, longs, ints, load_factor=.5):
        arr = cls._allocate_map(len(longs), load_factor, map_entry_t)
        hash_map_insert_keys(arr, longs+1, ints)

        return cls(arr)

    def __getitem__(self, keys):
        if isinstance(keys, (np.uint64, np.int64, int, np.uint32, np.int32)):
            return hash_map_get_key(self._arr, np.uint64(keys+1))
        elif isinstance(keys, np.ndarray):
            return hash_map_get_keys(self._arr, keys+1)
        else:
            raise TypeError(f'unknown type {type(keys)}')
