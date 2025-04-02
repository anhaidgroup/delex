import numpy as np
import numba as nb

span_map_entry_t = np.dtype([('hash', np.int32), ('offset', np.int32), ('len', np.int16)])

njit_kwargs = {
        'cache' : False, 
        'parallel' : False
}
@nb.njit(**njit_kwargs)
def create_span_map(keys, offsets, lengths, load_factor=.75):
    """
    create a new span map of for `keys`, `offsets`, and `lengths`

    Returns
    -------
    np.ndarray
    """
    if len(keys) != len(offsets):
        raise ValueError()
    arr = np.zeros(np.int32(len(keys) / load_factor), dtype=span_map_entry_t)
    
    for i in range(len(arr)):
        arr[i].hash = -1

    span_map_insert_keys(arr, keys, offsets, lengths)
    return arr

@nb.njit(**njit_kwargs)
def span_map_insert_key(arr, key, offset, length):
    """
    insert a single key into the span_map `arr`
    """
    if key <= -1:
        raise ValueError('keys must be non-negative')

    i = key % len(arr)
    while True:
        if arr[i].hash == -1 or arr[i].hash == key:
            arr[i].hash = key
            arr[i].offset = offset
            arr[i].len = length
            return 
        else:
            i += 1
            if i == len(arr):
                i = 0

@nb.njit(**njit_kwargs)
def span_map_insert_keys(arr, keys, offsets, lengths):
    """
    insert many keys into the span_map `arr`
    """
    for i in range(len(keys)):
        span_map_insert_key(arr, keys[i], offsets[i], lengths[i])


@nb.njit(**njit_kwargs)
def span_map_get_key(arr, key):
    """
    get the entry from the span map, return the offset and length as a tuple
    """
    i = key % len(arr)
    while True:
        if arr[i].hash == key:
            # hash found, return value at position
            return arr[i].offset, arr[i].len
        elif arr[i].hash == -1:
            # hash not found, return -1
            return np.int32(-1), np.int16(0)
        else:
            i += 1
            if i == len(arr):
                i = 0
