import pyspark.sql.functions as F
import pyspark.sql.types as T
import pyspark
from typing import Iterator, Union
import numpy as np
from tempfile import mkstemp
from pathlib import Path
from delex.storage import MemmapArray, IdOffsetHashMap
import numba as nb
from delex.utils.numba_functions import sorted_set_overlap
from delex.storage.memmap_seqs import MemmapSeqs
import pandas as pd
from pydantic.dataclasses import dataclass
from delex.utils import CachedObjectKey
from delex.utils.funcs import type_check_call
from queue import Queue, Full
from threading import Thread, Event
from contextlib import contextmanager

_VECTOR_DTYPE = np.dtype([('ind', 'i4'), ('val', 'f4')])

@contextmanager
def iter_spark_rows(df, prefetch_size: int):
    count = df.count()
    local_iterator = df.toLocalIterator(True)
    stop_event = Event()
    
    queue = Queue(prefetch_size)
    thread = Thread( target=_prefetch_rows, args=(local_iterator, queue, stop_event) )
    thread.start()
    try:
        yield _iter_from_queue(queue, count)
    finally:
        stop_event.set()
        thread.join()


def _iter_from_queue(queue, count):
    for i in range(count):
        yield queue.get()

def _prefetch_rows(local_iterator, queue: Queue, stop_event: Event):
    for row in local_iterator:
        try:
            queue.put(row, timeout=5)

        except Full:
            # if timeout reached, check to see
            # if we have been terminated
            if stop_event.is_set():
                return
        






class MemmapVectorStore(MemmapSeqs):
    """
    a class for storing sorted sets of token ids (as arrays)
    """
    vector_dtype = _VECTOR_DTYPE

    @dataclass(frozen=True)
    class CacheKey(CachedObjectKey):
        index_col : str
        search_col : Union[str, None]
        tokenizer_type : str

    @staticmethod
    def arrays_to_encoded_sparse_vector(ind: np.ndarray, val: np.ndarray) -> bytes:
        arr = np.empty(len(ind), dtype=MemmapVectorStore.vector_dtype)
        arr['ind'] = ind
        arr['val'] = val
        return arr.tobytes()

    @staticmethod
    def decode_sparse_vector(bin: bytes) -> np.ndarray:
        arr = np.frombuffer(bin, dtype=MemmapVectorStore.vector_dtype)
        return arr



    @classmethod
    @type_check_call
    def build(cls, df: pyspark.sql.DataFrame, seq_col: str, id_col: str='_id'):
        """
        create a MemmapSeqs instance from a spark dataframe 

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            the dataframe containing the sequences and ids
        seq_col : str
            the name of the column in `df` that contains the sequences, e.g. strings, arrays
        dtype : type
            the dtype of the elements in `seq_col`
        id_col : str
            the name of the column in `df` that contains the ids for retrieving the sequences

        Returns
        -------
        MemmapSeqs
        """
        dtype = cls.vector_dtype
        obj = cls()

        df = df.filter(df[seq_col].isNotNull())\
                .select(id_col, seq_col, (F.length(seq_col) / 8).alias('size'))\
                .persist(pyspark.StorageLevel.DISK_ONLY)
        
        try:
            print('materializing DF')
            nrows = df.count()

            id_arr = np.empty(nrows, dtype=np.uint64) 
            size_arr = np.empty(nrows, dtype=np.uint64) 

            local_mmap_file = Path(mkstemp(suffix='.mmap_arr')[1])
            # buffer 256MB at a time
            with open(local_mmap_file, 'wb', buffering=(2**20) * 256) as ofs, iter_spark_rows(df, 2**16) as itr:
                for i, row in enumerate(itr):
                    id_arr[i] = row[id_col]
                    size_arr[i] = row['size']
                    ofs.write(memoryview(row[seq_col]))

            print('BUILD memmap file written')

            
            obj._id_to_offset_map = IdOffsetHashMap.build(
                    id_arr,
                    np.arange(len(id_arr), dtype=np.int32)
            )

            obj._offset_arr = MemmapArray(np.cumsum(np.concatenate([np.zeros(1, dtype=np.uint64), size_arr]))) 
            seq_arr = np.memmap(local_mmap_file, shape=obj._offset_arr.values[-1], dtype=dtype, mode='r+')
            
            obj._seq_arr = MemmapArray(seq_arr)

        finally:
            df.unpersist()

        return obj



    def dot(self, query: np.ndarray, ids : np.ndarray) -> np.ndarray:
        """
        compute cosine score between `query` and the sequences referenced by `ids`

        Parameters
        ----------
        query : np.ndarray
            a sorted unique array of token ids
        ids : np.ndarray
            an array of ids of token sets in `self`

        Returns
        -------
        an array of scores where 
            scores[i] = cosine(`query`, token_sets[`ids[i]`]) if `ids[i]` is in token_sets
            else scores[i] = np.nan
        """
        indexes = self._id_to_offset_map[ids]
        return _sparse_dot_many(self._offset_arr.values, self._seq_arr.values, query, indexes)


    def fetch(self, i: int, /) -> np.ndarray | None:
        """
        retrieve the sequence associated with `i`

        Returns 
        -------
        np.ndarray if `i` is found, else None
        """
        idx = self._id_to_offset_map[i]
        if idx == -1:
            return None
        else:
            start = self._offset_arr.values[idx]
            end = self._offset_arr.values[idx+1]
            return self._seq_arr.values[start:end]

njit_kwargs = {
        'nogil' : True,
        'fastmath' : True,
        'parallel' : False,
        'cache' : False
}

@nb.njit(**njit_kwargs)
def _sparse_dot_many(offsets, vecs, query, indexes):
    scores = np.empty(len(indexes), dtype=np.float32)
    for i in range(len(indexes)):
        if indexes[i] < 0:
            scores[i] = np.nan
        else:
            start = offsets[indexes[i]]
            end = offsets[indexes[i] + 1]
            if start == end or len(query) == 0:
                scores[i] = 0.0
            else:
                scores[i] = _sparse_dot(query, vecs[start:end])

    return scores

@nb.njit(**njit_kwargs)
def _sparse_dot(l_vec, r_vec, /):
    """
    compute dot product between two structured arrays

    Returns
    -------
    int 
    """
    l = 0
    r = 0
    s = np.float32(0.0)

    while l < len(l_vec) and r < len(r_vec):
        if l_vec[l].ind == r_vec[r].ind:
            s += l_vec[l].val * r_vec[r].val
            l += 1
            r += 1
        elif l_vec[l].ind <= r_vec[r].ind:
            l += 1
        else:
            r += 1

    return s

