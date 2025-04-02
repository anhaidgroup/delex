import numpy as np
from numba import njit
from numba.typed import List as TypedList
from scipy import sparse
from joblib import Parallel, delayed
from delex.storage import PackedMemmapArrays, MemmapArray
from delex.utils.numba_functions import  typed_list_to_array
import pyspark.sql.functions as F
from .filtered_set_sim_index_slice import JaccardSetSimIndexSlice, CosineSetSimIndexSlice
from delex.utils.traits import SparkDistributable
from delex.utils import CachedObjectKey
from pydantic.dataclasses import dataclass
from typing import Union, Tuple


class FilteredSetSimIndex(SparkDistributable):
    """
    an optimized memory mapped index for set similarity measures
    """

    @dataclass(frozen=True)
    class CacheKey(CachedObjectKey):
        index_col : str
        search_col : Union[str, None]
        tokenizer_type : str
        sim : str
        threshold : float

    SLICE_TYPES = {
            'jaccard' : JaccardSetSimIndexSlice,
            'cosine' : CosineSetSimIndexSlice
    }

    def __init__(self, sim, threshold, max_slice_size = 2**14):
        FilteredSetSimIndex.SLICE_TYPES[sim]
        self.nrow = None
        self.slices = None
        self._slice_size = max_slice_size
        self._packed_arrs = None
        self._slc_shapes = None
        self._ids = None
        self._sim = sim
        self._threshold = threshold
        self._max_slice_size = 2**14
        self._on_spark = False
    

    def size_in_bytes(self) -> int:
        return self._packed_arrs.size_in_bytes()

    @property
    def _slice_t(self):
        return FilteredSetSimIndex.SLICE_TYPES[self._sim]

    @classmethod
    def from_sparse_mat(cls, sparse_mat, sim, threshold, ids=None):

        obj = cls(sim, threshold)

        if not isinstance(sparse_mat, (sparse.csr_matrix, sparse.csr_array)):
            raise TypeError
        obj.nrow, _ = sparse_mat.shape
        sparse_mat = sparse_mat.astype(np.float32)

        obj.slices = TypedList()

        offset = 0
        while offset < obj.nrow:
            start = offset
            end = min(obj.nrow, start + obj._slice_size)
            slc = sparse_mat[start:end]

            
            idx_slice = obj._slice_t(
                        slc.shape[0],
                        slc.shape[1],
                        obj._threshold,
                        slc.indices,
                        slc.indptr,
                        offset
                    )

            obj.slices.append(idx_slice)
            offset += obj._slice_size
        
        obj._packed_arrs = None
        obj._slc_shapes = None
        if ids is not None:
            obj._ids = MemmapArray(ids)

        return obj
        
    def _build_slice(self, offset, tokens):
        tokens_slc = tokens[offset : min(len(tokens), offset+self._slice_size)]
        indptr = np.zeros(len(tokens_slc) + 1, dtype=np.int32)
        indptr[1:] = np.cumsum(list(map(len, tokens_slc)))

        return self._slice_t(
                    len(tokens_slc),
                    self._threshold,
                    np.concatenate(tokens_slc, dtype=np.int32, casting='unsafe'),
                    indptr,
                    offset
        )

    def build(self, df, token_col, id_col='_id'):
        
        tokens_df = df.select(id_col, token_col)\
                    .filter(F.col(token_col).isNotNull())\
                    .toPandas()

        self._ids = MemmapArray(tokens_df[id_col].values)
        offsets = range(0, len(tokens_df), self._slice_size)
        pool = Parallel(n_jobs=-1, backend='threading')
        tokens = tokens_df[token_col].values
        slices = pool(delayed(self._build_slice)(o, tokens) for o in offsets)

        self.slices = TypedList()
        for s in slices:
            self.slices.append(s)

    def to_spark(self):
        if not self._on_spark:
            if self._ids is not None:
                self._ids.to_spark()
            
            self._slc_shapes = []
            arrs = []
            for slc in self.slices:
                arrs.append(slc.data)
                arrs.append(slc.span_map)
                arrs.append(slc.set_data)
                arrs.append(slc.set_indptr)
                self._slc_shapes.append((slc.nrow, slc.thres, slc.offset))

            self._packed_arrs = PackedMemmapArrays(arrs)
            self._packed_arrs.to_spark()
            self.slices = None
            self._on_spark = True

    def init(self):
        if self._ids is not None:
            self._ids.init()

        self.slices = TypedList()
        arr_itr = iter(self._packed_arrs.unpack())

        for nrows, thres, offset in self._slc_shapes:
            data = next(arr_itr)
            span_map = next(arr_itr)
            set_data = next(arr_itr)
            set_indptr = next(arr_itr)
            self.slices.append( self._slice_t(
                        nrows,
                        thres, 
                        set_data, 
                        set_indptr,
                        offset,
                        data,
                        span_map,
                    )
            )
    
    def deinit(self):
        self._ids.deinit()
        self._packed_arrs.deinit()
        self.slices = None

    def search(self, tokens: np.ndarray, thres: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        search the index with tokens and retrieve all 
        ids with score > `thres` 

        Parameters
        ----------
        tokens : np.ndarray np.int32
            the tokens for searching
        thres : float
            the minimum threshold to retrieve

        Returns
        -------
        np.ndarray np.int64
            the ids from the index with score that satisfies the threshold
        """
        scores, indexes = _search(self.slices, tokens, thres)
        ids = indexes if self._ids is None else self._ids.values[indexes]
        return scores, ids



njit_kwargs = {
        'nogil' : True,
        'fastmath' : True,
        'parallel' : False,
        'cache' : False
}

@njit( **njit_kwargs)
def _search(slices, indexes, thres):
    score_out = TypedList.empty_list(np.float32)
    indexes_out = TypedList.empty_list(np.int32)
    for s in slices:
        s.search(indexes, thres, score_out, indexes_out)
    
    return typed_list_to_array(score_out), typed_list_to_array(indexes_out)
