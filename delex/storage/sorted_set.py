import pyspark.sql.functions as F
import pyspark.sql.types as T
import pyspark
from typing import Iterator, Union
import numpy as np
import numba as nb
from delex.utils.numba_functions import sorted_set_overlap
from delex.storage.memmap_seqs import MemmapSeqs
import pandas as pd
from pydantic.dataclasses import dataclass
from delex.utils import CachedObjectKey
from delex.utils.funcs import type_check_call

@F.pandas_udf(T.BinaryType())
def _array_to_binary(itr : Iterator[pd.Series]) -> Iterator[pd.Series]:
    for s in itr:
        yield s.apply(lambda x : x.tobytes() if x is not None else x)

class MemmapSortedSets(MemmapSeqs):
    """
    a class for storing sorted sets of token ids (as arrays)
    """

    @dataclass(frozen=True)
    class CacheKey(CachedObjectKey):
        index_col : str
        search_col : Union[str, None]
        tokenizer_type : str
    
    @classmethod
    @type_check_call
    def build(cls, df: pyspark.sql.DataFrame, col: str, id_col: str='_id'):
        """
        Create a new MemmapSortedSets over tokens in `df[col]` and writing to disk
        """
        return super().build(df, col, np.int32, id_col)

    def jaccard(self, query: np.ndarray, ids : np.ndarray) -> np.ndarray:
        """
        compute jaccard score between `query` and the sequences referenced by `ids`

        Parameters
        ----------
        query : np.ndarray
            a sorted unique array of token ids
        ids : np.ndarray
            an array of ids of token sets in `self`

        Returns
        -------
        an array of scores where 
            scores[i] = jaccard(`query`, token_sets[`ids[i]`]) if `ids[i]` is in token_sets
            else scores[i] = np.nan
        """
        indexes = self._id_to_offset_map[ids]
        return _jaccard(self._offset_arr.values, self._seq_arr.values, query, indexes)
    
    def overlap_coeff(self, query: np.ndarray, ids : np.ndarray) -> np.ndarray:
        """
        compute overlap_coefficient score between `query` and the sequences referenced by `ids`

        Parameters
        ----------
        query : np.ndarray
            a sorted unique array of token ids
        ids : np.ndarray
            an array of ids of token sets in `self`

        Returns
        -------
        an array of scores where 
            scores[i] = overlap_coefficient(`query`, token_sets[`ids[i]`]) if `ids[i]` is in token_sets
            else scores[i] = np.nan
        """
        indexes = self._id_to_offset_map[ids]
        return _overlap_coeff(self._offset_arr.values, self._seq_arr.values, query, indexes)

    def cosine(self, query: np.ndarray, ids : np.ndarray) -> np.ndarray:
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
        return _cosine(self._offset_arr.values, self._seq_arr.values, query, indexes)

njit_kwargs = {
        'nogil' : True,
        'fastmath' : True,
        'parallel' : False,
        'cache' : False
}

@nb.njit(**njit_kwargs)
def _jaccard(offsets, sets, query, indexes):
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
                olap = sorted_set_overlap(query, sets[start:end])
                scores[i] = olap / (len(query) + (end - start) - olap)

    return scores

@nb.njit(**njit_kwargs)
def _overlap_coeff(offsets, sets, query, indexes):
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
                olap = sorted_set_overlap(query, sets[start:end])
                scores[i] =  olap / min(len(query), (end - start))

    scores = np.empty(len(indexes), dtype=np.float32)
    for i in range(len(indexes)):
        start = offsets[indexes[i]]
        end = offsets[indexes[i] + 1]

        olap = sorted_set_overlap(query, sets[start:end])

    return scores

@nb.njit(**njit_kwargs)
def _cosine(offsets, sets, query, indexes):
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
                olap = sorted_set_overlap(query, sets[start:end])
                scores[i] = olap / np.sqrt(len(query) * (end - start))

    return scores
