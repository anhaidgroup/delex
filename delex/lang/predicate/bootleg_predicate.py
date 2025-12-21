import operator
from delex.index import HashIndex
from delex.storage import MemmapStrings
from delex.lang.predicate import ThresholdPredicate
from delex.utils import CachedObjectKey
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pandas as pd
from typing import Iterator
import numpy as np
from pydantic.dataclasses import dataclass
import re

from .name_map import NAME_MAP as _NAME_MAP

@dataclass(frozen=True)
class CachedNamesKey(CachedObjectKey):
    index_col : str

@dataclass(frozen=True)
class CachedNameIndexKey(CachedObjectKey):
    index_col : str
    lowercase : bool

@dataclass(frozen=True)
class BootlegSim:
    index_col : str
    search_col : str
    invert : bool

_ALPHA_RE = re.compile('[a-z]+')


def _normalize_name(n):
    if isinstance(n, str):
        chunks = [_NAME_MAP.get(c, c) for c in _ALPHA_RE.findall(n.lower()) if len(c) > 1]
        chunks.sort()
        return ' '.join(chunks)
    else:
        return None

@F.pandas_udf(T.StringType())
def _normalize_name_spark(itr : Iterator[pd.Series]) -> Iterator[pd.Series]:
    for s in itr:
        yield s.apply(_normalize_name)

class BootlegPredicate(ThresholdPredicate):
    """
    an experimental user defined predicate for demonstration. In particular, 
    does some simple preprocessing of person names to make exact match more liberal
    by handling name variations
    """

    def __init__(self, index_col : str, search_col : str, invert : bool=False):
        super().__init__(index_col, search_col, operator.eq, 0.0 if invert else 1.0)
        
        self._invert = invert
        self._indexable = not invert
        self._built_for_search = None
        self._index = None
        self._sim = BootlegSim(
                self.index_col,
                self.search_col,
                self._invert
        )
    
    def __str__(self):
        return f'{self._sim_name}({self.index_col}, {self.search_col}) == {not self._invert}'

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, o):
        return type(self) == type(o) and\
                self.sim == o.sim 
    
    def index_size_in_bytes(self) -> int:
        return self._index.size_in_bytes()

    def index_component_sizes(self, for_search: bool) -> dict:
        # for now only considering the size of the index itself, not 
        # the size of the tokenizer
        if self._built_for_search is not None and for_search != self._built_for_search:
            raise ValueError('cannot get component sizes {for_search != self._built_for_search=}')
        return {
                self._get_index_key(for_search) : None if self._index is None else self._index.size_in_bytes()
        }
    
    @property
    def op(self):
        return operator.eq 

    @property
    def val(self):
        return 0.0 if self._invert else 1.0

    @property
    def index_col(self):
        return self._index_col

    @property
    def search_col(self):
        return self._search_col

    @property
    def sim(self):
        return self._sim
    
    @property
    def _sim_name(self) -> str:
        return 'name_match'

    @property
    def is_topk(self):
        return False

    @property
    def streamable(self):
        return True

    @property
    def indexable(self):
        return self._indexable

    def init(self):
        self._index.init()

    def deinit(self):
        self._index.deinit()

    def _get_index_key(self, for_search: bool):
        if for_search:
            key = CachedNameIndexKey(
                    index_col=self._index_col, 
                    lowercase=False
            )
        else:
            key = CachedNamesKey(
                    index_col=self._index_col
            )
        return key


    def build(self, for_search, index_table, index_id_col='_id', cache=None):
        if index_table.schema[self._index_col].dataType != T.StringType():
            raise TypeError('index column must be StringType')

        index_table = index_table.withColumn(self._index_col, _normalize_name_spark(self._index_col))

        if not for_search:
            if cache is not None:
                key = self._get_index_key(for_search)
                entry = cache.get(key)
                with entry:
                    if entry.obj is None:
                        self._index = MemmapStrings.build(index_table, self._index_col, index_id_col)
                        self._index.to_spark()
                        entry.obj = self._index
                    else:
                        self._index = entry.obj
            else:
                self._index = MemmapStrings.build(index_table, self._index_col, index_id_col)
        else:
            if cache is not None:
                key = self._get_index_key(for_search)
                entry = cache.get(key)
                with entry:
                    if entry.obj is None:
                        self._index = HashIndex()
                        self._index.build(index_table, self._index_col, index_id_col)
                        self._index.to_spark()
                        entry.obj = self._index
                    else:
                        self._index = entry.obj
            else:
                self._index = HashIndex()
                self._index.build(index_table, self._index_col, index_id_col)

        self._index.to_spark()
        self._built_for_search = for_search
    
    def compute_scores(self, query : str, id1_list):
        query = _normalize_name(query)

        return np.fromiter(
                ((query == self._index.fetch(i)) for i in id1_list),
                count=len(id1_list),
                dtype=np.float32
            )

    def search_index(self, query):
        query = _normalize_name(query)
        res = self._index.fetch(query)
        if res is None:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        else:
            return np.full(len(res), 1.0, dtype=np.float32), res

    def contains(self, other):
        return type(self) == type(other) and self.sim == other.sim



