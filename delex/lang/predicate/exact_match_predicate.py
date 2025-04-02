import operator
from delex.index import HashIndex
from delex.storage import  MemmapStrings
from delex.lang.predicate import ThresholdPredicate, Predicate
import pyspark.sql.functions as F
import pyspark.sql.types as T
import numpy as np
from pydantic.dataclasses import dataclass
from delex.utils.funcs import type_check_call
from delex.utils.build_cache import BuildCache
import pyspark

class ExactMatchPredicate(ThresholdPredicate):
    """
    an exact match predicate, i.e. if x == y return 1.0 else 0.0
    """
    @dataclass(frozen=True)
    class Sim:
        index_col : str
        search_col : str
        invert : bool
        lowercase : bool

    @type_check_call
    def __init__(self, index_col : str, search_col : str, invert: bool, lowercase: bool = False):
        """
        index_col : str
            the column to be indexed
        search_col : str
            the column that will be used for search
        invert : bool
            change predicate from `index_col` == `search_col` to `index_col` != `search_col` 
        lowercase : bool
            lowercase the strings before comparing them
        """
        super().__init__(index_col, search_col, operator.eq, 0.0 if invert else 1.0)
        self._index_col = index_col
        self._search_col = search_col

        self._indexable = not invert
        self._built_for_search = None
        self._index = None
        self._invert = invert
        self._lowercase = lowercase
        self._sim = ExactMatchPredicate.Sim(
                self.index_col,
                self.search_col,
                self._invert,
                self._lowercase
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
        if self._lowercase:
            return 'lowercase_exact_match'
        else:
            return 'exact_match'

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
                key = HashIndex.CacheKey(
                        index_col=self._index_col, 
                        lowercase=self._lowercase
                )
            else:
                key = MemmapStrings.CacheKey(
                        index_col=self._index_col
                )
            return key

    @type_check_call
    def build(self, for_search: bool, index_table: pyspark.sql.DataFrame, index_id_col: str='_id', cache: BuildCache=None):
               
        if index_table.schema[self._index_col].dataType in {T.IntegerType(), T.LongType()}:
            index_table = index_table.withColumn(self._index_col, index_table[self._index_col].cast(T.StringType()))
        
        if index_table.schema[self._index_col].dataType == T.StringType():
            if self._lowercase:
                index_table = index_table.withColumn(self._index_col, F.lower(index_table[self._index_col]))

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

        else:
            raise TypeError('index column must be StringType, LongType, or IntegerType')

        self._index.to_spark()
        self._built_for_search = for_search
    
    def compute_scores(self, query : str|int, id1_list) -> np.ndarray:
        query = str(query)
        if self._lowercase:
            query = query.lower()

        return np.fromiter(
                ((query == self._index.fetch(i)) for i in id1_list),
                count=len(id1_list),
                dtype=np.float32
            )

    def search_index(self, query) -> np.ndarray:
        res = self._index.fetch(str(query))
        if res is None:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        else:
            return np.full(len(res), 1.0, dtype=np.float32), res

    @type_check_call
    def contains(self, other: Predicate) -> bool:
        return type(self) == type(other) and self.sim == other.sim



