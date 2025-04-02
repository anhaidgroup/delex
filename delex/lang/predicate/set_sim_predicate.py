from .predicate import  OPERATORS, OPERATOR_TO_STR
from .threshold_predicate import ThresholdPredicate
import operator
from delex.index import FilteredSetSimIndex
from delex.storage import MemmapSortedSets
from delex.tokenizer import Tokenizer
import pyspark.sql.functions as F
from pydantic.dataclasses import dataclass
from abc import abstractmethod


    

class SetSimPredicate(ThresholdPredicate):

    @dataclass(frozen=True)
    class Sim:
        index_col : str
        search_col : str
        sim_name : str
        tokenizer_name : str

    def __init__(self, index_col : str, search_col : str, tokenizer, op, val : float):
        super().__init__(index_col, search_col, op, val)

        self._tokenizer = tokenizer
        self._indexable = self.op in (operator.gt, operator.ge)
        self._built_for_search = None
        self._index = None

        self._sim = SetSimPredicate.Sim(
                self.index_col,
                self.search_col,
                self._sim_name,
                str(self._tokenizer)
        )
    
    @property
    def sim(self):
        return self._sim
    
    def _get_index_key(self, for_search: bool):
            if for_search:
                key = FilteredSetSimIndex.CacheKey(
                        index_col=self._index_col,
                        search_col=self._search_col,
                        tokenizer_type=str(self._tokenizer),
                        sim=self._sim_name,
                        threshold=self._val
                )
            else:
                key = MemmapSortedSets.CacheKey(
                        index_col=self._index_col,
                        search_col=self._search_col,
                        tokenizer_type=str(self._tokenizer),
                )
            return key

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

    def contains(self, other):
        return super().contains(other) and self._tokenizer == other._tokenizer
    
    def __hash__(self):
        return super().__hash__()

    def __str__(self):
        return f'{self._sim_name}({self._tokenizer}, {self.index_col}, {self.search_col}) {OPERATOR_TO_STR[self.op]} {self._val}'

    def __eq__(self, o):
        return super().__eq__(o) and self._tokenizer == o._tokenizer

    def invert(self):
        return self.__class__(
                self.index_col, 
                self.search_col, 
                self.tokenizer,
                OPERATORS[self.op],
                self.val
        )
        
    @property
    @abstractmethod
    def _sim_name(self):
        pass

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
        self._tokenizer.init()

    def deinit(self):
        self._index.deinit()
        self._tokenizer.deinit()
    
    def _build_tokenizer(self, index_table, index_id_col):
        TMP_COL = '__string'
        df = index_table.select(F.col(self._index_col).alias(TMP_COL))
        self._tokenizer.build(df, TMP_COL)


    def _build_or_get_cached_tokenizer(self, index_table, index_id_col, cache):

        if cache is not None:
            key = Tokenizer.CacheKey(
                    index_col=self._index_col,
                    search_col=self._search_col,
                    tokenizer_type=str(self._tokenizer)
            )
            entry = cache.get(key) 
            with entry:
                if entry.obj is None:
                    self._build_tokenizer(index_table, index_id_col)
                    entry.obj = self._tokenizer
                else:
                    self._tokenizer = entry.obj
        else:
            self._build_tokenizer(index_table, index_id_col)

    def _build_index(self, for_search, index_table, index_id_col):
        token_col = '__TOKENS'
        index_table = index_table.select(
                index_id_col,
                self._tokenizer.tokenize_set_spark(self._index_col).alias(token_col)
            )

        if for_search:
            self._index = FilteredSetSimIndex(self._sim_name, self._val)
            self._index.build(index_table, token_col, index_id_col)
        else:
            self._index = MemmapSortedSets.build(index_table, token_col, index_id_col)

    def _build_or_get_cached_index(self, for_search, index_table, index_id_col, cache):
        if cache is not None:
            key = self._get_index_key(for_search)

            entry = cache.get(key) 
            with entry:
                if entry.obj is None:
                    self._build_index(for_search, index_table, index_id_col)
                    self._index.to_spark()
                    entry.obj = self._index
                else:
                    self._index = entry.obj
        else:
            self._build_index(for_search, index_table, index_id_col)
                


    def build(self, for_search, index_table, index_id_col='_id', cache=None):
        if for_search and not self.indexable:
            raise RuntimeError('cannot build {self} for search, not indexable')

        
        self._build_or_get_cached_tokenizer(index_table, index_id_col, cache)
        self._build_or_get_cached_index(for_search, index_table, index_id_col, cache)       

        self._index.to_spark()
        self._built_for_search = for_search

    def search_index(self, query):
        query_toks = self._tokenizer.tokenize_set(query)
        return self._index.search(query_toks, self._val)

            
class JaccardPredicate(SetSimPredicate):
    
    @property
    def _sim_name(self):
        return 'jaccard'

    def compute_scores(self, query, id1_list):
        query_toks = self._tokenizer.tokenize_set(query)
        return self._index.jaccard(query_toks, id1_list)

class OverlapCoeffPredicate(SetSimPredicate):
    
    @property
    def _sim_name(self):
        return 'overlap_coeff'

    @property
    def indexable(self):
        return False

    def compute_scores(self, query, id1_list):
        query_toks = self._tokenizer.tokenize_set(query)
        return self._index.overlap_coeff(query_toks, id1_list)

class CosinePredicate(SetSimPredicate):
  
    @property
    def _sim_name(self):
        return 'cosine'

    def compute_scores(self, query, id1_list):
        query_toks = self._tokenizer.tokenize_set(query)
        return self._index.cosine(query_toks, id1_list)
