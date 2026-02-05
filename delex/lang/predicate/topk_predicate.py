from delex.utils.funcs import type_check, init_jvm, attach_current_thread_jvm
from typing import Iterator
import pandas as pd
from .predicate import Predicate
from delex.utils import CachedObjectKey
from sparkly.index import LuceneIndex
from sparkly.index_config import IndexConfig
from sparkly.query_generator import QuerySpec
from pydantic.dataclasses import dataclass
from tempfile import mkdtemp
from delex.utils import size_in_bytes
from pathlib import Path
import threading

_sparkly_index_lock = threading.Lock()

@dataclass(frozen=True)
class CachedBM25IndexKey(CachedObjectKey):
    index_col : str
    tokenizer : str

class BM25TopkPredicate(Predicate):

    @dataclass(frozen=True)
    class Sim:
        index_col : str
        search_col : str
        tokenizer_name : str

    def __init__(self, index_col, search_col, tokenizer: str, k: int):

        type_check(k, 'k', int)
        type_check(index_col, 'index_col', str)
        type_check(search_col, 'search_col', str)

        self._index_col = index_col
        self._search_col = search_col
        self._k = k
        self._tokenizer = tokenizer
        self._index = None
        self._index_dir = None
        self._build_java_heap_size = '1024m'
        self._worker_java_heap_size = '256m'
        self._query_spec = QuerySpec({
                self._search_col : {f'{self._index_col}.{self._tokenizer}'}
            })


        self._sim = BM25TopkPredicate.Sim(
                self.index_col,
                self.search_col,
                self._tokenizer
        )

    def contains(self, other) -> bool:
        type_check(other, 'other', Predicate)
        return type(self) == type(other) and\
                self.sim == other.sim and\
                self.k >= other.k
    
    def _get_index_key(self, for_search: bool):
        if for_search:
            key = CachedBM25IndexKey(
                    index_col=self.index_col,
                    tokenizer=self._tokenizer
                )
        else:
             raise ValueError('filter not implemented')
        return key

    def index_size_in_bytes(self) -> int:
        return size_in_bytes(Path(self._index_dir))

    def index_component_sizes(self, for_search: bool) -> dict:
        # for now only considering the size of the index itself, not 
        # the size of the tokenizer
        return {
                self._get_index_key(for_search) : None if self._index_dir is None else self.index_size_in_bytes()
        }

    def __str__(self):
        return f'BM25_topk({self._tokenizer}, {self.index_col}, {self.search_col}, {self._k}) '

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, o):
        return type(self) == type(o) and\
                self.sim == o.sim 
    
    def init(self):
        init_jvm([f'-Xmx{self._worker_java_heap_size}'])
        self._index.init()

    def deinit(self):
        self._index.deinit()

    @property
    def sim(self):
        return self._sim

    @property
    def k(self):
        return self._k

    @property
    def invertable(self) -> bool:
        return False

    @property
    def index_col(self):
        return self._index_col

    @property
    def search_col(self):
        return self._search_col

    @property
    def streamable(self):
        """
        True if the predicate can be evaluated over a single
        partition of the indexed table, otherwise False
        """
        return False

    @property
    def indexable(self):
        """
        True if the predicate can be efficiently indexed 
        """
        return True

    @property
    def is_topk(self) -> bool:
        return True
    
    def _build_index(self, index_table, index_id_col):
        with _sparkly_index_lock:
            # needed for multithreaded build
            init_jvm([f'-Xmx{self._build_java_heap_size}'])
            attach_current_thread_jvm()
            index_config = IndexConfig(id_col=index_id_col)
            index_config.add_field(self.index_col, [self._tokenizer])
            index_dir = mkdtemp(prefix='bm25_index.')
            index = LuceneIndex(index_dir, index_config)
            index.upsert_docs(index_table, force_distributed=True)

        return index


    def build(self, for_search, index_table, index_id_col='_id', cache=None):
        if not for_search:
            raise RuntimeError('cannot build {self} for filtering, not streamable')

        
        if cache is not None:
            key = self._get_index_key(for_search)
            entry = cache.get(key)
            with entry:
                if entry.obj is None:
                    index = self._build_index(index_table, index_id_col)
                    entry.obj = index
                else:
                    index = entry.obj
        else:
            index = self._build_index(index_table, index_id_col)

        index.to_spark()

        self._index_dir = index.index_path
        self._index = index
    
    def search_batch(self, queries):
        queries = pd.DataFrame({self.search_col : queries})
        res = self._index.search_many(
                queries,
                self._query_spec,
                self.k
            ).rename(columns={'search_time' : 'time'})

        return res[['scores', 'id1_list', 'time']]

    def filter_batch(self, queries: pd.Series, id1_lists: pd.Series) -> Iterator[pd.DataFrame]:
        raise RuntimeError('topk cannot be used as a filter')

