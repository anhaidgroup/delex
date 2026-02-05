from abc import ABC,  abstractmethod
import time
from .predicate import Predicate, OPERATORS, LT_OPS, GT_OPS
from delex.utils.funcs import type_check
from typing import Callable, Iterator
import pandas as pd
import numpy as np

class ThresholdPredicate(Predicate, ABC):

    def __init__(self, index_col, search_col, op, val: float):
        if op not in OPERATORS:
            raise ValueError('operator must be in {OPERATORS}')

        type_check(val, 'val', float)
        type_check(index_col, 'index_col', str)
        type_check(search_col, 'search_col', str)

        self._index_col = index_col
        self._search_col = search_col
        self._op = op
        self._val = val

    def contains(self, other) -> bool:
        type_check(other, 'other', Predicate)
        if type(self) == type(other) and self.sim == other.sim:
            if self.op == other.op:
                return self.op(other.val, self.val) or other.val == self.val
            # both greater than ops or less than ops
            elif (self.op in GT_OPS and other.op in GT_OPS) or (self.op in LT_OPS and other.op in LT_OPS):
                return self.op(other.val, self.val)
            else:
                return False

        return False

    def __hash__(self):
        return hash(f'{type(self)}{self.index_col}{self.search_col}{self.op}{self.val}') 

    def __eq__(self, o):
        return type(self) == type(o) and\
                self.sim == o.sim and\
                self.op == o.op and\
                self.val == o.val 

    @abstractmethod
    def compute_scores(self, query, id1_list):
        pass

    @abstractmethod
    def search_index(self, query):
        pass

    @property
    def invertable(self) -> bool:
        return True

    @property
    def index_col(self):
        return self._index_col

    @property
    def search_col(self):
        return self._search_col

    @property
    def op(self) -> Callable:
        return self._op

    @property
    def val(self) -> float:
        return self._val
    
    def search_batch(self, queries: pd.Series) -> pd.DataFrame:
        res = []
        for query in queries:
            if query is None:
                res.append((np.empty(0, np.float32), np.empty(0, np.int64), 0.0))
            else:
                start_t = time.perf_counter()
                scores, ids = self.search_index(query) 
                t = time.perf_counter() - start_t
                res.append((scores, ids, t))

        return pd.DataFrame(res, columns=['scores', 'id1_list', 'time'])

    def filter_batch(self, queries: pd.Series, id1_lists: pd.Series) -> Iterator[pd.DataFrame]:
        res = []
        for query, id_list in zip(queries, id1_lists):
            if query is None:
                res.append((np.empty(0, np.float32), np.empty(0, np.int64), 0.0))
            else:
                start_t = time.perf_counter()
                scores = self.compute_scores(query, id_list)
                t = time.perf_counter() - start_t
                mask = self._op(scores, self._val)
                res.append( (scores[mask], id_list[mask], t) ) 

        return pd.DataFrame(res, columns=['scores', 'id1_list', 'time'])


