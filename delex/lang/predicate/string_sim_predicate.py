from .predicate import  OPERATORS, OPERATOR_TO_STR
from .threshold_predicate import ThresholdPredicate
from delex.storage import  MemmapStrings
from abc import  abstractproperty, abstractmethod
from py_stringmatching import Levenshtein, Jaro, JaroWinkler, SmithWaterman
import numpy as np
from pydantic.dataclasses import dataclass


class StringSimPredicate(ThresholdPredicate):

    @dataclass(frozen=True)
    class Sim:
        index_col : str
        search_col : str
        sim_name : str

    def __init__(self, index_col : str, search_col : str, op, val):
        super().__init__(index_col, search_col, op, val)

        self._indexable = False
        self._built_for_search = None
        self._index = None
        self._sim = StringSimPredicate.Sim(
                self.index_col,
                self.search_col,
                self._sim_name
        )
    
    def __str__(self):
        return f'{self._sim_name}({self.index_col}, {self.search_col}) {OPERATOR_TO_STR[self.op]} {self._val}'

    """
    fine to inherit from ThresholdPredicate because type determines sim
    def __eq__(self, o):
        pass
    """
    
    def index_size_in_bytes(self) -> int:
        return self._index.size_in_bytes()

    def index_component_sizes(self, for_search: bool) -> dict:
        if self._built_for_search is not None and for_search != self._built_for_search:
            raise ValueError('cannot get component sizes {for_search != self._built_for_search=}')

        return {
                self._get_index_key(for_search) : None if self._index is None else self._index.size_in_bytes()
        }

    @property
    def sim(self):
        return self._sim

    @abstractproperty
    def _sim_name(self) -> str:
        pass

    def invert(self):
        return self.__class__(
                self.index_col, 
                self.search_col, 
                OPERATORS[self.op],
                self.val
        )

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
            raise ValueError('search not implemented')
        else:
            key = MemmapStrings.CacheKey(
                    index_col=self._index_col
            )
        return key

    def build(self, for_search, index_table, index_id_col='_id', cache=None):
        if for_search and not self.indexable:
            raise RuntimeError('cannot build {self} for search, not indexable')

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

        self._index.to_spark()
        self._built_for_search = for_search
    
    @abstractmethod
    def _compute_score(self, x : str, y : str):
        """
        x is assumed to be not null as _compute_scores checks x already
        """
        pass

    def compute_scores(self, query : str, id1_list):
        return np.fromiter(
                (self._compute_score(query, self._index.fetch(i)) for i in id1_list),
                count=len(id1_list),
                dtype=np.float32
            )

    def search_index(self, query):
        raise NotImplementedError()

    def search(self, itr):
        raise NotImplementedError()


class EditDistancePredicate(StringSimPredicate):
    _SIM = Levenshtein().get_sim_score

    @property
    def _sim_name(self) -> str:
        return 'edit_distance'

    def _compute_score(self, x : str, y : str):
        if y is None:
            return np.nan
        else:
            return self._SIM(x, y)
        
class JaroPredicate(StringSimPredicate):
    _SIM = Jaro().get_sim_score

    @property
    def _sim_name(self) -> str:
        return 'jaro'

    def _compute_score(self, x : str, y : str):
        if y is None:
            return np.nan
        else:
            return self._SIM(x, y)


class JaroWinklerPredicate(StringSimPredicate):

    @dataclass(frozen=True)
    class Sim(StringSimPredicate.Sim):
        prefix_weight : float

    def __init__(self, index_col : str, search_col : str, op, val, prefix_weight=.1):
        self._prefix_weight = prefix_weight
        super().__init__(index_col, search_col, op, val)
        self._sim_func = JaroWinkler(self._prefix_weight).get_sim_score
        self._sim = JaroWinklerPredicate.Sim(
                self.index_col, 
                self.search_col,
                'jaro_winkler',
                self._prefix_weight
        )

    @property
    def _sim_name(self) -> str:
        return f'jaro_winkler[{self._prefix_weight}]'

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, o):
        return super().__eq__(o) and self._prefix_weight == o._prefix_weight

    def contains(self, o):
        return super().contains(o) and self._prefix_weight == o._prefix_weight

    def _compute_score(self, x : str, y : str):
        if y is None:
            return np.nan
        else:
            return self._sim_func(x, y)

    
class SmithWatermanPredicate(StringSimPredicate):

    @dataclass(frozen=True)
    class Sim(StringSimPredicate.Sim):
        gap_cost : float

    def __init__(self, index_col : str, search_col : str, op, val, gap_cost=1.0):
        self._gap_cost = gap_cost
        super().__init__(index_col, search_col, op, val)
        self._sim_func = SmithWaterman(gap_cost=self._gap_cost).get_raw_score
        self._sim = SmithWatermanPredicate.Sim(
                self.index_col, 
                self.search_col,
                'smith_waterman',
                self._gap_cost
        )

    @property
    def _sim_name(self) -> str:
        return f'smith_waterman[{self._gap_cost}]'

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, o):
        return super().__eq__(o) and self._gap_cost == o._gap_cost

    def contains(self, o):
        return super().contains(o) and self._gap_cost == o._gap_cost

    def _compute_score(self, x : str, y : str):

        if y is None:
            return np.nan
        else:
            return self._sim_func(x, y) / max(len(x), len(y))
