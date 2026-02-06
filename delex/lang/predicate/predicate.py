from abc import ABC, abstractproperty, abstractmethod
import operator
from typing import Iterator, Tuple
import pandas as pd
from delex.utils.build_cache import BuildCache
from typing import Optional
import pyspark

# mapping of operators to their inverses
OPERATORS = {
        operator.ge : operator.lt,
        operator.gt : operator.le,
        operator.eq : operator.ne,
        operator.ne : operator.eq,
        operator.le : operator.gt,
        operator.lt : operator.ge
}

OPERATOR_TO_STR = {
        operator.ge : '>=',
        operator.gt : '>',
        operator.eq : '==',
        operator.ne : '!=',
        operator.le : '<=', 
        operator.lt : '<'
}

GT_OPS = {
        operator.ge,
        operator.gt
}
LT_OPS = {
        operator.le,
        operator.lt
}



class Predicate(ABC):
    """
    abstract base class for all Predicates to be used in writing blocking programs
    """
    
    @abstractproperty
    def streamable(self):
        """
        True if the predicate can be evaluated over a single
        partition of the indexed table, otherwise False
        """

    @abstractproperty
    def indexable(self):
        """
        True if the predicate can be efficiently indexed 
        """

    @abstractproperty
    def sim(self):
        """
        The simiarlity used by the predicate
        """

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return f'{type(self).__name__}({str(self)})'

    @abstractproperty
    def is_topk(self) -> bool:
        """
        True if the `self` is Topk based, else False
        """
        pass

    @abstractmethod
    def build(self, for_search : bool, index_table: pyspark.sql.DataFrame, index_id_col: str='_id', cache: Optional[BuildCache]=None):
        """
        build the Predicate over `index_table` using `index_id_col` as a unique id, 
        optionally using `cache` to get or set the index

        Parameters
        ----------
        for_search : bool
            build the predicate for searching, otherwise streaming / filtering

        index_table : pyspark.sql.DataFrame
            the dataframe that will be preprocessed / indexed

        index_id_col : str
            the name of the unique id column in `index_table`

        cache : Optional[BuildCache] = None
            the cache for built indexes and hash tables
        """


    @abstractmethod
    def contains(self, other) -> bool:
        """
        True if the set output by self is a superset (non-strict) of `other`
        """

    @abstractmethod
    def search_batch(self, queries: pd.Series) -> pd.DataFrame:
        """
        perform search with `queries` return a dataframe 
        with schema (id1_list array<long>, scores array<float>, time float)
        """

    def search(self, itr : Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        """
        perform `search_batch` for each batch in `itr`
        """
        self.init()
        for queries in itr:
            yield self.search_batch(queries)
        self.deinit()

    @abstractmethod
    def filter_batch(self, queries: pd.Series, id1_lists: pd.Series) -> pd.DataFrame:
        """
        filter each id_list in id1_lists using this predicate. This is, 
        for each query, id_list pair in zip(`queries`, `id1_lists`), return only the ids 
        which satisfy predicate(query, id) for id in id_list. Return a dataframe 
        with schema (id1_list array<long>, scores array<float>, time float)
        """

    def filter(self, itr : Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.DataFrame]:
        """
        perform `filter_batch` for each batch in `itr`
        """
        self.init()
        for queries, id1_lists in itr:
            yield self.filter_batch(queries, id1_lists)
        self.deinit()

    @abstractmethod
    def init(self):
        """
        initialize the predicate for searching or filtering
        """

    @abstractmethod
    def deinit(self):
        """
        release the resources acquired by `self.init()`
        """

    @abstractmethod
    def index_size_in_bytes(self) -> int:
        """
        return the total size in bytes of all the files associated with this predicate
        """

    @abstractmethod
    def index_component_sizes(self, for_search : bool) -> dict:
        """
        return a dictionary of file sizes for each data structure used by this 
        predicate, if the predicate hasn't been built yet, the sizes are None

        Parameters
        ----------
        for_search : bool
            return the sizes for searching or for filtering

        Returns
        -------
            dict[Any, int | None]
        """
