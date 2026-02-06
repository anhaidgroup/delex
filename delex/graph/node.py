from abc import ABC, abstractmethod, abstractproperty
import pyspark.sql.types as T
from typing import Iterator, Optional
import pandas as pd
import numpy as np
from delex.utils.funcs import type_check
from delex.utils.build_cache import BuildCache
import pyspark

class Node(ABC):
    """
    abstract base class for all graph nodes
    """
    def __init__(self):
        self._in_edges = set()
        # list of outgoing edges (direct decendants)
        self._out_edges = set()

        self._output_col = f'col_{id(self)}'
        self._id_string = str(self)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)
    
    @property
    def is_source(self) -> bool:
        return len(self._in_edges) == 0

    @property
    def is_sink(self) -> bool:
        return len(self._out_edges) == 0

    @property
    def output_col(self) -> str:
        return self._output_col

    @property
    def in_degree(self) -> int:
        return len(self._in_edges)

    @property
    def out_degree(self) -> int:
        return len(self._out_edges)

    def iter_dependencies(self) -> Iterator:
        """
        return an iterator over the dependencies of this node
        """
        return (node.output_col for node in self.iter_in())

    def iter_out(self) -> Iterator:
        """
        return an iterator of nodes for the outgoing edges
        """
        return iter(self._out_edges)

    def iter_in(self) -> Iterator:
        """
        return an iterator of nodes for the incoming edges
        """
        return iter(self._in_edges)

    def add_in_edge(self, other):
        """
        add an edge between `other` -> `self`
        """
        if not isinstance(other, Node):
            raise TypeError(type(other))
        self._in_edges.add(other)
        other._out_edges.add(self)

    def add_out_edge(self, other):
        """
        add an edge between `self` -> `other`
        """
        if not isinstance(other, Node):
            raise TypeError(type(other))
        self._out_edges.add(other)
        other._in_edges.add(self)

    def remove_in_edges(self):
        """
        remove all incoming edges, x -> `self`
        """
        for n in list(self.iter_in()):
            self.remove_in_edge(n)

    def remove_out_edges(self):
        """
        remove all outgoing edges, `self` -> x
        """
        for n in list(self.iter_out()):
            self.remove_out_edge(n)

    def remove_in_edge(self, other):
        """
        remove `other` -> `self`

        Raises
        ------
        KeyError 
            if `other` -> `self` doesn't exist
        """
        if not isinstance(other, Node):
            raise TypeError(type(other))
        self._in_edges.remove(other)
        other._out_edges.remove(self)

    def remove_out_edge(self, other):
        """
        remove `self` -> `other`

        Raises
        ------
        KeyError 
            if `self` -> `other` doesn't exist
        """
        if not isinstance(other, Node):
            raise TypeError(type(other))
        self._out_edges.remove(other)
        other._in_edges.remove(self)
        

    def pop(self):
        """
        remove this node from the graph and reconnect edges between in and out 
    
        Returns
        -------
        self

        Raises
        ------
        RuntimeError
            if self.out_degree > 1 and self.in_degree > 1

        """
        if self.in_degree > 1 and self.out_degree > 1:
            raise RuntimeError('unable to pop node when in and out degree are > 1')
        # sever connections to this node
        in_nodes = list(self.iter_in())
        out_nodes = list(self.iter_out())
        for n in in_nodes:
            self.remove_in_edge(n)
        for n in out_nodes:
            self.remove_out_edge(n)
        # add edges 
        for in_node in in_nodes:
            for out_node in out_nodes:
                in_node.add_out_edge(out_node)

        return self

    def insert_after(self, node):
        """
        insert `node` after this node, e.g.

        self -> x  becomes self -> node -> x

        Parameters
        ----------
        node : Node
            the node to be inserted
        """
        type_check(node, 'node', Node)
        out_nodes = list(self.iter_out())
        for n in out_nodes:
            self.remove_out_edge(n)
            node.add_out_edge(n)

        self.add_out_edge(node)

    def insert_before(self, node):
        """
        insert `node` before this node, e.g.

        x -> self  becomes x -> node -> self

        Parameters
        ----------
        node : Node
            the node to be inserted
        """
        type_check(node, 'node', Node)
        in_nodes = list(self.iter_in())
        for n in in_nodes:
            self.remove_in_edge(n)
            node.add_in_edge(n)

        self.add_in_edge(node)


    @abstractmethod
    def execute(self, stream):
        """
        execute the operation of this node over a DataFrameStream 
        and return a new DataFrameStream
        """


    @abstractmethod
    def build(self, index_table: pyspark.sql.DataFrame, id_col: str, cache: Optional[BuildCache]=None):
        """
        build this node over `index_table` using `id_col` as the unique id, optionally with 
        `cache`

        Parameters
        ----------
        index_table : pyspark.sql.DataFrame
            the dataframe that will be preprocessed / indexed

        id_col : str
            the name of the unique id column in `index_table`

        cache : Optional[BuildCache] = None
            the cache for built indexes and hash tables
        """

    @abstractmethod
    def validate(self):
        """
        perform validation for this node, e.g. ensure that
        UnionNodes have multiple inputs, MinusNodes have two inputs, etc.

        Raises
        ------
        ValueError 
            if validation fails
        """
        pass

    @abstractproperty
    def streamable(self) -> bool:
        """
        True if the operation at this node can be streamed, else False
        """

    @property
    def id_string(self) -> str:
        """
        a string and identifies this node for graph comparison
        without accounting for edges
        """
        return self._id_string
    
    def _ancestors(self, visited):
        for tail in self.iter_in():
            if tail not in visited:
                visited.add(tail)
                tail._ancestors(visited)

    def ancestors(self) -> set:
        """
        get all ancestors of this node

        Returns
        -------
        Set[Node]
            all the ancestors of this node
        """

        visited = set()
        self._ancestors(visited)
        return visited

    def _equivalent(self, other, visited) -> bool:
        
        if self.in_degree != other.in_degree\
                or self.out_degree != other.out_degree\
                or self.id_string != other.id_string:
            return False

        visited[self] = other
        if self.in_degree != 0:
            # need to recurse
            id_string_to_node = {n.id_string : n for n in other.iter_in()}
            if len(id_string_to_node) != other.in_degree:
                raise RuntimeError(f'node id strings node unique for incoming edges {id_string_to_node=}')

            for n in self.iter_in():
                if n.id_string not in id_string_to_node:
                    return False

            for n in self.iter_in():
                if n not in visited:
                    if not n._equivalent(id_string_to_node[n.id_string], visited):
                        # if an ancestor is not equivalent return false
                        return False
                elif visited[n] != id_string_to_node[n.id_string]:
                    # node was visited but matched with a different node in the graph
                    return False
        # all checks pass, return true
        return True


    def equivalent(self, other) -> bool:
        """
        check `self` is equivalent to `other`, this does a
        recursive check and can be used to compare two graphs if 
        `self` and `other` are both sinks

        Parameters
        ----------
        other : Node
            the node to be compared to

        Returns
        -------
        True if equivalent else False
        """
        visited = {}
        return self._equivalent(other, visited)

    @abstractmethod
    def working_set_size(self) -> dict:
        """
        return the working set size of each component use for this node, 
        dict values are None if self.build has not been called yet
        """



class PredicateNode(Node):
    """
    a node that execute a Predicate
    """
    _OUTPUT_TYPE = T.StructType([
            T.StructField('scores', T.ArrayType(T.FloatType())),
            T.StructField('id1_list', T.ArrayType(T.LongType())),
            T.StructField('time', T.FloatType()),
    ])

    def __init__(self, predicate):
        self._predicate = predicate
        super().__init__()
    
    def __str__(self):
        return str(self._predicate)

    def __repr__(self):
        return f'PredicateNode({self})'
    
    def iter_dependencies(self):
        yield self.predicate.search_col
        yield from (node.output_col for node in self.iter_in())

    def init(self):
        return self.predicate.init()

    @property
    def predicate(self):
        return self._predicate

    @property
    def streamable(self):
        return self._predicate.streamable


    def execute(self, stream):
        if self.is_source:
            input_cols = [self.predicate.search_col]
            func = self.predicate.search_batch
        else:
            in_col = (next(self.iter_in()).output_col, 'id1_list')
            input_cols = [self.predicate.search_col, in_col]
            func = self.predicate.filter_batch

        return stream.apply(func, input_cols, self.output_col, self._OUTPUT_TYPE)

    def build(self, index_table, id_col, cache=None):
        self.predicate.build(self.is_source, index_table, id_col, cache=cache)

    def validate(self):
        if self.is_source:
            if not self.predicate.indexable:
                raise RuntimeError
        else:
            if self.in_degree != 1:
                raise RuntimeError

    def working_set_size(self) -> dict:
        return self.predicate.index_component_sizes(self.is_source)


class SetOpNode(Node):
    """
    Base Class for all set operations nodes
    """

    _OUTPUT_TYPE = T.StructType([
            T.StructField('id1_list', T.ArrayType(T.LongType())),
    ])

    def __init__(self):
        super().__init__()
    
    def __repr__(self):
        return str(type(self))

    def init(self):
        pass

    @abstractmethod
    def _execute_batch(self, *cols):
        pass

    def working_set_size(self) -> dict:
        return {}

    def execute(self, stream):
        input_cols = [(n.output_col, 'id1_list') for n in self.iter_in()]
        return stream.apply(self._execute_batch, input_cols, self.output_col, self._OUTPUT_TYPE)


    @property
    def streamable(self):
        return True

    def build(self, index_table, id_col, cache=None):
        pass 

class UnionNode(SetOpNode):
    """
    union the output of all incoming edges and return a single set
    """
    def __str__(self):
        return 'UNION'
    
    def _execute_batch(self, *cols):
        itr = (np.unique(np.concatenate(t)) for t in zip(*cols))
        return pd.DataFrame({'id1_list' : np.fromiter(itr, dtype=object, count=len(cols[0]))})

    def validate(self):
        if self.in_degree < 2:
            raise RuntimeError


class IntersectNode(SetOpNode):
    """
    intersect the output of all incoming edges and return a single set
    """
    def __str__(self):
        return 'INTERSECT'

    def _execute_batch(self, *cols):
        res = cols[0]
        for c in cols[1:]:
            res = [np.intersect1d(x,y) for x, y in zip(res, c)]
        return pd.DataFrame({'id1_list' : np.array(res, dtype=object)})

    def validate(self):
        if self.in_degree < 2:
            raise RuntimeError

class MinusNode(SetOpNode):
    """
    compute the set minus of two sets
    `left` - `right`
    """
    def __init__(self, left, right):
        super().__init__()
        self._left = left
        self._right = right
        self.add_in_edge(left)
        self.add_in_edge(right)
        self._id_string = f'{self._left} {self} {self._right}'

    def _execute_batch(self, *cols):
        res = (np.setdiff1d(x,y) for x, y in zip(*cols))
        return pd.DataFrame({'id1_list' : np.fromiter(res, dtype=object, count=len(cols[0]))})
    
    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    def __str__(self):
        return 'MINUS'

    def validate(self):
        if self.in_degree != 2:
            raise RuntimeError
