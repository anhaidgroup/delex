"""
Tests for graph.node module.

This module tests the Node base class and its implementations.
"""
import pytest
from delex.graph.node import Node, PredicateNode
from delex.lang.predicate import ExactMatchPredicate

@pytest.mark.unit
class TestNode:
    """Tests for Node class."""

    class DummyNode(Node):
        def __init__(self):
            super().__init__()

        def execute(self, stream):
            return "called"

        def build(self, index_table, id_col, cache=None):
            return None

        def validate(self):
            return None

        def working_set_size(self):
            return {}

        def streamable(self):
            return True

    def test_node_init(self):
        """Test Node initialization."""
        node = self.DummyNode()
        assert node is not None

    def test_is_source(self):
        """Test is_source property."""
        node = self.DummyNode()
        assert node.is_source

    def test_is_sink(self):
        """Test is_sink property."""
        node = self.DummyNode()
        assert node.is_sink

    def test_output_col(self):
        """Test output_col property."""
        node = self.DummyNode()
        assert node.output_col == f'col_{id(node)}'

    def test_in_degree(self):
        """Test in_degree property."""
        node = self.DummyNode()
        assert node.in_degree == 0

    def test_out_degree(self):
        """Test out_degree property."""
        node = self.DummyNode()
        assert node.out_degree == 0

    def test_iter_dependencies(self):
        """Test iter_dependencies method."""
        node = self.DummyNode()
        assert list(node.iter_dependencies()) == []

    def test_iter_out(self):
        """Test iter_out method."""
        node = self.DummyNode()
        assert list(node.iter_out()) == []

    def test_iter_in(self):
        """Test iter_in method."""
        node = self.DummyNode()
        assert list(node.iter_in()) == []

    def test_add_in_edge(self):
        """Test add_in_edge method."""
        node = self.DummyNode()
        other_node = self.DummyNode()
        node.add_in_edge(other_node)
        assert node.in_degree == 1
        assert other_node.out_degree == 1

    def test_add_out_edge(self):
        """Test add_out_edge method."""
        node = self.DummyNode()
        other_node = self.DummyNode()
        node.add_out_edge(other_node)
        assert node.out_degree == 1
        assert other_node.in_degree == 1

    def test_remove_in_edges(self):
        """Test remove_in_edges method."""
        node = self.DummyNode()
        other_node = self.DummyNode()
        node.add_in_edge(other_node)
        node.remove_in_edges()
        assert node.in_degree == 0
        assert other_node.out_degree == 0

    def test_remove_out_edges(self):
        """Test remove_out_edges method."""
        node = self.DummyNode()
        other_node = self.DummyNode()
        node.add_out_edge(other_node)
        node.remove_out_edges()
        assert node.out_degree == 0
        assert other_node.in_degree == 0

    def test_remove_in_edge(self):
        """Test remove_in_edge method."""
        node = self.DummyNode()
        other_node = self.DummyNode()
        node.add_in_edge(other_node)
        node.remove_in_edge(other_node)
        assert node.in_degree == 0
        assert other_node.out_degree == 0

    def test_remove_out_edge(self):
        """Test remove_out_edge method."""
        node = self.DummyNode()
        other_node = self.DummyNode()
        node.add_out_edge(other_node)
        node.remove_out_edge(other_node)
        assert node.out_degree == 0
        assert other_node.in_degree == 0

    def test_pop(self):
        """Test pop method."""
        node = self.DummyNode()
        other_node = self.DummyNode()
        node.add_in_edge(other_node)
        node.pop()
        assert node.in_degree == 0
        assert other_node.out_degree == 0

    def test_insert_after(self):
        """Test insert_after method."""
        node = self.DummyNode()
        other_node = self.DummyNode()
        node.insert_after(other_node)
        assert node.out_degree == 1
        assert other_node.in_degree == 1

    def test_insert_before(self):
        """Test insert_before method."""
        node = self.DummyNode()
        other_node = self.DummyNode()
        node.insert_before(other_node)
        assert node.in_degree == 1
        assert other_node.out_degree == 1
        assert other_node.in_degree == 0

    def test_ancestors(self):
        """Test ancestors method."""
        node = self.DummyNode()
        other_node = self.DummyNode()
        node.add_in_edge(other_node)
        assert node.ancestors() == {other_node}


@pytest.mark.unit
class TestPredicateNode:
    """Tests for PredicateNode class."""

    def test_node_init(self):
        """Test Node initialization."""
        pred = ExactMatchPredicate(index_col='title', search_col='title', invert=False)
        node = PredicateNode(pred)
        assert node is not None
        assert node.predicate == pred

    def test_iter_dependencies(self):
        """Test iter_dependencies method."""
        pred = ExactMatchPredicate(index_col='title', search_col='title', invert=False)
        node = PredicateNode(pred)
        assert list(node.iter_dependencies()) == ['title']

    def test_node_is_source(self):
        """Test is_source property."""
        pred = ExactMatchPredicate(index_col='title', search_col='title', invert=False)
        node = PredicateNode(pred)
        assert node.is_source

    def test_node_is_sink(self):
        """Test is_sink property."""
        pred = ExactMatchPredicate(index_col='title', search_col='title', invert=False)
        node = PredicateNode(pred)
        assert node.is_sink

    def test_streamable(self):
        """Test streamable property."""
        pred = ExactMatchPredicate(index_col='title', search_col='title', invert=False)
        node = PredicateNode(pred)
        assert node.streamable

    def test_execute(self):
        """Test execute method."""
        pred = ExactMatchPredicate(index_col='title', search_col='title', invert=False)
        node = PredicateNode(pred)

        class DummyStream:
            def apply(self, func, input_cols, output_col, output_type):
                return "called"

        dummy_stream = DummyStream()
        assert node.execute(dummy_stream) == "called"

    def test_validate(self):
        """Test validate method."""
        with pytest.raises(RuntimeError):
            pred = ExactMatchPredicate(index_col='title', search_col='title', invert=True)
            node = PredicateNode(pred)
            node.validate()

    def test_working_set_size(self, spark_session):
        """Test working_set_size method."""
        from delex.index import HashIndex
        from delex.utils import BuildCache
        pred = ExactMatchPredicate(index_col='title', search_col='title', invert=False)
        node = PredicateNode(pred)
        df = spark_session.createDataFrame([(1, 'a'), (2, 'b'), (3, 'c')], ['_id', 'title'])
        cache = BuildCache()
        node.build(df, id_col='_id', cache=cache)
        sizes = node.working_set_size()
        key = HashIndex.CacheKey(index_col='title', lowercase=False)
        assert key in sizes
        assert sizes[key] is not None and sizes[key] > 0
