"""
Tests for graph.algorithms module.

This module tests graph algorithms like topological sort.
"""
import pytest
from delex.graph.algorithms import (
    topological_sort, find_all_nodes, find_sink, clone_graph
)
from delex.graph import PredicateNode, UnionNode
from delex.lang.predicate import ExactMatchPredicate


@pytest.mark.unit
class TestGraphAlgorithms:
    """Tests for other graph algorithms."""

    def test_topological_sort(self):
        """Test topological sort function."""
        # Create a simple graph: node1 -> node2 -> sink
        pred1 = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        pred2 = ExactMatchPredicate(
            index_col='description', search_col='description', invert=False)
        node1 = PredicateNode(pred1)
        node2 = PredicateNode(pred2)
        node1.add_out_edge(node2)
        sorted_nodes = topological_sort(node2)
        assert len(sorted_nodes) == 2
        assert node1 in sorted_nodes
        assert node2 in sorted_nodes
        assert sorted_nodes.index(node1) < sorted_nodes.index(node2)

    def test_find_all_nodes(self):
        """Test find_all_nodes function."""
        # Create a graph: node1 -> union -> node2
        pred1 = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        pred2 = ExactMatchPredicate(
            index_col='description', search_col='description', invert=False)
        node1 = PredicateNode(pred1)
        node2 = PredicateNode(pred2)
        union = UnionNode()
        node1.add_out_edge(union)
        union.add_in_edge(node2)
        all_nodes = find_all_nodes(node1)
        assert len(all_nodes) == 3
        assert node1 in all_nodes
        assert node2 in all_nodes
        assert union in all_nodes

    def test_find_sink(self):
        """Test find_sink function."""
        # Create a simple graph: node1 -> node2 (sink)
        pred1 = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        pred2 = ExactMatchPredicate(
            index_col='description', search_col='description', invert=False)
        node1 = PredicateNode(pred1)
        node2 = PredicateNode(pred2)
        node1.add_out_edge(node2)
        sink = find_sink(node1)
        assert sink == node2
        assert sink.is_sink
        sink2 = find_sink(node2)
        assert sink2 == node2
