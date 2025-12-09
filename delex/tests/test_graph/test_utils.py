"""
Tests for graph.utils module.

This module tests graph utility functions.
"""
import pytest
from delex.graph.utils import nodes_to_dot
from delex.graph import PredicateNode
from delex.lang.predicate import ExactMatchPredicate


@pytest.mark.unit
class TestGraphUtils:
    """Tests for graph utility functions."""

    def test_nodes_to_dot_single_node(self):
        """Test converting single node to dot format."""
        node = PredicateNode(ExactMatchPredicate(index_col='title', search_col='title', invert=False))
        dot = nodes_to_dot(node)
        assert isinstance(dot, str)
        assert 'digraph' in dot
        assert 'title' in dot

    def test_nodes_to_dot_multiple_nodes(self):
        """Test converting multiple nodes to dot format."""
        node1 = PredicateNode(ExactMatchPredicate(index_col='title', search_col='title', invert=False))
        node2 = PredicateNode(ExactMatchPredicate(index_col='description', search_col='description', invert=False))
        node1.add_out_edge(node2)
        dot = nodes_to_dot(node1)
        assert isinstance(dot, str)
        assert 'digraph' in dot
        assert 'title' in dot
        assert 'description' in dot
        # Should have an edge between nodes
        assert '->' in dot

    def test_nodes_to_dot_node_list(self):
        """Test converting list of nodes to dot format."""
        node1 = PredicateNode(ExactMatchPredicate(index_col='title', search_col='title', invert=False))
        node2 = PredicateNode(ExactMatchPredicate(index_col='description', search_col='description', invert=False))
        node1.add_out_edge(node2)
        dot = nodes_to_dot([node1, node2])
        assert isinstance(dot, str)
        assert 'digraph' in dot
        assert 'title' in dot
        assert 'description' in dot
        assert '->' in dot

    def test_nodes_to_dot_empty_graph(self):
        """Test converting empty graph to dot format."""
        dot = nodes_to_dot([])
        assert isinstance(dot, str)
        assert 'digraph' in dot
        assert '->' not in dot
