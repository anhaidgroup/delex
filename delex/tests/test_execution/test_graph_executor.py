"""
Tests for execution.graph_executor module.

This module tests graph execution functionality.
"""
import pytest
from delex.execution.graph_executor import GraphExecutor


@pytest.mark.unit
class TestGraphExecutor:
    """Tests for GraphExecutor class."""

    @pytest.mark.requires_spark
    def test_graph_executor_init(self, spark_session):
        """Test GraphExecutor initialization."""
        index_table = spark_session.createDataFrame([(i,) for i in range(1000)], ['_id'])
        search_table = spark_session.createDataFrame([(i,) for i in range(1000)], ['_id'])
        executor = GraphExecutor(index_table=index_table, search_table=search_table)
        assert executor is not None
        assert executor.index_table is not None
        assert executor.search_table is not None
        assert executor.index_table_count == 1000

    def test_graph_executor_topological_order(self):
        """Test that nodes are executed in topological order."""
        from delex.graph import PredicateNode
        from delex.graph.algorithms import topological_sort
        from delex.lang import ExactMatchPredicate
        node1 = PredicateNode(ExactMatchPredicate(index_col='_id', search_col='_id', invert=False))
        node2 = PredicateNode(ExactMatchPredicate(index_col='_id', search_col='_id', invert=False))
        node3 = PredicateNode(ExactMatchPredicate(index_col='_id', search_col='_id', invert=False))
        node4 = PredicateNode(ExactMatchPredicate(index_col='_id', search_col='_id', invert=False))
        node5 = PredicateNode(ExactMatchPredicate(index_col='_id', search_col='_id', invert=False))
        node6 = PredicateNode(ExactMatchPredicate(index_col='_id', search_col='_id', invert=False))
        node7 = PredicateNode(ExactMatchPredicate(index_col='_id', search_col='_id', invert=False))
        node1.add_out_edge(node2)
        node1.add_out_edge(node3)
        node3.add_out_edge(node6)
        node3.add_out_edge(node4)
        node4.add_out_edge(node7)
        node5.add_out_edge(node7)
        topo_sorted = topological_sort(node7)
        expected_nodes = {id(node1), id(node3), id(node4), id(node5), id(node7)}
        actual_nodes = {id(n) for n in topo_sorted}
        # node6 should not be in the sorted list, since it is not connected to node7
        assert id(node6) not in actual_nodes
        assert actual_nodes == expected_nodes
        index = {n: i for i, n in enumerate(topo_sorted)}
        assert index[node1] < index[node3]
        assert index[node3] < index[node4]
        assert index[node5] < index[node7]
        assert index[node4] < index[node7]
