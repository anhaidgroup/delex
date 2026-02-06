"""
Tests for execution.graph_executor module.

This module tests graph execution functionality.
"""
import pytest
from delex.execution.graph_executor import GraphExecutor
from delex.execution.cost_estimation import CostEstimator


@pytest.mark.unit
class TestGraphExecutor:
    """Tests for GraphExecutor class."""

    @pytest.mark.requires_spark
    def test_graph_executor_init(self, table_a, table_b):
        """Test GraphExecutor initialization."""
        cost_est = CostEstimator(table_a, table_b, 4)
        executor = GraphExecutor(
            index_table=table_a, search_table=table_b, cost_est=cost_est)
        assert executor is not None
        assert executor.index_table is not None
        assert executor.search_table is not None
        assert executor.index_table_count == 1000

    @pytest.mark.requires_spark
    def test_graph_executor_properties(self, table_a, table_b):
        """Test GraphExecutor properties."""
        cost_est = CostEstimator(table_a, table_b, 4)
        executor = GraphExecutor(
            index_table=table_a, search_table=table_b, cost_est=cost_est)
        assert executor.use_cost_estimation is True
        assert executor.use_chunking is False
        assert executor.index_table_count == 1000

    @pytest.mark.slow
    def test_graph_executor_execute_without_chunking(self, table_a, table_b):
        """Test execute method without chunking."""
        from delex.graph import PredicateNode
        from delex.lang.predicate import ExactMatchPredicate
        cost_est = CostEstimator(table_a, table_b, 4)
        from delex.index import HashIndex
        cache_key = HashIndex.CacheKey(index_col='title', lowercase=False)
        cost_est._index_size[cache_key] = type(
            "DummyModel", (), {"predict": lambda self, size: 13276})()

        executor = GraphExecutor(
            index_table=table_a, search_table=table_b, cost_est=cost_est)
        pred = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        node = PredicateNode(pred)
        s_table, g_stats = executor.execute(sink=node, search_table_id_col='_id')
        assert s_table is not None
        assert g_stats is not None
        assert len(g_stats.nodes) >= 1
        assert len(g_stats.sub_graph_stats) >= 1
        assert g_stats.sub_graph_stats[0].exec_time >= 0.0
        assert g_stats.sub_graph_stats[0].build_time >= 0.0
        assert g_stats.sub_graph_stats[0].working_set_size >= 0.0
        assert g_stats.sub_graph_stats[0].total_time >= 0.0
        assert g_stats.exec_time >= 0.0
        assert g_stats.build_time >= 0.0
        assert g_stats.working_set_size >= 0.0
        assert g_stats.total_time >= 0.0

    @pytest.mark.slow
    def test_graph_executor_execute_with_chunking(self, table_a, table_b):
        """Test execute method with chunking."""
        from delex.graph import PredicateNode
        from delex.lang.predicate import BM25TopkPredicate, ExactMatchPredicate
        from delex.index import HashIndex
        from delex.storage import MemmapStrings

        cost_est = CostEstimator(table_a, table_b, 4)
        pred = BM25TopkPredicate(
            index_col='title', search_col='title', tokenizer='3gram', k=10)
        node = PredicateNode(pred)
        bm25_key = pred._get_index_key(for_search=True)
        executor = GraphExecutor(
            index_table=table_a, search_table=table_b, cost_est=cost_est,
            ram_size_in_bytes=13270)
        pred = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        em_node = PredicateNode(pred)
        node._out_edges.add(em_node)
        em_node._in_edges.add(node)
        # Dummy model that scales with size so chunking can find a solution
        # Returns size * 13.276 to ensure it's > ram_size for full dataset
        cost_est._index_size[bm25_key] = type(
            "DummyModel", (),
            {"predict": lambda self, size: max(size * 13.276, 1000)})()

        cache_key = HashIndex.CacheKey(index_col='title', lowercase=False)
        cost_est._index_size[cache_key] = type(
            "DummyModel", (),
            {"predict": lambda self, size: max(size * 13.276, 1000)})()

        memmap_key = MemmapStrings.CacheKey(index_col='title')
        cost_est._index_size[memmap_key] = type(
            "DummyModel", (),
            {"predict": lambda self, size: max(size * 13.276, 1000)})()

        s_table, g_stats = executor.execute(sink=em_node, search_table_id_col='_id')
        assert s_table is not None
        assert g_stats is not None
        assert len(g_stats.nodes) >= 1
        assert len(g_stats.sub_graph_stats) >= 1
        assert g_stats.sub_graph_stats[0].exec_time >= 0.0
        assert g_stats.sub_graph_stats[0].build_time >= 0.0
        assert g_stats.sub_graph_stats[0].working_set_size >= 0.0
        assert g_stats.sub_graph_stats[0].total_time >= 0.0
        assert g_stats.exec_time >= 0.0
        assert g_stats.build_time >= 0.0
        assert g_stats.working_set_size >= 0.0
        assert g_stats.total_time >= 0.0

    @pytest.mark.slow
    def test_graph_executor_execute_without_chunking_and_projection(self, table_a, table_b):
        """Test execute method without chunking."""
        from delex.graph import PredicateNode
        from delex.lang.predicate import ExactMatchPredicate
        cost_est = CostEstimator(table_a, table_b, 4)
        from delex.index import HashIndex
        cache_key = HashIndex.CacheKey(index_col='title', lowercase=False)
        cost_est._index_size[cache_key] = type(
            "DummyModel", (), {"predict": lambda self, size: 13276})()

        executor = GraphExecutor(
            index_table=table_a, search_table=table_b, cost_est=cost_est)
        pred = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        node = PredicateNode(pred)
        s_table, g_stats = executor.execute(sink=node, search_table_id_col='_id', projection=['_id', 'title'])
        assert s_table.columns == ['id2', 'title', 'id1_list']
        assert s_table.count() == table_b.count()

