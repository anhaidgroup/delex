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


    @pytest.mark.requires_spark
    def test_graph_executor_execute_rejects_duplicate_search_ids(self, table_a, table_b):
        """execute raises when the search table id column has duplicates."""
        from delex.graph import PredicateNode
        from delex.lang.predicate import ExactMatchPredicate
        executor = GraphExecutor(
            index_table=table_a, search_table=table_b.union(table_b))
        pred = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        node = PredicateNode(pred)
        with pytest.raises(ValueError, match="search_table: id column '_id' must be unique"):
            executor.execute(sink=node, search_table_id_col='_id')

    @pytest.mark.requires_spark
    def test_graph_executor_execute_rejects_null_index_ids(self, table_a, table_b):
        """execute raises when the index table id column contains nulls."""
        import pyspark.sql.functions as F
        from delex.graph import PredicateNode
        from delex.lang.predicate import ExactMatchPredicate
        with_null = table_a.withColumn(
            '_id', F.when(F.col('_id') == 0, None).otherwise(F.col('_id')))
        executor = GraphExecutor(index_table=with_null, search_table=table_b)
        pred = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        node = PredicateNode(pred)
        with pytest.raises(ValueError, match="index_table: nulls are present"):
            executor.execute(sink=node, search_table_id_col='_id')

    @pytest.mark.requires_spark
    def test_graph_executor_execute_rejects_missing_search_id_column(self, table_a, table_b):
        """execute raises when search_table_id_col is not in the search table."""
        from delex.graph import PredicateNode
        from delex.lang.predicate import ExactMatchPredicate
        executor = GraphExecutor(index_table=table_a, search_table=table_b)
        pred = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        node = PredicateNode(pred)
        with pytest.raises(ValueError, match="search_table: missing id column 'no_such_col'"):
            executor.execute(sink=node, search_table_id_col='no_such_col')

    @pytest.mark.slow
    def test_graph_executor_execute_validate_ids_disabled(self, table_a, table_b):
        """execute skips id validation when validate_ids is False."""
        from delex.graph import PredicateNode
        from delex.lang.predicate import ExactMatchPredicate
        duplicated = table_b.union(table_b)
        executor = GraphExecutor(index_table=table_a, search_table=duplicated)
        pred = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        node = PredicateNode(pred)
        s_table, g_stats = executor.execute(
            sink=node, search_table_id_col='_id', validate_ids=False)
        assert s_table.count() == duplicated.count()

    @pytest.mark.requires_spark
    def test_graph_executor_execute_rejects_reserved_rid_column(self, table_a, table_b):
        """execute raises when the index table already has a _rid column."""
        import pyspark.sql.functions as F
        from delex.graph import PredicateNode
        from delex.lang.predicate import ExactMatchPredicate
        executor = GraphExecutor(
            index_table=table_a.withColumn('_rid', F.col('_id')), search_table=table_b)
        pred = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        node = PredicateNode(pred)
        with pytest.raises(ValueError, match="index_table: column '_rid' is reserved"):
            executor.execute(sink=node, search_table_id_col='_id')

    @pytest.mark.requires_spark
    def test_graph_executor_execute_rejects_rid_as_id_col(self, table_a, table_b):
        """execute raises when index_table_id_col is the reserved _rid name."""
        from delex.graph import PredicateNode
        from delex.lang.predicate import ExactMatchPredicate
        executor = GraphExecutor(
            index_table=table_a.withColumnRenamed('_id', '_rid'),
            search_table=table_b,
            index_table_id_col='_rid')
        pred = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        node = PredicateNode(pred)
        with pytest.raises(ValueError, match="index_table_id_col cannot be '_rid'"):
            executor.execute(sink=node, search_table_id_col='_id')

    @pytest.mark.slow
    def test_graph_executor_execute_outputs_index_table_ids(self, spark_session):
        """id1_list holds the ids of the index table, whatever their values."""
        import pyspark.sql.functions as F
        from delex.graph import PredicateNode
        from delex.lang.predicate import ExactMatchPredicate

        words = ['apple', 'banana', 'cherry', 'dog', 'elephant']
        index_table = spark_session.createDataFrame(
            [(i, words[i % 5], words[(i // 5) % 5]) for i in range(200)],
            ['_id', 'title', 'description'])
        search_table = spark_session.createDataFrame(
            [(i, words[i % 5], words[(i // 7) % 5]) for i in range(100)],
            ['_id', 'title', 'description'])

        def pairs(index_table):
            executor = GraphExecutor(index_table=index_table, search_table=search_table)
            pred = ExactMatchPredicate(
                index_col='title', search_col='title', invert=False)
            node = PredicateNode(pred)
            out, _ = executor.execute(sink=node, search_table_id_col='_id')
            assert out.columns == ['id2', 'id1_list']
            rows = out.select(F.explode('id1_list').alias('a'), 'id2').toPandas()
            return set(rows.itertuples(index=False, name=None))

        baseline = pairs(index_table)
        shifted = pairs(index_table.withColumn('_id', F.col('_id') * 1000 - 50_000))
        assert len(baseline) == 200 * 100 // 5
        assert {(a * 1000 - 50_000, b) for a, b in baseline} == shifted
