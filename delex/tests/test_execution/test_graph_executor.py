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
