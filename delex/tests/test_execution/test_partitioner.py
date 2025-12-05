"""
Tests for execution.partitioner module.

This module tests graph partitioning functionality.
"""
import pytest
from delex.execution.partitioner import DataFramePartitioner


@pytest.mark.unit
class TestDataFramePartitioner:
    """Tests for DataFramePartitioner class."""

    def test_partitioner_init(self):
        """Test DataFramePartitioner initialization."""
        partitioner = DataFramePartitioner(column='_id', nparts=4)
        assert partitioner is not None
        assert partitioner.column == '_id'
        assert partitioner.nparts == 4

    @pytest.mark.requires_spark
    def test_partitioner_get_partition_with_spark(
            self, spark_session, table_a):
        """Test getting a partition with Spark DataFrame."""
        partitioner = DataFramePartitioner(column='_id', nparts=4)
        partition = partitioner.get_partition(table_a, 0)
        assert partition is not None
        assert hasattr(partition, 'collect')
        partition1 = partitioner.get_partition(table_a, 1)
        assert partition1 is not None
        partition3 = partitioner.get_partition(table_a, 3)
        assert partition3 is not None

    @pytest.mark.requires_spark
    def test_partitioner_get_partition_invalid_pnum(
            self, spark_session, table_a):
        """Test get_partition with invalid partition numbers."""
        partitioner = DataFramePartitioner(column='_id', nparts=4)
        with pytest.raises(ValueError):
            partitioner.get_partition(table_a, -1)
        with pytest.raises(ValueError):
            partitioner.get_partition(table_a, 4)
        with pytest.raises(ValueError):
            partitioner.get_partition(table_a, 10)

    @pytest.mark.requires_spark
    def test_partitioner_single_partition(self, spark_session, table_a):
        """Test partitioner with single partition."""
        partitioner = DataFramePartitioner(column='_id', nparts=1)
        partition = partitioner.get_partition(table_a, 0)
        assert partition is not None
        assert hasattr(partition, 'collect')
        with pytest.raises(ValueError):
            partitioner.get_partition(table_a, 1)
