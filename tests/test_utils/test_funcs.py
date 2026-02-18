"""
Tests for delex.utils.funcs module.
"""
import pytest
from pyspark import SparkContext

from delex.utils.funcs import get_num_partitions


@pytest.mark.unit
class TestGetNumPartitions:
    """Tests for get_num_partitions function.

    These tests exercise the behavior of get_num_partitions using real Spark
    DataFrames, without mocking internal Spark statistics. The function:

    - Always uses at least `cores * 2` partitions, where `cores` is
      `SparkContext.defaultParallelism`.
    - Optionally increases the partition count when Spark provides a
      meaningful `sizeInBytes` estimate; however, Spark is free to return
      placeholder stats, so tests MUST NOT assume a particular size-based
      partition count.
    """

    @pytest.mark.requires_spark
    def test_small_dataframe_returns_at_least_cores_times_two(self, spark_session):
        """Small DataFrame: result is an int and >= cores * 2."""
        df = spark_session.createDataFrame([(1, "a"), (2, "b")], ["_id", "x"])

        partitions = get_num_partitions(df)

        spark_context = SparkContext.getOrCreate()
        cores = spark_context.defaultParallelism
        expected_min = cores * 2

        assert isinstance(partitions, int)
        assert partitions >= expected_min

    @pytest.mark.requires_spark
    def test_larger_dataframe_still_respects_minimum(self, spark_session):
        """Larger DataFrame: still at least cores * 2 (may or may not be larger)."""
        data = [(i, f"value_{i}", f"data_{i}") for i in range(10_000)]
        df = spark_session.createDataFrame(data, ["_id", "x", "y"])

        partitions = get_num_partitions(df)

        spark_context = SparkContext.getOrCreate()
        cores = spark_context.defaultParallelism
        expected_min = cores * 2

        assert isinstance(partitions, int)
        assert partitions >= expected_min

    @pytest.mark.requires_spark
    def test_different_schemas_do_not_break_function(self, spark_session):
        """Function works for various schemas and always respects minimum."""
        df_int = spark_session.createDataFrame([(1, 10), (2, 20)], ["_id", "value"])
        df_str = spark_session.createDataFrame([(1, "a"), (2, "b")], ["_id", "name"])
        df_mixed = spark_session.createDataFrame(
            [(1, "a", 10.5), (2, "b", 20.3)],
            ["_id", "name", "score"],
        )

        p_int = get_num_partitions(df_int)
        p_str = get_num_partitions(df_str)
        p_mixed = get_num_partitions(df_mixed)

        spark_context = SparkContext.getOrCreate()
        cores = spark_context.defaultParallelism
        expected_min = cores * 2

        for p in (p_int, p_str, p_mixed):
            assert isinstance(p, int)
            assert p >= expected_min

    @pytest.mark.requires_spark
    def test_repartitioned_dataframe_still_uses_heuristic(self, spark_session):
        """Pre-existing Spark partitioning doesn't break the heuristic."""
        df = spark_session.createDataFrame(
            [(i, f"value_{i}") for i in range(1_000)],
            ["_id", "x"],
        )

        df_repartitioned = df.repartition(10)
        partitions = get_num_partitions(df_repartitioned)

        spark_context = SparkContext.getOrCreate()
        cores = spark_context.defaultParallelism
        expected_min = cores * 2

        assert isinstance(partitions, int)
        assert partitions >= expected_min

    @pytest.mark.requires_spark
    def test_consistent_for_same_dataframe(self, spark_session):
        """Calling get_num_partitions twice on same DF yields same result."""
        df = spark_session.createDataFrame(
            [(i, f"value_{i}") for i in range(1_000)],
            ["_id", "x"],
        )

        p1 = get_num_partitions(df)
        p2 = get_num_partitions(df)

        assert isinstance(p1, int)
        assert p1 == p2

    @pytest.mark.requires_spark
    def test_wide_dataframe(self, spark_session):
        """Wide DataFrame (many columns) still works and respects minimum."""
        data = [(i, *[f"col_{j}_{i}" for j in range(20)]) for i in range(1_000)]
        columns = ["_id"] + [f"col_{j}" for j in range(20)]
        df = spark_session.createDataFrame(data, columns)

        partitions = get_num_partitions(df)

        spark_context = SparkContext.getOrCreate()
        cores = spark_context.defaultParallelism
        expected_min = cores * 2

        assert isinstance(partitions, int)
        assert partitions >= expected_min

    @pytest.mark.requires_spark
    def test_empty_dataframe(self, spark_session):
        """Empty DataFrame: still returns a sensible minimum."""
        df = spark_session.createDataFrame([], "_id INT, x STRING")

        partitions = get_num_partitions(df)

        spark_context = SparkContext.getOrCreate()
        cores = spark_context.defaultParallelism
        expected_min = cores * 2

        assert isinstance(partitions, int)
        assert partitions >= expected_min
