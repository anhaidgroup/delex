"""
Tests for delex.utils.checks module.
"""
import pandas as pd
import pytest

from delex.utils.checks import check_tables, _check_id_spark


@pytest.mark.unit
class TestCheckTables:
    """Tests for check_tables."""

    def test_valid_tables(self, spark_session):
        """Valid Spark tables with integer id columns pass."""
        table_a = spark_session.createDataFrame([(1, "a"), (2, "b")], ["_id", "x"])
        table_b = spark_session.createDataFrame([(1, "a"), (2, "b")], ["_id", "x"])
        check_tables(table_a, "_id", table_b, "_id")  # no raise

    def test_table_a_missing_id_column(self, spark_session):
        """table_a missing id column raises ValueError."""
        table_a = spark_session.createDataFrame([(1, "a"), (2, "b")], ["id_wrong", "x"])
        table_b = spark_session.createDataFrame([(1, "a")], ["_id", "x"])
        with pytest.raises(ValueError, match="table_a: missing id column '_id'"):
            check_tables(table_a, "_id", table_b, "_id")

    def test_table_b_missing_id_column(self, spark_session):
        """table_b missing id column raises ValueError."""
        table_a = spark_session.createDataFrame([(1, "a"), (2, "b")], ["_id", "x"])
        table_b = spark_session.createDataFrame([(1, "a")], ["id_wrong", "x"])
        with pytest.raises(ValueError, match="table_b: missing id column '_id'"):
            check_tables(table_a, "_id", table_b, "_id")

    def test_table_a_empty(self, spark_session):
        """Empty table_a raises ValueError."""
        table_a = spark_session.createDataFrame([], "_id BIGINT, x STRING")
        table_b = spark_session.createDataFrame([(1, "a")], ["_id", "x"])
        with pytest.raises(ValueError, match="table_a: empty dataframe"):
            check_tables(table_a, "_id", table_b, "_id")

    def test_table_b_empty(self, spark_session):
        """Empty table_b raises ValueError."""
        table_a = spark_session.createDataFrame([(1, "a")], ["_id", "x"])
        table_b = spark_session.createDataFrame([], "_id BIGINT, x STRING")
        with pytest.raises(ValueError, match="table_b: empty dataframe"):
            check_tables(table_a, "_id", table_b, "_id")

    def test_table_a_id_not_integer_type(self, spark_session):
        """Non-integer id column in table_a raises ValueError."""
        table_a = spark_session.createDataFrame([(1.0, "a"), (2.0, "b")], ["_id", "x"])
        table_b = spark_session.createDataFrame([(1, "a")], ["_id", "x"])
        with pytest.raises(ValueError, match="table_a: id column '_id' must be an integer type"):
            check_tables(table_a, "_id", table_b, "_id")

    def test_table_a_nulls_in_id(self, spark_session):
        """Nulls in table_a id column raises ValueError."""
        table_a = spark_session.createDataFrame([(1, "a"), (None, "b"), (3, "c")], ["_id", "x"])
        table_b = spark_session.createDataFrame([(1, "a")], ["_id", "x"])
        with pytest.raises(ValueError, match="table_a: nulls are present"):
            check_tables(table_a, "_id", table_b, "_id")

    def test_table_a_id_not_unique(self, spark_session):
        """Non-unique id column in table_a raises ValueError."""
        table_a = spark_session.createDataFrame([(1, "a"), (1, "b"), (3, "c")], ["_id", "x"])
        table_b = spark_session.createDataFrame([(1, "a")], ["_id", "x"])
        with pytest.raises(ValueError, match="table_a: id column '_id' must be unique"):
            check_tables(table_a, "_id", table_b, "_id")


@pytest.mark.unit
class TestCheckIdSpark:
    """Tests for _check_id (unit-level)."""

    def test_missing_id_column(self, spark_session):
        df = spark_session.createDataFrame([(1, "a")], ["x", "y"])
        with pytest.raises(ValueError, match="mytable: missing id column 'id'"):
            _check_id_spark(df, "id", "mytable")

    def test_empty_dataframe(self, spark_session):
        df = spark_session.createDataFrame([], "id BIGINT, x STRING")
        with pytest.raises(ValueError, match="mytable: empty dataframe"):
            _check_id_spark(df, "id", "mytable")

    def test_id_not_integer_type(self, spark_session):
        df = spark_session.createDataFrame([(1.0, "a")], ["id", "x"])
        with pytest.raises(ValueError, match="mytable: id column 'id' must be an integer type"):
            _check_id_spark(df, "id", "mytable")

    def test_nulls_in_id(self, spark_session):
        df = spark_session.createDataFrame([(1, "a"), (None, "b")], ["id", "x"])
        with pytest.raises(ValueError, match="mytable: nulls are present"):
            _check_id_spark(df, "id", "mytable")

    def test_id_not_unique(self, spark_session):
        df = spark_session.createDataFrame([(1, "a"), (1, "b")], ["id", "x"])
        with pytest.raises(ValueError, match="mytable: id column 'id' must be unique"):
            _check_id_spark(df, "id", "mytable")

    def test_valid_passes(self, spark_session):
        df = spark_session.createDataFrame([(1, "a"), (2, "b")], ["id", "x"])
        _check_id_spark(df, "id", "mytable")  # no raise
