"""
Tests for index.hash_index module.

This module tests hash-based indexing functionality.
"""
import pytest
import numpy as np
import pandas as pd
from pyspark.sql import Row
from delex.index.hash_index import HashIndex


@pytest.mark.unit
class TestHashIndex:
    """Tests for HashIndex class."""

    def test_hash_index_init(self):
        """Test HashIndex initialization."""
        hash_index = HashIndex()
        assert hash_index is not None
        assert hash_index._id_lists is None
        assert hash_index._offset_arr is None
        assert hash_index._string_to_idx is None

    def test_hash_index_build(self, spark_session):
        """Test building the hash index from Spark DataFrame."""
        hash_index = HashIndex()
        hash_index.build(spark_session.createDataFrame([
            (1, 'apple'),
            (2, 'banana'),
            (3, 'cherry'),
        ], ['_id', 'key']), 'key', '_id')
        assert hash_index._id_lists is not None
        assert hash_index._offset_arr is not None
        assert hash_index._string_to_idx is not None

    def test_hash_index_fetch_existing(self, spark_session):
        """Test fetching existing keys."""
        hash_index = HashIndex()
        hash_index.build(spark_session.createDataFrame([
            (1, 'apple'),
            (2, 'banana'),
            (3, 'cherry'),
        ], ['_id', 'key']), 'key', '_id')
        assert np.array_equal(hash_index.fetch('apple'), np.array([1]))
        assert np.array_equal(hash_index.fetch('banana'), np.array([2]))
        assert np.array_equal(hash_index.fetch('cherry'), np.array([3]))

    def test_hash_index_fetch_nonexistent(self, spark_session):
        """Test fetching non-existent keys."""
        hash_index = HashIndex()
        hash_index.build(spark_session.createDataFrame([
            (1, 'apple'),
            (2, 'banana'),
            (3, 'cherry'),
        ], ['_id', 'key']), 'key', '_id')
        assert hash_index.fetch('pineapple') is None

    def test_hash_index_size_in_bytes(self, spark_session):
        """Test size in bytes method."""
        hash_index = HashIndex()
        hash_index.build(spark_session.createDataFrame([
            (1, 'apple'),
            (2, 'banana'),
            (3, 'cherry'),
        ], ['_id', 'key']), 'key', '_id')
        assert hash_index.size_in_bytes() > 0
