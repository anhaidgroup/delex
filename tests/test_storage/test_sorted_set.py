"""
Tests for storage.sorted_set module.

This module tests sorted set functionality.
"""
import pytest
import numpy as np
from delex.storage import MemmapSortedSets, MemmapArray, IdOffsetHashMap


@pytest.mark.unit
class TestMemmapSortedSets:
    """Tests for MemmapSortedSets class."""

    def test_memmap_sorted_sets_init(self):
        """Test MemmapSortedSets initialization."""
        sorted_sets = MemmapSortedSets()
        assert sorted_sets._offset_arr is None
        assert sorted_sets._seq_arr is None
        assert sorted_sets._id_to_offset_map is None

    def test_memmap_sorted_sets_fetch(self):
        """Test fetch method with manually constructed object."""
        sorted_sets = MemmapSortedSets()
        seq_data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        sorted_sets._seq_arr = MemmapArray(seq_data)

        offsets = np.array([0, 3, 5], dtype=np.uint64)
        sorted_sets._offset_arr = MemmapArray(offsets)
        id_arr = np.array([1, 2], dtype=np.uint64)
        offset_indices = np.array([0, 1], dtype=np.int32)
        sorted_sets._id_to_offset_map = IdOffsetHashMap.build(id_arr, offset_indices)

        sorted_sets.init()

        result = sorted_sets.fetch(1)
        np.testing.assert_array_equal(result, np.array([1, 2, 3], dtype=np.int32))

        result = sorted_sets.fetch(2)
        np.testing.assert_array_equal(result, np.array([4, 5], dtype=np.int32))

        result = sorted_sets.fetch(999)
        assert result is None

        sorted_sets.deinit()
        sorted_sets.delete()

    def test_memmap_sorted_sets_jaccard(self):
        """Test jaccard method."""
        sorted_sets = MemmapSortedSets()

        seq_data = np.array([1, 2, 3, 2, 3, 4], dtype=np.int32)
        sorted_sets._seq_arr = MemmapArray(seq_data)
        offsets = np.array([0, 3, 6], dtype=np.uint64)
        sorted_sets._offset_arr = MemmapArray(offsets)
        id_arr = np.array([1, 2], dtype=np.uint64)
        offset_indices = np.array([0, 1], dtype=np.int32)
        sorted_sets._id_to_offset_map = IdOffsetHashMap.build(id_arr, offset_indices)

        sorted_sets.init()

        query = np.array([2, 3], dtype=np.int32)
        ids = np.array([1, 2], dtype=np.int64)
        scores = sorted_sets.jaccard(query, ids)

        assert len(scores) == 2
        assert not np.isnan(scores[0])
        assert not np.isnan(scores[1])
        assert scores[0] > 0
        assert scores[1] > 0

        ids2 = np.array([999], dtype=np.int64)
        scores2 = sorted_sets.jaccard(query, ids2)
        assert np.isnan(scores2[0])

        sorted_sets.deinit()
        sorted_sets.delete()

    def test_memmap_sorted_sets_overlap_coeff(self):
        """Test overlap_coeff method."""
        sorted_sets = MemmapSortedSets()

        seq_data = np.array([1, 2, 3, 2, 3, 4], dtype=np.int32)
        sorted_sets._seq_arr = MemmapArray(seq_data)
        offsets = np.array([0, 3, 6], dtype=np.uint64)
        sorted_sets._offset_arr = MemmapArray(offsets)
        id_arr = np.array([1, 2], dtype=np.uint64)
        offset_indices = np.array([0, 1], dtype=np.int32)
        sorted_sets._id_to_offset_map = IdOffsetHashMap.build(id_arr, offset_indices)

        sorted_sets.init()

        query = np.array([2, 3], dtype=np.int32)
        ids = np.array([1, 2], dtype=np.int64)
        scores = sorted_sets.overlap_coeff(query, ids)

        assert len(scores) == 2
        assert isinstance(scores, np.ndarray)

        ids2 = np.array([999], dtype=np.int64)
        scores2 = sorted_sets.overlap_coeff(query, ids2)
        assert scores2[0] == 0.0

        sorted_sets.deinit()
        sorted_sets.delete()

    def test_memmap_sorted_sets_cosine(self):
        """Test cosine method."""
        sorted_sets = MemmapSortedSets()

        seq_data = np.array([1, 2, 3, 2, 3, 4], dtype=np.int32)
        sorted_sets._seq_arr = MemmapArray(seq_data)

        offsets = np.array([0, 3, 6], dtype=np.uint64)
        sorted_sets._offset_arr = MemmapArray(offsets)

        id_arr = np.array([1, 2], dtype=np.uint64)
        offset_indices = np.array([0, 1], dtype=np.int32)
        sorted_sets._id_to_offset_map = IdOffsetHashMap.build(id_arr, offset_indices)

        sorted_sets.init()

        query = np.array([2, 3], dtype=np.int32)
        ids = np.array([1, 2], dtype=np.int64)
        scores = sorted_sets.cosine(query, ids)

        assert len(scores) == 2
        assert not np.isnan(scores[0])
        assert not np.isnan(scores[1])
        assert scores[0] > 0
        assert scores[1] > 0
        sorted_sets.deinit()
        sorted_sets.delete()

    def test_memmap_sorted_sets_size_in_bytes(self):
        """Test size_in_bytes method."""
        sorted_sets = MemmapSortedSets()

        seq_data = np.array([1, 2, 3], dtype=np.int32)
        sorted_sets._seq_arr = MemmapArray(seq_data)

        offsets = np.array([0, 3], dtype=np.uint64)
        sorted_sets._offset_arr = MemmapArray(offsets)

        id_arr = np.array([1], dtype=np.uint64)
        offset_indices = np.array([0], dtype=np.int32)
        sorted_sets._id_to_offset_map = IdOffsetHashMap.build(id_arr, offset_indices)

        size = sorted_sets.size_in_bytes()
        assert size > 0
        sorted_sets.delete()

    @pytest.mark.requires_spark
    def test_memmap_sorted_sets_build(self, spark_session):
        """Test MemmapSortedSets build from Spark DataFrame."""
        import pyspark.sql.types as T
        schema = T.StructType([
            T.StructField('id', T.IntegerType()),
            T.StructField('tokens', T.ArrayType(T.IntegerType()))
        ])
        df = spark_session.createDataFrame(
            [(1, [1, 2, 3]),
             (2, [2, 3, 4]),
             (3, [5, 6])],
            schema=schema)
        sorted_sets = MemmapSortedSets.build(df, 'tokens', 'id')
        assert sorted_sets._offset_arr is not None
        assert sorted_sets._seq_arr is not None
        assert sorted_sets._id_to_offset_map is not None
        sorted_sets.delete()
