"""
Tests for storage.memmap_seqs module.

This module tests memory-mapped sequences functionality.
"""
import pytest
import numpy as np
from delex.storage.memmap_seqs import MemmapSeqs
from delex.storage import MemmapArray, IdOffsetHashMap


@pytest.mark.unit
class TestMemmapSeqs:
    """Tests for MemmapSeqs class."""

    def test_memmap_seqs_init(self):
        """Test MemmapSeqs initialization."""
        seqs = MemmapSeqs()
        assert seqs._offset_arr is None
        assert seqs._seq_arr is None
        assert seqs._id_to_offset_map is None

    def test_memmap_seqs_fetch(self):
        """Test fetch method with manually constructed object."""
        seqs = MemmapSeqs()
        seq_data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        seqs._seq_arr = MemmapArray(seq_data)
        offsets = np.array([0, 3, 5], dtype=np.uint64)
        seqs._offset_arr = MemmapArray(offsets)
        id_arr = np.array([1, 2], dtype=np.uint64)
        offset_indices = np.array([0, 1], dtype=np.int32)
        seqs._id_to_offset_map = IdOffsetHashMap.build(id_arr, offset_indices)

        seqs.init()

        result = seqs.fetch(1)
        expected = np.array([1, 2, 3], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

        result = seqs.fetch(2)
        np.testing.assert_array_equal(result, np.array([4, 5], dtype=np.int32))

        result = seqs.fetch(999)
        assert result is None

        seqs.deinit()
        seqs.delete()

    def test_memmap_seqs_size_in_bytes(self):
        """Test size_in_bytes method."""
        seqs = MemmapSeqs()

        seq_data = np.array([1, 2, 3], dtype=np.int32)
        seqs._seq_arr = MemmapArray(seq_data)

        offsets = np.array([0, 3], dtype=np.uint64)
        seqs._offset_arr = MemmapArray(offsets)

        id_arr = np.array([1], dtype=np.uint64)
        offset_indices = np.array([0], dtype=np.int32)
        seqs._id_to_offset_map = IdOffsetHashMap.build(id_arr, offset_indices)

        size = seqs.size_in_bytes()
        assert size > 0
        seqs.delete()

    def test_memmap_seqs_init_deinit(self):
        """Test init and deinit methods."""
        seqs = MemmapSeqs()

        seq_data = np.array([1, 2, 3], dtype=np.int32)
        seqs._seq_arr = MemmapArray(seq_data)

        offsets = np.array([0, 3], dtype=np.uint64)
        seqs._offset_arr = MemmapArray(offsets)

        id_arr = np.array([1], dtype=np.uint64)
        offset_indices = np.array([0], dtype=np.int32)
        seqs._id_to_offset_map = IdOffsetHashMap.build(id_arr, offset_indices)

        seqs.init()
        assert seqs._seq_arr._mmap_arr is not None
        assert seqs._offset_arr._mmap_arr is not None

        seqs.deinit()
        assert seqs._seq_arr._mmap_arr is None
        assert seqs._offset_arr._mmap_arr is None
        seqs.delete()

    def test_memmap_seqs_to_spark(self):
        """Test converting to Spark format."""
        seqs = MemmapSeqs()

        seq_data = np.array([1, 2, 3], dtype=np.int32)
        seqs._seq_arr = MemmapArray(seq_data)

        offsets = np.array([0, 3], dtype=np.uint64)
        seqs._offset_arr = MemmapArray(offsets)

        id_arr = np.array([1], dtype=np.uint64)
        offset_indices = np.array([0], dtype=np.int32)
        seqs._id_to_offset_map = IdOffsetHashMap.build(id_arr, offset_indices)

        seqs.to_spark()
        assert seqs._seq_arr._on_spark is True
        assert seqs._offset_arr._on_spark is True
        seqs.delete()

    def test_memmap_seqs_delete(self):
        """Test delete method."""
        seqs = MemmapSeqs()

        seq_data = np.array([1, 2, 3], dtype=np.int32)
        seqs._seq_arr = MemmapArray(seq_data)

        offsets = np.array([0, 3], dtype=np.uint64)
        seqs._offset_arr = MemmapArray(offsets)

        id_arr = np.array([1], dtype=np.uint64)
        offset_indices = np.array([0], dtype=np.int32)
        seqs._id_to_offset_map = IdOffsetHashMap.build(id_arr, offset_indices)

        seqs.delete()
        assert not seqs._seq_arr._local_mmap_file.exists()
        assert not seqs._offset_arr._local_mmap_file.exists()

    @pytest.mark.requires_spark
    def test_memmap_seqs_build(self, spark_session):
        """Test creating MemmapSeqs from Spark DataFrame."""
        import pyspark.sql.types as T
        import time

        try:
            spark_context = spark_session.sparkContext
            _ = spark_context.parallelize([1]).collect()
        except Exception:
            pass
        time.sleep(0.5)

        schema = T.StructType([
            T.StructField('id', T.IntegerType()),
            T.StructField('seq', T.ArrayType(T.IntegerType()))
        ])
        df = spark_session.createDataFrame(
            [(1, [1, 2, 3]),
             (2, [4, 5]),
             (3, [6, 7, 8, 9])],
            schema=schema)

        seqs = MemmapSeqs.build(df, 'seq', np.int32, 'id')
        assert seqs._offset_arr is not None
        assert seqs._seq_arr is not None
        assert seqs._id_to_offset_map is not None
        seqs.delete()
