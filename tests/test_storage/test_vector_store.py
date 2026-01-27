"""
Tests for storage.vector_store module.

This module tests vector storage functionality.
"""
import pytest
import numpy as np
from delex.storage.vector_store import (
    MemmapVectorStore,
    _sparse_dot,
    _sparse_dot_many
)
from delex.storage import MemmapArray, IdOffsetHashMap


@pytest.mark.unit
class TestMemmapVectorStore:
    """Tests for MemmapVectorStore class."""

    def test_arrays_to_encoded_sparse_vector(self):
        """Test encoding sparse vectors."""
        ind = np.array([0, 2, 5], dtype=np.int32)
        val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        encoded = MemmapVectorStore.arrays_to_encoded_sparse_vector(ind, val)
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

    def test_decode_sparse_vector(self):
        """Test decoding sparse vectors."""
        ind = np.array([0, 2, 5], dtype=np.int32)
        val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        encoded = MemmapVectorStore.arrays_to_encoded_sparse_vector(ind, val)
        decoded = MemmapVectorStore.decode_sparse_vector(encoded)
        assert len(decoded) == 3
        np.testing.assert_array_equal(decoded['ind'], ind)
        np.testing.assert_array_equal(decoded['val'], val)

    def test_fetch(self):
        """Test fetch method."""
        vs = MemmapVectorStore()
        vec = np.array([(0, 1.0), (2, 2.0)], dtype=vs.vector_dtype)
        vs._seq_arr = MemmapArray(vec)
        vs._offset_arr = MemmapArray(np.array([0, 2], dtype=np.uint64))
        vs._id_to_offset_map = IdOffsetHashMap.build(
            np.array([1], dtype=np.uint64),
            np.array([0], dtype=np.int32))
        vs.init()

        result = vs.fetch(1)
        assert result is not None
        assert len(result) == 2

        assert vs.fetch(999) is None

        vs.deinit()
        vs.delete()

    def test_dot(self):
        """Test dot product."""
        vs = MemmapVectorStore()
        vec1 = np.array([(0, 1.0), (2, 2.0)], dtype=vs.vector_dtype)
        vec2 = np.array([(1, 3.0), (2, 4.0)], dtype=vs.vector_dtype)
        vs._seq_arr = MemmapArray(np.concatenate([vec1, vec2]))
        vs._offset_arr = MemmapArray(np.array([0, 2, 4], dtype=np.uint64))
        vs._id_to_offset_map = IdOffsetHashMap.build(
            np.array([1, 2], dtype=np.uint64),
            np.array([0, 1], dtype=np.int32))
        vs.init()

        query = np.array([(0, 1.0), (2, 1.0)], dtype=vs.vector_dtype)
        scores = vs.dot(query, np.array([1, 2, 999], dtype=np.int64))

        assert len(scores) == 3
        assert not np.isnan(scores[0])
        assert not np.isnan(scores[1])
        assert np.isnan(scores[2])

        vs.deinit()
        vs.delete()

    def test_dot_empty_cases(self):
        """Test dot with empty query and empty vector."""
        vs = MemmapVectorStore()
        vec = np.array([(0, 1.0)], dtype=vs.vector_dtype)
        vs._seq_arr = MemmapArray(vec)
        vs._offset_arr = MemmapArray(np.array([0, 0, 1], dtype=np.uint64))
        vs._id_to_offset_map = IdOffsetHashMap.build(
            np.array([1], dtype=np.uint64),
            np.array([0], dtype=np.int32))
        vs.init()

        # Empty query
        scores = vs.dot(np.array([], dtype=vs.vector_dtype),
                        np.array([1], dtype=np.int64))
        assert scores[0] == 0.0

        # Empty vector (start == end)
        query = np.array([(0, 1.0)], dtype=vs.vector_dtype)
        scores = vs.dot(query, np.array([1], dtype=np.int64))
        assert scores[0] == 0.0

        vs.deinit()
        vs.delete()

    def test_size_in_bytes(self):
        """Test size_in_bytes method."""
        vs = MemmapVectorStore()
        vec = np.array([(0, 1.0)], dtype=vs.vector_dtype)
        vs._seq_arr = MemmapArray(vec)
        vs._offset_arr = MemmapArray(np.array([0, 1], dtype=np.uint64))
        vs._id_to_offset_map = IdOffsetHashMap.build(
            np.array([1], dtype=np.uint64),
            np.array([0], dtype=np.int32))
        assert vs.size_in_bytes() > 0
        vs.delete()

    def test_init_deinit(self):
        """Test init and deinit methods."""
        vs = MemmapVectorStore()
        vec = np.array([(0, 1.0)], dtype=vs.vector_dtype)
        vs._seq_arr = MemmapArray(vec)
        vs._offset_arr = MemmapArray(np.array([0, 1], dtype=np.uint64))
        vs._id_to_offset_map = IdOffsetHashMap.build(
            np.array([1], dtype=np.uint64),
            np.array([0], dtype=np.int32))

        vs.deinit()
        assert vs._seq_arr._mmap_arr is None

        vs.init()
        assert vs._seq_arr._mmap_arr is not None
        vs.delete()

    def test_to_spark(self):
        """Test converting to Spark format."""
        vs = MemmapVectorStore()
        vec = np.array([(0, 1.0)], dtype=vs.vector_dtype)
        vs._seq_arr = MemmapArray(vec)
        vs._offset_arr = MemmapArray(np.array([0, 1], dtype=np.uint64))
        vs._id_to_offset_map = IdOffsetHashMap.build(
            np.array([1], dtype=np.uint64),
            np.array([0], dtype=np.int32))

        assert not vs._seq_arr._on_spark
        vs.to_spark()
        assert vs._seq_arr._on_spark
        vs.delete()

    def test_sparse_dot(self):
        """Test _sparse_dot function."""
        dtype = MemmapVectorStore.vector_dtype

        # Matching indices
        l_vec = np.array([(0, 1.0), (1, 2.0)], dtype=dtype)
        r_vec = np.array([(0, 3.0), (1, 4.0)], dtype=dtype)
        result = _sparse_dot(l_vec, r_vec)
        assert abs(result - (1.0 * 3.0 + 2.0 * 4.0)) < 1e-5

        # Partial overlap
        l_vec = np.array([(0, 1.0), (2, 2.0)], dtype=dtype)
        r_vec = np.array([(1, 3.0), (2, 4.0)], dtype=dtype)
        result = _sparse_dot(l_vec, r_vec)
        assert abs(result - (2.0 * 4.0)) < 1e-5

        # No overlap
        l_vec = np.array([(0, 1.0)], dtype=dtype)
        r_vec = np.array([(2, 2.0)], dtype=dtype)
        assert _sparse_dot(l_vec, r_vec) == 0.0

        # Empty cases
        assert _sparse_dot(
            np.array([], dtype=dtype),
            np.array([(0, 1.0)], dtype=dtype)) == 0.0
        assert _sparse_dot(
            np.array([(0, 1.0)], dtype=dtype),
            np.array([], dtype=dtype)) == 0.0

        # Different advance patterns
        l_vec = np.array([(0, 1.0), (2, 2.0)], dtype=dtype)
        r_vec = np.array([(2, 3.0)], dtype=dtype)
        assert abs(_sparse_dot(l_vec, r_vec) - 6.0) < 1e-5

        l_vec = np.array([(2, 2.0)], dtype=dtype)
        r_vec = np.array([(0, 1.0), (2, 3.0)], dtype=dtype)
        assert abs(_sparse_dot(l_vec, r_vec) - 6.0) < 1e-5

    def test_sparse_dot_many(self):
        """Test _sparse_dot_many function."""
        dtype = MemmapVectorStore.vector_dtype

        # Basic functionality
        offsets = np.array([0, 2, 4], dtype=np.uint64)
        vec1 = np.array([(0, 1.0), (2, 2.0)], dtype=dtype)
        vec2 = np.array([(1, 3.0), (2, 4.0)], dtype=dtype)
        vecs = np.concatenate([vec1, vec2])
        query = np.array([(0, 1.0), (2, 1.0)], dtype=dtype)
        indexes = np.array([0, 1, -1], dtype=np.int32)
        scores = _sparse_dot_many(offsets, vecs, query, indexes)
        assert len(scores) == 3
        assert not np.isnan(scores[0])
        assert not np.isnan(scores[1])
        assert np.isnan(scores[2])

        # Empty query
        scores = _sparse_dot_many(offsets, vecs, np.array([], dtype=dtype),
                                  np.array([0], dtype=np.int32))
        assert scores[0] == 0.0

        # Empty vector
        offsets = np.array([0, 0, 2], dtype=np.uint64)
        query = np.array([(0, 1.0)], dtype=dtype)
        scores = _sparse_dot_many(
            offsets, vecs, query, np.array([0], dtype=np.int32))
        assert scores[0] == 0.0

    @pytest.mark.requires_spark
    def test_build(self, spark_session):
        """Test build method."""
        dtype = MemmapVectorStore.vector_dtype
        vec1 = np.array([(0, 1.0), (2, 2.0)], dtype=dtype)
        vec2 = np.array([(1, 3.0), (2, 4.0)], dtype=dtype)
        vec3 = np.array([(0, 5.0)], dtype=dtype)

        data = [
            (1, vec1.tobytes()),
            (2, vec2.tobytes()),
            (3, vec3.tobytes())
        ]
        df = spark_session.createDataFrame(data, ['_id', 'vec'])

        vector_store = MemmapVectorStore.build(df, seq_col='vec', id_col='_id')

        assert vector_store._seq_arr is not None
        assert vector_store._offset_arr is not None
        assert vector_store._id_to_offset_map is not None

        vector_store.init()
        assert vector_store.fetch(1) is not None
        assert vector_store.fetch(2) is not None
        assert vector_store.fetch(3) is not None

        query = np.array([(0, 1.0), (2, 1.0)], dtype=dtype)
        scores = vector_store.dot(query, np.array([1, 2, 3], dtype=np.int64))
        assert len(scores) == 3
        assert not np.isnan(scores[0])

        vector_store.deinit()
        vector_store.delete()

    @pytest.mark.requires_spark
    def test_build_filters_nulls(self, spark_session):
        """Test build filters out null values."""
        dtype = MemmapVectorStore.vector_dtype
        vec1 = np.array([(0, 1.0)], dtype=dtype)
        vec2 = np.array([(1, 2.0)], dtype=dtype)

        data = [
            (1, vec1.tobytes()),
            (2, None),
            (3, vec2.tobytes())
        ]
        df = spark_session.createDataFrame(data, ['_id', 'vec'])

        vector_store = MemmapVectorStore.build(df, seq_col='vec', id_col='_id')

        vector_store.init()
        assert vector_store.fetch(1) is not None
        assert vector_store.fetch(3) is not None

        vector_store.deinit()
        vector_store.delete()
