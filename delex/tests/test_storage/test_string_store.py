"""
Tests for storage.string_store module.

This module tests string storage functionality.
"""
import pytest
import numpy as np
from delex.storage import MemmapStrings, MemmapArray, IdOffsetHashMap


@pytest.mark.unit
class TestMemmapStrings:
    """Tests for MemmapStrings class."""

    def test_memmap_strings_fetch(self):
        """Test fetch method with manually constructed object."""
        strings = MemmapStrings()
        hello_bytes = b'hello'
        world_bytes = b'world'
        seq_data = np.frombuffer(hello_bytes + world_bytes, dtype=np.uint8)
        strings._seq_arr = MemmapArray(seq_data)
        offsets = np.array([0, 5, 10], dtype=np.uint64)
        strings._offset_arr = MemmapArray(offsets)
        id_arr = np.array([1, 2], dtype=np.uint64)
        offset_indices = np.array([0, 1], dtype=np.int32)
        strings._id_to_offset_map = IdOffsetHashMap.build(id_arr, offset_indices)

        strings.init()

        result = strings.fetch(1)
        assert result == 'hello'

        result = strings.fetch(2)
        assert result == 'world'

        result = strings.fetch(999)
        assert result is None

        strings.deinit()
        strings.delete()

    def test_memmap_strings_fetch_bytes(self):
        """Test fetch_bytes method."""
        strings = MemmapStrings()

        hello_bytes = b'hello'
        world_bytes = b'world'
        seq_data = np.frombuffer(hello_bytes + world_bytes, dtype=np.uint8)
        strings._seq_arr = MemmapArray(seq_data)

        offsets = np.array([0, 5, 10], dtype=np.uint64)
        strings._offset_arr = MemmapArray(offsets)

        id_arr = np.array([1, 2], dtype=np.uint64)
        offset_indices = np.array([0, 1], dtype=np.int32)
        strings._id_to_offset_map = IdOffsetHashMap.build(id_arr, offset_indices)

        strings.init()

        result = strings.fetch_bytes(1)
        assert result == b'hello'

        result = strings.fetch_bytes(2)
        assert result == b'world'

        result = strings.fetch_bytes(999)
        assert result is None

        strings.deinit()
        strings.delete()

    def test_memmap_strings_size_in_bytes(self):
        """Test size_in_bytes method."""
        strings = MemmapStrings()

        hello_bytes = b'hello'
        seq_data = np.frombuffer(hello_bytes, dtype=np.uint8)
        strings._seq_arr = MemmapArray(seq_data)

        offsets = np.array([0, 5], dtype=np.uint64)
        strings._offset_arr = MemmapArray(offsets)

        id_arr = np.array([1], dtype=np.uint64)
        offset_indices = np.array([0], dtype=np.int32)
        strings._id_to_offset_map = IdOffsetHashMap.build(id_arr, offset_indices)

        size = strings.size_in_bytes()
        assert size > 0
        strings.delete()

    def test_memmap_strings_to_spark(self):
        """Test converting to Spark format."""
        strings = MemmapStrings()

        hello_bytes = b'hello'
        seq_data = np.frombuffer(hello_bytes, dtype=np.uint8)
        strings._seq_arr = MemmapArray(seq_data)

        offsets = np.array([0, 5], dtype=np.uint64)
        strings._offset_arr = MemmapArray(offsets)

        id_arr = np.array([1], dtype=np.uint64)
        offset_indices = np.array([0], dtype=np.int32)
        strings._id_to_offset_map = IdOffsetHashMap.build(id_arr, offset_indices)

        strings.to_spark()
        assert strings._seq_arr._on_spark is True
        strings.delete()

    @pytest.mark.requires_spark
    def test_memmap_strings_build(self, spark_session):
        """Test MemmapStrings build from Spark DataFrame."""
        df = spark_session.createDataFrame(
            [(1, 'hello'),
             (2, 'world'),
             (3, 'test')],
            ['id', 'text'])

        strings = MemmapStrings.build(df, 'text', 'id')
        assert strings._offset_arr is not None
        assert strings._seq_arr is not None
        assert strings._id_to_offset_map is not None
        strings.delete()
