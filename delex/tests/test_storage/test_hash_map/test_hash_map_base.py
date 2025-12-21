"""
Tests for storage.hash_map.hash_map_base module.

This module tests hash map base functionality.
"""
import pytest
import numpy as np
from delex.storage.hash_map.hash_map_base import DistributableHashMap


@pytest.mark.unit
class TestDistributableHashMap:
    """Tests for DistributableHashMap class."""

    def test_distributable_hash_map_init(self):
        """Test DistributableHashMap initialization."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        hash_map = DistributableHashMap(arr)
        assert hash_map._memmap_arr is not None
        np.testing.assert_array_equal(hash_map._arr, arr)
        hash_map._memmap_arr.delete()

    def test_distributable_hash_map_size_in_bytes(self):
        """Test size_in_bytes method."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        hash_map = DistributableHashMap(arr)
        size = hash_map.size_in_bytes()
        assert size > 0
        hash_map._memmap_arr.delete()

    def test_distributable_hash_map_init_deinit(self):
        """Test init and deinit methods."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        hash_map = DistributableHashMap(arr)
        hash_map.deinit()
        assert hash_map._memmap_arr._mmap_arr is None

        hash_map.init()
        assert hash_map._memmap_arr._mmap_arr is not None
        hash_map._memmap_arr.delete()

    def test_distributable_hash_map_to_spark(self):
        """Test converting to Spark format."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        hash_map = DistributableHashMap(arr)
        assert hash_map._memmap_arr._on_spark is False
        hash_map.to_spark()
        assert hash_map._memmap_arr._on_spark is True
        hash_map._memmap_arr.delete()

    def test_distributable_hash_map_allocate_map(self):
        """Test _allocate_map static method."""
        arr = DistributableHashMap._allocate_map(10, 0.5, np.int32)
        assert len(arr) >= 10
        assert len(arr) % 2 == 1  # Should be odd

        arr2 = DistributableHashMap._allocate_map(10, 0.75, np.float32)
        assert len(arr2) >= 10
