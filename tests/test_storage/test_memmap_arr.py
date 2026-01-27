"""
Tests for storage.memmap_arr module.

This module tests memory-mapped array functionality.
"""
import pytest
import numpy as np
from delex.storage import MemmapArray


@pytest.mark.unit
class TestMemmapArray:
    """Tests for MemmapArray class."""

    def test_memmap_array_init_from_array(self):
        """Test MemmapArray initialization from numpy array."""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        memmap_arr = MemmapArray(arr)
        assert memmap_arr.shape == arr.shape
        assert memmap_arr.values.dtype == arr.dtype
        assert len(memmap_arr) == 5
        np.testing.assert_array_equal(memmap_arr.values, arr)
        memmap_arr.delete()

    def test_memmap_array_init_from_memmap(self):
        """Test MemmapArray initialization from existing memmap."""
        arr = np.array([1, 2, 3], dtype=np.float32)
        memmap_arr1 = MemmapArray(arr)
        memmap_file = memmap_arr1._local_mmap_file
        memmap_arr1.delete()
        
        # Create new memmap from file
        mmap = np.memmap(memmap_file, shape=(3,), dtype=np.float32, mode='w+')
        mmap[:] = arr[:]
        memmap_arr2 = MemmapArray(mmap)
        np.testing.assert_array_equal(memmap_arr2.values, arr)
        memmap_arr2.delete()

    def test_memmap_array_size_in_bytes(self):
        """Test size_in_bytes method."""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        memmap_arr = MemmapArray(arr)
        size = memmap_arr.size_in_bytes()
        assert size > 0
        assert size >= arr.nbytes
        memmap_arr.delete()

    def test_memmap_array_init_deinit(self):
        """Test init and deinit methods."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        memmap_arr = MemmapArray(arr)
        memmap_arr.deinit()
        assert memmap_arr._mmap_arr is None
        
        memmap_arr.init()
        assert memmap_arr._mmap_arr is not None
        np.testing.assert_array_equal(memmap_arr.values, arr)
        memmap_arr.delete()

    def test_memmap_array_to_spark(self):
        """Test converting to Spark format."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        memmap_arr = MemmapArray(arr)
        assert memmap_arr._on_spark is False
        memmap_arr.to_spark()
        assert memmap_arr._on_spark is True
        memmap_arr.delete()

    def test_memmap_array_delete(self):
        """Test delete method."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        memmap_arr = MemmapArray(arr)
        memmap_file = memmap_arr._local_mmap_file
        assert memmap_file.exists()
        memmap_arr.delete()
        assert not memmap_file.exists()

    def test_memmap_array_reduce(self):
        """Test __reduce__ method."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        memmap_arr = MemmapArray(arr)
        memmap_arr.__reduce__()
        assert memmap_arr._mmap_arr is None
        memmap_arr.delete()
