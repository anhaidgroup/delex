"""
Tests for storage.packed_memmap_arrs module.

This module tests packed memory-mapped arrays functionality.
"""
import pytest
import numpy as np
from delex.storage import PackedMemmapArrays, MemmapArray


@pytest.mark.unit
class TestPackedMemmapArrays:
    """Tests for PackedMemmapArrays class."""

    def test_packed_memmap_arrays_init(self):
        """Test PackedMemmapArrays initialization."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([4, 5, 6, 7], dtype=np.float32)
        arr3 = np.array([8, 9], dtype=np.int64)
        
        packed = PackedMemmapArrays([arr1, arr2, arr3])
        assert len(packed._shapes) == 3
        assert len(packed._dtypes) == 3
        assert len(packed._offsets) == 4
        assert packed._total_bytes > 0
        packed.delete()

    def test_packed_memmap_arrays_unpack(self):
        """Test unpack method."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([4, 5, 6], dtype=np.float32)
        
        packed = PackedMemmapArrays([arr1, arr2])
        packed.init()
        unpacked = packed.unpack()
        
        assert len(unpacked) == 2
        np.testing.assert_array_equal(unpacked[0], arr1)
        np.testing.assert_array_equal(unpacked[1], arr2)
        packed.delete()

    def test_packed_memmap_arrays_size_in_bytes(self):
        """Test size_in_bytes method."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([4, 5, 6], dtype=np.float32)
        
        packed = PackedMemmapArrays([arr1, arr2])
        size = packed.size_in_bytes()
        assert size > 0
        packed.delete()

    def test_packed_memmap_arrays_init_deinit(self):
        """Test init and deinit methods."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        packed = PackedMemmapArrays([arr1])
        
        assert packed._mmap_arr is None
        packed.init()
        assert packed._mmap_arr is not None
        
        packed.deinit()
        assert packed._mmap_arr is None
        packed.delete()

    def test_packed_memmap_arrays_to_spark(self):
        """Test converting to Spark format."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        packed = PackedMemmapArrays([arr1])
        assert packed._on_spark is False
        packed.to_spark()
        assert packed._on_spark is True
        packed.delete()

    def test_packed_memmap_arrays_delete(self):
        """Test delete method."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        packed = PackedMemmapArrays([arr1])
        memmap_file = packed._local_mmap_file
        assert memmap_file.exists()
        packed.delete()
        assert not memmap_file.exists()
