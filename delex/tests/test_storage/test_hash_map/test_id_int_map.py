"""
Tests for storage.hash_map.id_int_map module.

This module tests ID to integer mapping functionality.
"""
import pytest
import numpy as np
from delex.storage.hash_map import IdOffsetHashMap


@pytest.mark.unit
class TestIdOffsetHashMap:
    """Tests for IdOffsetHashMap class."""

    def test_id_offset_hash_map_build(self):
        """Test IdOffsetHashMap build."""
        longs = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
        ints = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        
        hash_map = IdOffsetHashMap.build(longs, ints)
        assert hash_map._memmap_arr is not None
        hash_map._memmap_arr.delete()

    def test_id_offset_hash_map_getitem_single(self):
        """Test __getitem__ with single key."""
        longs = np.array([1, 2, 3], dtype=np.uint64)
        ints = np.array([10, 20, 30], dtype=np.int32)
        
        hash_map = IdOffsetHashMap.build(longs, ints)
        hash_map.init()
        
        assert hash_map[1] == 10
        assert hash_map[2] == 20
        assert hash_map[3] == 30
        assert hash_map[999] == -1  # Non-existent key
        
        hash_map.deinit()
        hash_map._memmap_arr.delete()

    def test_id_offset_hash_map_getitem_array(self):
        """Test __getitem__ with array of keys."""
        longs = np.array([1, 2, 3], dtype=np.uint64)
        ints = np.array([10, 20, 30], dtype=np.int32)
        
        hash_map = IdOffsetHashMap.build(longs, ints)
        hash_map.init()
        
        keys = np.array([1, 2, 999], dtype=np.int64)
        results = hash_map[keys]
        
        assert len(results) == 3
        assert results[0] == 10
        assert results[1] == 20
        assert results[2] == -1  # Non-existent key
        
        hash_map.deinit()
        hash_map._memmap_arr.delete()

    def test_id_offset_hash_map_getitem_different_types(self):
        """Test __getitem__ with different key types."""
        longs = np.array([1, 2], dtype=np.uint64)
        ints = np.array([10, 20], dtype=np.int32)
        
        hash_map = IdOffsetHashMap.build(longs, ints)
        hash_map.init()
        
        # Test different numeric types
        assert hash_map[np.uint64(1)] == 10
        assert hash_map[np.int64(2)] == 20
        assert hash_map[np.uint32(1)] == 10
        assert hash_map[np.int32(2)] == 20
        assert hash_map[1] == 10
        
        hash_map.deinit()
        hash_map._memmap_arr.delete()

    def test_id_offset_hash_map_getitem_invalid_type(self):
        """Test __getitem__ with invalid type."""
        longs = np.array([1], dtype=np.uint64)
        ints = np.array([10], dtype=np.int32)
        
        hash_map = IdOffsetHashMap.build(longs, ints)
        hash_map.init()
        
        with pytest.raises(TypeError):
            hash_map["invalid"]
        
        hash_map.deinit()
        hash_map._memmap_arr.delete()

    def test_id_offset_hash_map_size_in_bytes(self):
        """Test size_in_bytes method."""
        longs = np.array([1, 2, 3], dtype=np.uint64)
        ints = np.array([0, 1, 2], dtype=np.int32)
        
        hash_map = IdOffsetHashMap.build(longs, ints)
        size = hash_map.size_in_bytes()
        assert size > 0
        hash_map._memmap_arr.delete()

    def test_id_offset_hash_map_to_spark(self):
        """Test converting to Spark format."""
        longs = np.array([1, 2], dtype=np.uint64)
        ints = np.array([0, 1], dtype=np.int32)
        
        hash_map = IdOffsetHashMap.build(longs, ints)
        assert hash_map.on_spark is False
        hash_map.to_spark()
        assert hash_map.on_spark is True
        hash_map._memmap_arr.delete()
