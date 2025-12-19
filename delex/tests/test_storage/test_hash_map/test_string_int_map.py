"""
Tests for storage.hash_map.string_int_map module.

This module tests string to integer mapping functionality.
"""
import pytest
import numpy as np
from delex.storage.hash_map import StringIntHashMap


@pytest.mark.unit
class TestStringIntHashMap:
    """Tests for StringIntHashMap class."""

    def test_string_int_hash_map_build(self):
        """Test StringIntHashMap build."""
        strings = ['hello', 'world', 'test']
        ints = np.array([0, 1, 2], dtype=np.int32)
        
        hash_map = StringIntHashMap.build(strings, ints)
        assert hash_map._memmap_arr is not None
        assert hash_map._hash_func is not None
        hash_map._memmap_arr.delete()

    def test_string_int_hash_map_getitem_single(self):
        """Test __getitem__ with single string key."""
        strings = ['hello', 'world', 'test']
        ints = np.array([10, 20, 30], dtype=np.int32)
        
        hash_map = StringIntHashMap.build(strings, ints)
        hash_map.init()
        
        assert hash_map['hello'] == 10
        assert hash_map['world'] == 20
        assert hash_map['test'] == 30
        assert hash_map['nonexistent'] == -1
        
        hash_map.deinit()
        hash_map._memmap_arr.delete()

    def test_string_int_hash_map_getitem_array(self):
        """Test __getitem__ with array of string keys."""
        strings = ['hello', 'world', 'test']
        ints = np.array([10, 20, 30], dtype=np.int32)
        
        hash_map = StringIntHashMap.build(strings, ints)
        hash_map.init()
        
        keys = ['hello', 'world', 'nonexistent']
        results = hash_map[keys]
        
        assert len(results) == 3
        assert results[0] == 10
        assert results[1] == 20
        assert results[2] == -1  # Non-existent key
        
        hash_map.deinit()
        hash_map._memmap_arr.delete()

    def test_string_int_hash_map_size_in_bytes(self):
        """Test size_in_bytes method."""
        strings = ['hello', 'world']
        ints = np.array([0, 1], dtype=np.int32)
        
        hash_map = StringIntHashMap.build(strings, ints)
        size = hash_map.size_in_bytes()
        assert size > 0
        hash_map._memmap_arr.delete()

    def test_string_int_hash_map_to_spark(self):
        """Test converting to Spark format."""
        strings = ['hello', 'world']
        ints = np.array([0, 1], dtype=np.int32)

        hash_map = StringIntHashMap.build(strings, ints)
        assert hash_map._memmap_arr._on_spark is False
        hash_map.to_spark()
        assert hash_map._memmap_arr._on_spark is True
        hash_map._memmap_arr.delete()

    def test_string_int_hash_map_init_deinit(self):
        """Test init and deinit methods."""
        strings = ['hello']
        ints = np.array([0], dtype=np.int32)
        
        hash_map = StringIntHashMap.build(strings, ints)
        hash_map.deinit()
        assert hash_map._memmap_arr._mmap_arr is None
        
        hash_map.init()
        assert hash_map._memmap_arr._mmap_arr is not None
        hash_map._memmap_arr.delete()
