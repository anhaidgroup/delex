"""
Tests for storage.span_map module.

This module tests span mapping functionality.
"""
import pytest
import numpy as np
from delex.storage.span_map import (
    create_span_map, span_map_get_key, span_map_insert_key)


@pytest.mark.unit
class TestSpanMap:
    """Tests for span_map functions."""

    def test_create_span_map(self):
        """Test create_span_map function."""
        keys = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        offsets = np.array([0, 10, 20, 30, 40], dtype=np.int32)
        lengths = np.array([10, 10, 10, 10, 10], dtype=np.int16)

        span_map = create_span_map(keys, offsets, lengths)
        assert len(span_map) > 0
        assert span_map.dtype.names == ('hash', 'offset', 'len')

    def test_span_map_get_key(self):
        """Test span_map_get_key function."""
        keys = np.array([1, 2, 3], dtype=np.int32)
        offsets = np.array([0, 10, 20], dtype=np.int32)
        lengths = np.array([10, 10, 10], dtype=np.int16)

        span_map = create_span_map(keys, offsets, lengths)

        offset, length = span_map_get_key(span_map, 1)
        assert offset == 0
        assert length == 10

        offset, length = span_map_get_key(span_map, 2)
        assert offset == 10
        assert length == 10

        offset, length = span_map_get_key(span_map, 999)
        assert offset == -1
        assert length == 0

    def test_span_map_insert_key(self):
        """Test span_map_insert_key function."""
        # Create map with space for more keys (use lower load factor)
        keys = np.array([1, 2], dtype=np.int32)
        offsets = np.array([0, 10], dtype=np.int32)
        lengths = np.array([10, 10], dtype=np.int16)

        # Use lower load factor to ensure space for insertions
        span_map = create_span_map(keys, offsets, lengths, load_factor=0.5)

        # Insert new key
        span_map_insert_key(span_map, 5, 50, 15)
        offset, length = span_map_get_key(span_map, 5)
        assert offset == 50
        assert length == 15

        # Update existing key
        span_map_insert_key(span_map, 1, 100, 20)
        offset, length = span_map_get_key(span_map, 1)
        assert offset == 100
        assert length == 20

    def test_span_map_insert_key_invalid(self):
        """Test span_map_insert_key with invalid key."""
        keys = np.array([1], dtype=np.int32)
        offsets = np.array([0], dtype=np.int32)
        lengths = np.array([10], dtype=np.int16)

        span_map = create_span_map(keys, offsets, lengths)

        with pytest.raises(ValueError, match='keys must be non-negative'):
            span_map_insert_key(span_map, -1, 0, 10)

    def test_create_span_map_length_mismatch(self):
        """Test create_span_map with mismatched lengths."""
        keys = np.array([1, 2], dtype=np.int32)
        offsets = np.array([0, 10, 20], dtype=np.int32)  # Different length
        lengths = np.array([10, 10], dtype=np.int16)

        with pytest.raises(ValueError):
            create_span_map(keys, offsets, lengths)

    def test_span_map_get_key_wrap_around(self):
        """Test span_map_get_key with wrap-around (collision handling)."""
        keys = np.array([1, 2], dtype=np.int32)
        offsets = np.array([0, 10], dtype=np.int32)
        lengths = np.array([10, 10], dtype=np.int16)

        span_map = create_span_map(keys, offsets, lengths, load_factor=0.5)
        # Map size is 4, so key 5 will hash to 1, then wrap around
        span_map_insert_key(span_map, 5, 50, 15)

        offset, length = span_map_get_key(span_map, 5)
        assert offset == 50
        assert length == 15

    def test_span_map_insert_key_wrap_around(self):
        """Test span_map_insert_key with wrap-around."""
        keys = np.array([1], dtype=np.int32)
        offsets = np.array([0], dtype=np.int32)
        lengths = np.array([10], dtype=np.int16)

        span_map = create_span_map(keys, offsets, lengths, load_factor=0.5)
        # Map size is 2, key 3 will hash to 1, collide, wrap to 0
        span_map_insert_key(span_map, 3, 30, 15)
        offset, length = span_map_get_key(span_map, 3)
        assert offset == 30
        assert length == 15

    def test_span_map_get_key_wrap_around_collision(self):
        """Test span_map_get_key with wrap-around collision."""
        keys = np.array([1, 2], dtype=np.int32)
        offsets = np.array([0, 10], dtype=np.int32)
        lengths = np.array([10, 10], dtype=np.int16)

        span_map = create_span_map(keys, offsets, lengths, load_factor=0.5)
        # Map size is 4, insert key that will cause wrap-around
        span_map_insert_key(span_map, 5, 50, 15)

        # Test getting key that wraps around
        offset, length = span_map_get_key(span_map, 5)
        assert offset == 50
        assert length == 15

    def test_create_span_map_different_load_factor(self):
        """Test create_span_map with different load factor."""
        keys = np.array([1, 2, 3], dtype=np.int32)
        offsets = np.array([0, 10, 20], dtype=np.int32)
        lengths = np.array([10, 10, 10], dtype=np.int16)

        span_map = create_span_map(keys, offsets, lengths, load_factor=0.5)
        assert len(span_map) == 6  # int(3 / 0.5) = 6
        assert span_map.dtype.names == ('hash', 'offset', 'len')
