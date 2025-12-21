"""
Tests for lang.predicate.name_map module.

This module tests name mapping functionality.
"""
import pytest
from delex.lang.predicate.name_map import NAME_MAP


@pytest.mark.unit
class TestNameMap:
    """Tests for NAME_MAP dictionary."""

    def test_name_map_exists(self):
        """Test NAME_MAP exists and is a dict."""
        assert NAME_MAP is not None
        assert isinstance(NAME_MAP, dict)

    def test_name_map_get(self):
        """Test NAME_MAP get method."""
        assert NAME_MAP.get('alison') == 'ali'
