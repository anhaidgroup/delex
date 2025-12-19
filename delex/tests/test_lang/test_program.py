"""
Tests for lang.program module.

This module tests blocking program functionality.
"""
import pytest
from delex.lang import BlockingProgram, KeepRule, DropRule
from delex.lang.predicate import ExactMatchPredicate


@pytest.mark.unit
class TestBlockingProgram:
    """Tests for BlockingProgram class."""

    def test_blocking_program_init(self):
        """Test BlockingProgram initialization."""
        pred = ExactMatchPredicate(index_col='title', search_col='title', invert=False, lowercase=False)
        keep_rule = KeepRule(predicates=[pred])
        drop_rule = DropRule(predicates=[pred])
        
        program = BlockingProgram(keep_rules=[keep_rule], drop_rules=[drop_rule])
        assert program.keep_rules == [keep_rule]
        assert program.drop_rules == [drop_rule]

    def test_blocking_program_validate_empty_keep_rules(self):
        """Test validation fails with empty keep_rules."""
        with pytest.raises(ValueError, match='blocking program must contain at least one keep rule'):
            BlockingProgram(keep_rules=[], drop_rules=[])

    def test_blocking_program_pretty_str(self):
        """Test pretty_str method."""
        pred = ExactMatchPredicate(index_col='title', search_col='title', invert=False, lowercase=False)
        keep_rule = KeepRule(predicates=[pred])
        
        program = BlockingProgram(keep_rules=[keep_rule], drop_rules=[])
        s = program.pretty_str()
        assert 'KEEP (' in s
        assert 'exact_match(title, title)' in s
        assert 'DROP' not in s

    def test_blocking_program_pretty_str_with_drop_rules(self):
        """Test pretty_str method with drop rules."""
        pred = ExactMatchPredicate(index_col='title', search_col='title', invert=False, lowercase=False)
        keep_rule = KeepRule(predicates=[pred])
        drop_rule = DropRule(predicates=[pred])
        
        program = BlockingProgram(keep_rules=[keep_rule], drop_rules=[drop_rule])
        s = program.pretty_str()
        assert 'KEEP (' in s
        assert 'DROP (' in s
        assert 'exact_match(title, title)' in s
        assert 'OR' not in s # only one rule each
        
        # Test with multiple rules for OR
        program2 = BlockingProgram(keep_rules=[keep_rule, keep_rule], drop_rules=[])
        s2 = program2.pretty_str()
        assert 'OR' in s2
