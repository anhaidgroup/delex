"""
Tests for lang.rule module.

This module tests rule functionality.
"""
import pytest
import operator
from delex.lang import KeepRule, DropRule
from delex.lang.predicate import (
    ExactMatchPredicate, EditDistancePredicate, BM25TopkPredicate)


@pytest.mark.unit
class TestRule:
    """Tests for Rule base class."""

    def test_rule_init(self):
        """Test Rule initialization via KeepRule."""
        pred = ExactMatchPredicate('title', 'title', invert=False)
        rule = KeepRule(predicates=[pred])
        assert rule.predicates == [pred]

    def test_rule_iter(self):
        """Test Rule iteration."""
        pred1 = ExactMatchPredicate('title', 'title', invert=False)
        pred2 = ExactMatchPredicate('name', 'name', invert=False)
        rule = KeepRule(predicates=[pred1, pred2])
        assert list(rule) == [pred1, pred2]
        assert [p for p in rule] == [pred1, pred2]

    def test_rule_contains(self):
        """Test Rule contains method."""
        p1 = EditDistancePredicate('name', 'name', operator.ge, 0.5)
        q1 = EditDistancePredicate('name', 'name', operator.ge, 0.6)

        rule_a = DropRule(predicates=[p1])
        rule_b = DropRule(predicates=[q1])

        assert rule_a.contains(rule_b) is True
        assert rule_b.contains(rule_a) is False
        p2 = EditDistancePredicate('title', 'title', operator.ge, 0.3)
        q2 = EditDistancePredicate('title', 'title', operator.ge, 0.4)
        rule_a2 = DropRule(predicates=[p1, p2])
        rule_b2 = DropRule(predicates=[q1, q2])
        assert rule_a2.contains(rule_b2) is True
        rule_b3 = DropRule(predicates=[q1])
        assert rule_a2.contains(rule_b3) is False
        with pytest.raises(TypeError):
            rule_a.contains("not a rule")

    def test_rule_pretty_str(self):
        """Test Rule pretty_str method."""
        pred1 = ExactMatchPredicate('title', 'title', invert=False)
        pred2 = ExactMatchPredicate('name', 'name', invert=False)
        rule = KeepRule(predicates=[pred1, pred2])
        s = rule.pretty_str()
        assert 'exact_match(title, title)' in s
        assert 'AND' in s
        assert 'exact_match(name, name)' in s
        rule2 = KeepRule(predicates=[pred1])
        s2 = rule2.pretty_str()
        assert s2 == 'exact_match(title, title) == True'


@pytest.mark.unit
class TestKeepRule:
    """Tests for KeepRule class."""

    def test_keep_rule_init_valid(self):
        """Test KeepRule initialization with valid predicates."""
        pred_idx = ExactMatchPredicate('title', 'title', invert=False)
        rule = KeepRule(predicates=[pred_idx])
        assert rule.predicates == [pred_idx]
        pred_stream = EditDistancePredicate('name', 'name', operator.ge, 0.5)
        rule2 = KeepRule(predicates=[pred_idx, pred_stream])
        assert len(rule2.predicates) == 2

    def test_keep_rule_validate_no_indexable(self):
        """Test KeepRule validation fails with no indexable."""
        pred_stream = EditDistancePredicate('name', 'name', operator.ge, 0.5)
        with pytest.raises(
                ValueError,
                match='keep rule must contain at least one indexable'):
            KeepRule(predicates=[pred_stream])


@pytest.mark.unit
class TestDropRule:
    """Tests for DropRule class."""

    def test_drop_rule_init_valid(self):
        """Test DropRule initialization with valid predicates."""
        pred = EditDistancePredicate('name', 'name', operator.ge, 0.5)
        rule = DropRule(predicates=[pred])
        assert rule.predicates == [pred]
        pred2 = EditDistancePredicate('title', 'title', operator.ge, 0.5)
        rule2 = DropRule(predicates=[pred, pred2])
        assert len(rule2.predicates) == 2

    def test_drop_rule_validate_empty(self):
        """Test DropRule validation fails with empty predicates."""
        with pytest.raises(
                ValueError,
                match='drop rule must contain at least one predicate'):
            DropRule(predicates=[])

    def test_drop_rule_validate_not_streamable(self):
        """Test DropRule validation fails when predicate not streamable."""
        pred_topk = BM25TopkPredicate(
            'title', 'title', 'standard', 10)
        with pytest.raises(
                ValueError,
                match='all predicates in drop rules must be streamable'):
            DropRule(predicates=[pred_topk])
