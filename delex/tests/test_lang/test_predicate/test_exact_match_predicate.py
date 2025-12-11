"""
Tests for lang.predicate.exact_match_predicate module.

This module tests exact match predicate functionality.
"""
import pytest
from delex.lang.predicate import ExactMatchPredicate
import operator
from delex.utils import BuildCache

@pytest.mark.unit
class TestExactMatchPredicate:
    """Tests for ExactMatchPredicate class."""

    def test_exact_match_predicate_init(self):
        """Test ExactMatchPredicate initialization."""
        exact_match_pred = ExactMatchPredicate(
            index_col='name', search_col='name', invert=True, lowercase=False)
        assert exact_match_pred is not None
        assert exact_match_pred.index_col == 'name'
        assert exact_match_pred.search_col == 'name'
        assert exact_match_pred.val == 0.0
        assert exact_match_pred.op == operator.eq
        assert exact_match_pred.sim == ExactMatchPredicate.Sim(
            index_col='name', search_col='name', invert=True, lowercase=False)
        assert exact_match_pred.is_topk is False
        assert exact_match_pred.streamable is True
        assert exact_match_pred.indexable is False

    def test_exact_match_predicate_build_not_for_search_no_cache(self, spark_session):
        """Test build method."""
        exact_match_pred = ExactMatchPredicate(
            index_col='name', search_col='name', invert=True, lowercase=False)
        exact_match_pred.build(for_search=False, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id')
        assert exact_match_pred._index is not None
        assert exact_match_pred.index_size_in_bytes() > 0

    def test_exact_match_predicate_build_for_search_no_cache(self, spark_session):
        """Test build method for search."""
        exact_match_pred = ExactMatchPredicate(
            index_col='name', search_col='name', invert=True)
        exact_match_pred.build(for_search=True, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id')
        assert exact_match_pred._index is not None
        assert exact_match_pred.index_size_in_bytes() > 0

    def test_exact_match_predicate_build_not_for_search_with_cache(self, spark_session):
        """Test build method for search."""
        exact_match_pred = ExactMatchPredicate(
            index_col='name', search_col='name', invert=True)
        exact_match_pred.build(for_search=False, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id', cache=BuildCache())
        assert exact_match_pred._index is not None
        assert exact_match_pred.index_size_in_bytes() > 0

    def test_exact_match_predicate_build_for_search_with_cache(self, spark_session):
        """Test build method for search."""
        exact_match_pred = ExactMatchPredicate(
            index_col='name', search_col='name', invert=True)
        exact_match_pred.build(for_search=True, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id', cache=BuildCache())
        assert exact_match_pred._index is not None
        assert exact_match_pred.index_size_in_bytes() > 0

    def test_exact_match_predicate_compute_scores(self, spark_session):
        """Test compute_scores method."""
        exact_match_pred = ExactMatchPredicate(
            index_col='name', search_col='name', invert=True)
        exact_match_pred.build(for_search=False, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id')
        scores = exact_match_pred.compute_scores('John Doe', [1, 2, 3])
        assert scores is not None
        assert len(scores) == 3
        assert scores[0] == 1.0
        assert scores[1] == 0.0
        assert scores[2] == 0.0

    def test_exact_match_predicate_search_index(self, spark_session):
        """Test search_index method."""
        exact_match_pred = ExactMatchPredicate(
            index_col='name', search_col='name', invert=True)
        exact_match_pred.build(for_search=True, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id')
        scores, ids = exact_match_pred.search_index('John Doe')
        assert scores is not None
        assert len(scores) == 1
        assert scores[0] == 1.0
        assert ids[0] == 1
