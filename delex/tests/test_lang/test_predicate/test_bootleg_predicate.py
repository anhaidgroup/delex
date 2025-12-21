"""
Tests for lang.predicate.bootleg_predicate module.

This module tests bootleg predicate functionality.
"""
import pytest
from delex.lang.predicate import BootlegPredicate
from delex.lang.predicate.bootleg_predicate import BootlegSim
from delex.utils import BuildCache
import operator

@pytest.mark.unit
class TestBootlegPredicate:
    """Tests for BootlegPredicate class."""

    def test_bootleg_predicate_init(self):
        """Test BootlegPredicate initialization."""
        bootleg_pred = BootlegPredicate(
            index_col='name', search_col='name', invert=True)
        assert bootleg_pred is not None
        assert bootleg_pred.index_col == 'name'
        assert bootleg_pred.search_col == 'name'
        assert bootleg_pred.val == 0.0
        assert bootleg_pred.op == operator.eq
        assert bootleg_pred.sim == BootlegSim(
            index_col='name', search_col='name', invert=True)
        assert bootleg_pred.is_topk is False
        assert bootleg_pred.streamable is True
        assert bootleg_pred.indexable is False

    def test_bootleg_predicate_build_not_for_search_no_cache(self, spark_session):
        """Test build method."""
        bootleg_pred = BootlegPredicate(
            index_col='name', search_col='name', invert=True)
        bootleg_pred.build(for_search=False, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id')
        assert bootleg_pred._index is not None
        assert bootleg_pred.index_size_in_bytes() > 0

    def test_bootleg_predicate_build_for_search_no_cache(self, spark_session):
        """Test build method for search."""
        bootleg_pred = BootlegPredicate(
            index_col='name', search_col='name', invert=True)
        bootleg_pred.build(for_search=True, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id')
        assert bootleg_pred._index is not None
        assert bootleg_pred.index_size_in_bytes() > 0

    def test_bootleg_predicate_build_not_for_search_with_cache(self, spark_session):
        """Test build method for search."""
        bootleg_pred = BootlegPredicate(
            index_col='name', search_col='name', invert=True)
        bootleg_pred.build(for_search=False, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id', cache=BuildCache())
        assert bootleg_pred._index is not None
        assert bootleg_pred.index_size_in_bytes() > 0

    def test_bootleg_predicate_build_for_search_with_cache(self, spark_session):
        """Test build method for search."""
        bootleg_pred = BootlegPredicate(
            index_col='name', search_col='name', invert=True)
        bootleg_pred.build(for_search=True, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id', cache=BuildCache())
        assert bootleg_pred._index is not None
        assert bootleg_pred.index_size_in_bytes() > 0

    def test_bootleg_predicate_compute_scores(self, spark_session):
        """Test compute_scores method."""
        bootleg_pred = BootlegPredicate(
            index_col='name', search_col='name', invert=True)
        bootleg_pred.build(for_search=False, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id')
        scores = bootleg_pred.compute_scores('John Doe', [1, 2, 3])
        assert scores is not None
        assert len(scores) == 3
        assert scores[0] == 1.0
        assert scores[1] == 0.0
        assert scores[2] == 0.0

    def test_bootleg_predicate_search_index(self, spark_session):
        """Test search_index method."""
        bootleg_pred = BootlegPredicate(
            index_col='name', search_col='name', invert=True)
        bootleg_pred.build(for_search=True, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id')
        scores, ids = bootleg_pred.search_index('John Doe')
        assert scores is not None
        assert len(scores) == 1
        assert scores[0] == 1.0
        assert ids[0] == 1
