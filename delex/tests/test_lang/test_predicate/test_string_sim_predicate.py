"""
Tests for lang.predicate.string_sim_predicate module.

This module tests string similarity predicate functionality.
"""
import pytest
from delex.lang.predicate import (
    StringSimPredicate, EditDistancePredicate, JaroPredicate,
    JaroWinklerPredicate, SmithWatermanPredicate)
from delex.utils.build_cache import BuildCache
import operator


@pytest.mark.unit
class TestEditDistancePredicate:
    """Tests for EditDistancePredicate class."""

    def test_edit_distance_predicate_init(self):
        """Test EditDistancePredicate initialization."""
        edit_distance_pred = EditDistancePredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5)
        assert edit_distance_pred is not None
        assert edit_distance_pred.index_col == 'name'
        assert edit_distance_pred.search_col == 'name'
        assert edit_distance_pred.op == operator.ge
        assert edit_distance_pred.val == 0.5
        assert edit_distance_pred.is_topk is False
        assert edit_distance_pred.streamable is True
        assert edit_distance_pred.indexable is False
        assert edit_distance_pred.sim == StringSimPredicate.Sim(
            index_col='name', search_col='name', sim_name=edit_distance_pred._sim_name)
        assert str(edit_distance_pred) == 'edit_distance(name, name) >= 0.5'

    def test_edit_distance_predicate_index_size_in_bytes(self, spark_session):
        """Test EditDistancePredicate index size in bytes."""
        edit_distance_pred = EditDistancePredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5)
        edit_distance_pred.build(for_search=False, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id')
        assert edit_distance_pred.index_size_in_bytes() > 0

    def test_edit_distance_predicate_invert(self):
        """Test EditDistancePredicate invert."""
        edit_distance_pred = EditDistancePredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5)
        inverted_pred = edit_distance_pred.invert()
        assert inverted_pred is not None
        assert inverted_pred.index_col == 'name'
        assert inverted_pred.search_col == 'name'
        assert inverted_pred.op == operator.lt

    def test_edit_distance_predicate_build_not_for_search_no_cache(self, spark_session):
        """Test EditDistancePredicate build not for search no cache."""
        edit_distance_pred = EditDistancePredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5)
        edit_distance_pred.build(for_search=False, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id')
        assert edit_distance_pred._index is not None
        assert edit_distance_pred.index_size_in_bytes() > 0
        
    def test_edit_distance_predicate_build_for_search_no_cache(
            self, spark_session):
        """Test EditDistancePredicate build for search raises error."""
        edit_distance_pred = EditDistancePredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5)
        with pytest.raises(RuntimeError, match='cannot build.*for search'):
            edit_distance_pred.build(
                for_search=True,
                index_table=spark_session.createDataFrame(
                    [(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')],
                    ['id', 'name']),
                index_id_col='id')
        
    def test_edit_distance_predicate_build_not_for_search_with_cache(self, spark_session):
        """Test EditDistancePredicate build not for search with cache."""
        edit_distance_pred = EditDistancePredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5)
        edit_distance_pred.build(for_search=False, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id', cache=BuildCache())
        assert edit_distance_pred._index is not None
        assert edit_distance_pred.index_size_in_bytes() > 0
        
    def test_edit_distance_predicate_build_for_search_with_cache(
            self, spark_session):
        """Test EditDistancePredicate build for search raises error."""
        edit_distance_pred = EditDistancePredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5)
        with pytest.raises(RuntimeError, match='cannot build.*for search'):
            edit_distance_pred.build(
                for_search=True,
                index_table=spark_session.createDataFrame(
                    [(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')],
                    ['id', 'name']),
                index_id_col='id',
                cache=BuildCache())
        
    def test_edit_distance_predicate_compute_scores(self, spark_session):
        """Test EditDistancePredicate compute scores."""
        edit_distance_pred = EditDistancePredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5)
        edit_distance_pred.build(
            for_search=False,
            index_table=spark_session.createDataFrame(
                [(1, 'John Doe'), (2, None), (3, 'Jim Doe')],
                ['id', 'name']),
            index_id_col='id')
        edit_distance_pred.init()
        scores = edit_distance_pred.compute_scores(
            'John Doe', [1, 2, 3])
        assert scores is not None
        assert len(scores) == 3
        assert scores[0] == 1.0  # Exact match
        import numpy as np
        assert np.isnan(scores[1])  # y is None case
        edit_distance_pred.deinit()

    def test_string_sim_predicate_index_component_sizes(self, spark_session):
        """Test StringSimPredicate index_component_sizes."""
        pred = EditDistancePredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5)
        
        # Test with index None
        sizes = pred.index_component_sizes(for_search=False)
        assert len(sizes) == 1
        assert list(sizes.values())[0] is None

        # Build it
        pred.build(
            for_search=False,
            index_table=spark_session.createDataFrame([(1, 'John Doe')], ['id', 'name']),
            index_id_col='id')
        
        sizes = pred.index_component_sizes(for_search=False)
        assert len(sizes) == 1
        assert list(sizes.values())[0] > 0

        # Test error if for_search mismatch
        with pytest.raises(ValueError, match='cannot get component sizes'):
            pred.index_component_sizes(for_search=True)

    def test_string_sim_predicate_get_index_key_error(self):
        """Test _get_index_key raises ValueError for for_search=True."""
        pred = EditDistancePredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5)
        with pytest.raises(ValueError, match='search not implemented'):
            pred._get_index_key(for_search=True)

    def test_string_sim_predicate_build_cache_hit(self, spark_session):
        """Test build with cache hit."""
        pred1 = EditDistancePredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5)
        cache = BuildCache()
        df = spark_session.createDataFrame([(1, 'John Doe')], ['id', 'name'])
        
        pred1.build(for_search=False, index_table=df, index_id_col='id', cache=cache)
        
        pred2 = EditDistancePredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5)
        # This should hit the cache and execute line 106
        pred2.build(for_search=False, index_table=df, index_id_col='id', cache=cache)
        assert pred2._index is pred1._index

    def test_string_sim_predicate_not_implemented(self):
        """Test search_index and search raise NotImplementedError."""
        pred = EditDistancePredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5)
        with pytest.raises(NotImplementedError):
            pred.search_index('query')
        with pytest.raises(NotImplementedError):
            pred.search(iter([]))

@pytest.mark.unit
class TestJaroPredicate:
    """Tests for JaroPredicate class."""

    def test_jaro_predicate_init(self):
        """Test JaroPredicate initialization."""
        jaro_pred = JaroPredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5)
        assert jaro_pred is not None
        assert jaro_pred.index_col == 'name'
        assert jaro_pred.search_col == 'name'
        assert jaro_pred.op == operator.ge
        assert jaro_pred.val == 0.5
        assert jaro_pred._sim_name == 'jaro'

    def test_jaro_predicate_compute_scores(self, spark_session):
        """Test JaroPredicate compute scores."""
        jaro_pred = JaroPredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5)
        jaro_pred.build(
            for_search=False,
            index_table=spark_session.createDataFrame(
                [(1, 'John Doe'), (2, None)],
                ['id', 'name']),
            index_id_col='id')
        jaro_pred.init()
        scores = jaro_pred.compute_scores('John Doe', [1, 2])
        assert scores[0] == 1.0
        import numpy as np
        assert np.isnan(scores[1])
        jaro_pred.deinit()


@pytest.mark.unit
class TestJaroWinklerPredicate:
    """Tests for JaroWinklerPredicate class."""

    def test_jaro_winkler_predicate_init(self):
        """Test JaroWinklerPredicate initialization."""
        jaro_winkler_pred = JaroWinklerPredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5, prefix_weight=0.2)
        assert jaro_winkler_pred is not None
        assert jaro_winkler_pred.index_col == 'name'
        assert jaro_winkler_pred.search_col == 'name'
        assert jaro_winkler_pred.op == operator.ge
        assert jaro_winkler_pred.val == 0.5
        assert jaro_winkler_pred._prefix_weight == 0.2
        assert 'jaro_winkler[0.2]' in jaro_winkler_pred._sim_name

    def test_jaro_winkler_predicate_hash_eq_contains(self):
        """Test JaroWinklerPredicate __hash__, __eq__, and contains."""
        pred1 = JaroWinklerPredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5, prefix_weight=0.1)
        pred2 = JaroWinklerPredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5, prefix_weight=0.1)
        pred3 = JaroWinklerPredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5, prefix_weight=0.2)
        
        assert hash(pred1) == hash(pred2)
        assert pred1 == pred2
        assert pred1 != pred3
        assert pred1.contains(pred2)
        assert not pred1.contains(pred3)

    def test_jaro_winkler_predicate_compute_scores(self, spark_session):
        """Test JaroWinklerPredicate compute scores."""
        jaro_winkler_pred = JaroWinklerPredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5)
        jaro_winkler_pred.build(
            for_search=False,
            index_table=spark_session.createDataFrame(
                [(1, 'John Doe'), (2, None)],
                ['id', 'name']),
            index_id_col='id')
        jaro_winkler_pred.init()
        scores = jaro_winkler_pred.compute_scores('John Doe', [1, 2])
        assert scores[0] == 1.0
        import numpy as np
        assert np.isnan(scores[1])
        jaro_winkler_pred.deinit()


@pytest.mark.unit
class TestSmithWatermanPredicate:
    """Tests for SmithWatermanPredicate class."""

    def test_smith_waterman_predicate_init(self):
        """Test SmithWatermanPredicate initialization."""
        smith_waterman_pred = SmithWatermanPredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5, gap_cost=0.5)
        assert smith_waterman_pred is not None
        assert smith_waterman_pred.index_col == 'name'
        assert smith_waterman_pred.search_col == 'name'
        assert smith_waterman_pred.op == operator.ge
        assert smith_waterman_pred.val == 0.5
        assert smith_waterman_pred._gap_cost == 0.5
        assert 'smith_waterman[0.5]' in smith_waterman_pred._sim_name

    def test_smith_waterman_predicate_hash_eq_contains(self):
        """Test SmithWatermanPredicate __hash__, __eq__, and contains."""
        pred1 = SmithWatermanPredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5, gap_cost=1.0)
        pred2 = SmithWatermanPredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5, gap_cost=1.0)
        pred3 = SmithWatermanPredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5, gap_cost=2.0)
        
        assert hash(pred1) == hash(pred2)
        assert pred1 == pred2
        assert pred1 != pred3
        assert pred1.contains(pred2)
        assert not pred1.contains(pred3)

    def test_smith_waterman_predicate_compute_scores(self, spark_session):
        """Test SmithWatermanPredicate compute scores."""
        smith_waterman_pred = SmithWatermanPredicate(
            index_col='name', search_col='name', op=operator.ge, val=0.5)
        smith_waterman_pred.build(
            for_search=False,
            index_table=spark_session.createDataFrame(
                [(1, 'John Doe'), (2, None)],
                ['id', 'name']),
            index_id_col='id')
        smith_waterman_pred.init()
        scores = smith_waterman_pred.compute_scores('John Doe', [1, 2])
        assert scores[0] == 1.0
        import numpy as np
        assert np.isnan(scores[1])
        smith_waterman_pred.deinit()