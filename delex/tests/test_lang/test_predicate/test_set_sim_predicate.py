"""
Tests for lang.predicate.set_sim_predicate module.

This module tests set similarity predicate functionality.
"""
import pytest
import numpy as np
from delex.lang.predicate import JaccardPredicate, SetSimPredicate, OverlapCoeffPredicate, CosinePredicate
from delex.tokenizer import QGramTokenizer
import operator
from delex.utils import BuildCache

@pytest.mark.unit
class TestJaccardPredicate:
    """Tests for JaccardPredicate class."""

    def test_jaccard_predicate_init(self):
        """Test JaccardPredicate initialization."""
        jaccard_pred = JaccardPredicate(
            index_col='name', search_col='name', tokenizer=QGramTokenizer(q=3), op=operator.ge, val=0.5)
        assert jaccard_pred is not None
        assert jaccard_pred.index_col == 'name'
        assert jaccard_pred.search_col == 'name'
        assert jaccard_pred._tokenizer == QGramTokenizer(q=3)
        assert jaccard_pred.op == operator.ge
        assert jaccard_pred.val == 0.5
        assert jaccard_pred.is_topk is False
        assert jaccard_pred.streamable is True
        assert jaccard_pred.indexable is True
        assert jaccard_pred.sim == SetSimPredicate.Sim(
            index_col='name', search_col='name', sim_name=jaccard_pred._sim_name, tokenizer_name=str(jaccard_pred._tokenizer))

    def test_jaccard_index_size_in_bytes(self, spark_session):
        """Test JaccardPredicate index size in bytes."""
        jaccard_pred = JaccardPredicate(
            index_col='name', search_col='name', tokenizer=QGramTokenizer(q=3), op=operator.ge, val=0.5)
        jaccard_pred.build(for_search=False, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id')
        assert jaccard_pred.index_size_in_bytes() > 0

    def test_jaccard_build_not_for_search_no_cache(self, spark_session):
        """Test JaccardPredicate build not for search."""
        jaccard_pred = JaccardPredicate(
            index_col='name', search_col='name', tokenizer=QGramTokenizer(q=3), op=operator.ge, val=0.5)
        jaccard_pred.build(for_search=False, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id')
        assert jaccard_pred._index is not None
        assert jaccard_pred._index.size_in_bytes() > 0

    def test_jaccard_build_for_search_no_cache(self, spark_session):
        """Test JaccardPredicate build for search."""
        jaccard_pred = JaccardPredicate(
            index_col='name', search_col='name', tokenizer=QGramTokenizer(q=3), op=operator.ge, val=0.5)
        jaccard_pred.build(for_search=True, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id')
        assert jaccard_pred._index is not None
        assert jaccard_pred._index.size_in_bytes() > 0

    def test_jaccard_build_not_for_search_with_cache(self, spark_session):
        """Test JaccardPredicate build not for search with cache."""
        jaccard_pred = JaccardPredicate(
            index_col='name', search_col='name', tokenizer=QGramTokenizer(q=3), op=operator.ge, val=0.5)
        jaccard_pred.build(for_search=False, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id', cache=BuildCache())
        assert jaccard_pred._index is not None
        assert jaccard_pred._index.size_in_bytes() > 0

    def test_jaccard_build_for_search_with_cache(self, spark_session):
        """Test JaccardPredicate build for search with cache."""
        jaccard_pred = JaccardPredicate(
            index_col='name', search_col='name', tokenizer=QGramTokenizer(q=3), op=operator.ge, val=0.5)
        jaccard_pred.build(for_search=True, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id', cache=BuildCache())
        assert jaccard_pred._index is not None
        assert jaccard_pred._index.size_in_bytes() > 0

    def test_jaccard_compute_scores(self, spark_session):
        """Test JaccardPredicate compute scores."""
        jaccard_pred = JaccardPredicate(
            index_col='name', search_col='name', tokenizer=QGramTokenizer(q=3), op=operator.ge, val=0.5)
        jaccard_pred.build(
            for_search=False,
            index_table=spark_session.createDataFrame(
                [(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')],
                ['id', 'name']),
            index_id_col='id')
        jaccard_pred.init()
        scores = jaccard_pred.compute_scores('John Doe', np.array([1, 2, 3]))
        assert scores is not None
        assert len(scores) == 3
        assert scores[0] == 1.0
        jaccard_pred.deinit()

    def test_jaccard_predicate_search_index(self, spark_session):
        """Test JaccardPredicate search index."""
        jaccard_pred = JaccardPredicate(
            index_col='name', search_col='name', tokenizer=QGramTokenizer(q=3), op=operator.ge, val=0.5)
        jaccard_pred.build(for_search=True, index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']), index_id_col='id')
        jaccard_pred.init()
        scores, ids = jaccard_pred.search_index('John Doe')
        assert scores is not None
        assert len(scores) == 1
        jaccard_pred.deinit()

@pytest.mark.unit
class TestOverlapCoeffPredicate:
    """Tests for OverlapCoeffPredicate class."""

    def test_overlap_coeff_predicate_init(self):
        """Test OverlapCoeffPredicate initialization."""
        overlap_coeff_pred = OverlapCoeffPredicate(
            index_col='name', search_col='name', tokenizer=QGramTokenizer(q=3), op=operator.ge, val=0.5)
        assert overlap_coeff_pred is not None
        assert overlap_coeff_pred.index_col == 'name'
        assert overlap_coeff_pred.search_col == 'name'
        assert overlap_coeff_pred._tokenizer == QGramTokenizer(q=3)
        assert overlap_coeff_pred.op == operator.ge

    def test_overlap_coeff_predicate_indexable(self):
        """Test OverlapCoeffPredicate indexable."""
        overlap_coeff_pred = OverlapCoeffPredicate(
            index_col='name', search_col='name', tokenizer=QGramTokenizer(q=3), op=operator.ge, val=0.5)
        assert overlap_coeff_pred.indexable is False

    def test_overlap_coeff_predicate_compute_scores(self, spark_session):
        """Test OverlapCoeffPredicate compute scores."""
        overlap_coeff_pred = OverlapCoeffPredicate(
            index_col='name', search_col='name', tokenizer=QGramTokenizer(q=3), op=operator.ge, val=0.5)
        overlap_coeff_pred.build(
            for_search=False,
            index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']),
            index_id_col='id')
        overlap_coeff_pred.init()
        scores = overlap_coeff_pred.compute_scores('John Doe', np.array([1, 2, 3]))
        assert scores is not None


@pytest.mark.unit
class TestCosinePredicate:
    """Tests for CosinePredicate class."""

    def test_cosine_predicate_init(self):
        """Test CosinePredicate initialization."""
        cosine_pred = CosinePredicate(
            index_col='name', search_col='name', tokenizer=QGramTokenizer(q=3), op=operator.ge, val=0.5)
        assert cosine_pred is not None
        assert cosine_pred.index_col == 'name'
        assert cosine_pred.search_col == 'name'
        assert cosine_pred._tokenizer == QGramTokenizer(q=3)
        assert cosine_pred.op == operator.ge

    def test_cosine_predicate_compute_scores(self, spark_session):
        """Test CosinePredicate compute scores."""
        cosine_pred = CosinePredicate(
            index_col='name', search_col='name', tokenizer=QGramTokenizer(q=3), op=operator.ge, val=0.5)
        cosine_pred.build(
            for_search=False,
            index_table=spark_session.createDataFrame([(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')], ['id', 'name']),
            index_id_col='id')
        cosine_pred.init()
        scores = cosine_pred.compute_scores('John Doe', np.array([1, 2, 3]))
        assert scores is not None