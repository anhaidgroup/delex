"""
Tests for lang.predicate.topk_predicate module.

This module tests top-k predicate functionality.
"""
import pytest
from delex.lang.predicate.topk_predicate import BM25TopkPredicate, CachedBM25IndexKey
from delex.lang.predicate import ExactMatchPredicate
from delex.utils.build_cache import BuildCache
import pandas as pd


@pytest.mark.unit
class TestBM25TopkPredicate:
    """Tests for BM25TopkPredicate class."""

    def test_bm25_topk_predicate_init(self):
        """Test BM25TopkPredicate initialization."""
        pred = BM25TopkPredicate(
            index_col='title', search_col='title', tokenizer='standard', k=10)
        assert pred.index_col == 'title'
        assert pred.search_col == 'title'
        assert pred.k == 10
        assert pred.is_topk is True
        assert pred.invertable is False
        assert pred.streamable is False
        assert pred.indexable is True
        assert str(pred) == 'BM25_topk(standard, title, title, 10) '
        with pytest.raises(TypeError):
            BM25TopkPredicate(1, 'title', 'standard', 10)
        with pytest.raises(TypeError):
            BM25TopkPredicate('title', 2, 'standard', 10)
        with pytest.raises(TypeError):
            BM25TopkPredicate('title', 'title', 'standard', 10.5)

    def test_bm25_topk_predicate_contains(self):
        """Test contains method."""
        pred1 = BM25TopkPredicate('title', 'title', 'standard', 10)
        pred2 = BM25TopkPredicate('title', 'title', 'standard', 5)
        pred3 = BM25TopkPredicate('title', 'title', 'standard', 15)
        pred_other = ExactMatchPredicate('title', 'title', invert=False)
        assert pred1.contains(pred2) is True
        assert pred1.contains(pred3) is False
        assert pred1.contains(pred_other) is False

    def test_bm25_topk_predicate_hash_eq(self):
        """Test __hash__ and __eq__."""
        pred1 = BM25TopkPredicate('title', 'title', 'standard', 10)
        pred2 = BM25TopkPredicate('title', 'title', 'standard', 10)
        pred3 = BM25TopkPredicate('title', 'title', 'standard', 5)
        pred4 = BM25TopkPredicate('title', 'title', 'whitespace', 10)
        assert pred1 == pred2
        assert pred1 == pred3
        assert pred1 != pred4
        assert hash(pred1) == hash(pred2)

    def test_bm25_topk_predicate_get_index_key(self):
        """Test _get_index_key method."""
        pred = BM25TopkPredicate('title', 'title', 'standard', 10)
        key = pred._get_index_key(for_search=True)
        assert isinstance(key, CachedBM25IndexKey)
        assert key.index_col == 'title'
        assert key.tokenizer == 'standard'
        with pytest.raises(ValueError, match='filter not implemented'):
            pred._get_index_key(for_search=False)

    def test_bm25_topk_predicate_index_size(self, spark_session):
        """Test index_size_in_bytes and index_component_sizes."""
        pred = BM25TopkPredicate('title', 'title', 'standard', 10)
        index_table = spark_session.createDataFrame(
            [(1, 'John Doe'), (2, 'Jane Doe')],
            ['id', 'title'])
        pred.build(for_search=True, index_table=index_table, index_id_col='id')
        assert pred.index_size_in_bytes() > 0
        sizes = pred.index_component_sizes(for_search=True)
        key = pred._get_index_key(for_search=True)
        assert sizes[key] > 0

    def test_bm25_topk_predicate_build(self, spark_session):
        """Test build method."""
        pred = BM25TopkPredicate('title', 'title', 'standard', 10)
        index_table = spark_session.createDataFrame(
            [(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')],
            ['id', 'title'])
        with pytest.raises(RuntimeError, match='cannot build.*for filtering'):
            pred.build(for_search=False, index_table=index_table)
        pred.build(for_search=True, index_table=index_table, index_id_col='id')
        assert pred._index is not None
        assert pred._index_dir is not None
        assert pred.index_size_in_bytes() > 0
        cache = BuildCache()
        pred2 = BM25TopkPredicate('title', 'title', 'standard', 10)
        pred2.build(for_search=True, index_table=index_table, index_id_col='id', cache=cache)
        assert pred2._index is not None
        assert pred2._index_dir is not None
        pred3 = BM25TopkPredicate('title', 'title', 'standard', 10)
        pred3.build(for_search=True, index_table=index_table, index_id_col='id', cache=cache)
        assert pred3._index is pred2._index

    def test_bm25_topk_predicate_search_batch(self, spark_session):
        """Test search_batch method."""
        pred = BM25TopkPredicate('title', 'title', 'standard', 10)
        index_table = spark_session.createDataFrame(
            [(1, 'John Doe'), (2, 'Jane Doe'), (3, 'Jim Doe')],
            ['id', 'title'])
        pred.build(for_search=True, index_table=index_table, index_id_col='id')
        pred.init()
        queries = pd.Series(['John Doe', 'Jane'])
        res = pred.search_batch(queries)
        assert isinstance(res, pd.DataFrame)
        assert len(res) == 2
        assert 'scores' in res.columns
        assert 'id1_list' in res.columns
        assert 'time' in res.columns
        assert len(res.iloc[0]['scores']) > 0
        assert len(res.iloc[0]['id1_list']) > 0
        pred.deinit()

    def test_bm25_topk_predicate_filter_batch(self):
        """Test filter_batch raises error."""
        pred = BM25TopkPredicate('title', 'title', 'standard', 10)
        with pytest.raises(RuntimeError, match='topk cannot be used as a filter'):
            pred.filter_batch(None, None)

    def test_bm25_topk_predicate_init_deinit(self, spark_session):
        """Test init and deinit."""
        pred = BM25TopkPredicate('title', 'title', 'standard', 10)
        index_table = spark_session.createDataFrame(
            [(1, 'John Doe'), (2, 'Jane Doe')],
            ['id', 'title'])
        pred.build(for_search=True, index_table=index_table, index_id_col='id')
        pred.init()
        assert pred._index is not None
        pred.deinit()
        assert pred._index is not None
