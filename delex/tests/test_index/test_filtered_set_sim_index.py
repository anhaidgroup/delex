"""
Tests for index.filtered_set_sim_index module.

This module tests filtered set similarity indexing.
"""
import pytest
import numpy as np
from scipy import sparse
from delex.index.filtered_set_sim_index import FilteredSetSimIndex
from delex.tokenizer import StrippedWhiteSpaceTokenizer


@pytest.mark.unit
class TestFilteredSetSimIndex:
    """Tests for FilteredSetSimIndex class."""

    def test_filtered_set_sim_index_init_jaccard(self):
        """Test FilteredSetSimIndex initialization with Jaccard."""
        index = FilteredSetSimIndex('jaccard', 0.5)
        assert index is not None
        assert index._sim == 'jaccard'
        assert index._threshold == 0.5
        assert index.slices is None
        assert index._on_spark is False

    def test_filtered_set_sim_index_init_cosine(self):
        """Test FilteredSetSimIndex initialization with Cosine."""
        index = FilteredSetSimIndex('cosine', 0.7)
        assert index is not None
        assert index._sim == 'cosine'
        assert index._threshold == 0.7
        assert index.slices is None
        assert index._on_spark is False

    def test_filtered_set_sim_index_init_invalid_sim(self):
        """Test that invalid similarity type raises error."""
        with pytest.raises(KeyError):
            FilteredSetSimIndex('invalid', 0.5)

    def test_filtered_set_sim_index_build(self, spark_session):
        """Test building the filtered index from Spark DataFrame."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        df = spark_session.createDataFrame([
            (1, 'apple banana cherry'),
            (2, 'banana cherry date'),
            (3, 'cherry date elderberry'),
        ], ['_id', 'text'])

        tokenizer.build(df, 'text')
        df_tokens = df.select(
            '_id',
            tokenizer.tokenize_set_spark('text').alias('tokens')
        ).filter('tokens is not null')

        index = FilteredSetSimIndex('jaccard', 0.3)
        index.build(df_tokens, 'tokens', '_id')

        assert index.slices is not None
        assert len(index.slices) > 0
        assert index._ids is not None

    def test_filtered_set_sim_index_search(self, spark_session):
        """Test searching the filtered index."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        df = spark_session.createDataFrame([
            (1, 'apple banana cherry'),
            (2, 'banana cherry date'),
            (3, 'cherry date elderberry'),
        ], ['_id', 'text'])

        tokenizer.build(df, 'text')
        df_tokens = df.select(
            '_id',
            tokenizer.tokenize_set_spark('text').alias('tokens')
        ).filter('tokens is not null')

        index = FilteredSetSimIndex('jaccard', 0.3)
        index.build(df_tokens, 'tokens', '_id')

        tokenizer.init()
        query_tokens = tokenizer.tokenize_set('apple banana cherry')
        scores, ids = index.search(query_tokens, 0.3)

        assert len(scores) > 0
        assert len(ids) > 0
        assert len(scores) == len(ids)
        assert 1 in ids
        tokenizer.deinit()

    def test_filtered_set_sim_index_search_threshold(self, spark_session):
        """Test that threshold filtering works correctly."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        df = spark_session.createDataFrame([
            (1, 'apple banana cherry'),
            (2, 'banana cherry date'),
            (3, 'date elderberry fig'),
        ], ['_id', 'text'])

        tokenizer.build(df, 'text')
        df_tokens = df.select(
            '_id',
            tokenizer.tokenize_set_spark('text').alias('tokens')
        ).filter('tokens is not null')

        index = FilteredSetSimIndex('jaccard', 0.3)
        index.build(df_tokens, 'tokens', '_id')

        tokenizer.init()
        query_tokens = tokenizer.tokenize_set('apple banana cherry')
        scores_low, ids_low = index.search(query_tokens, 0.1)
        scores_high, ids_high = index.search(query_tokens, 0.9)

        assert len(scores_low) >= len(scores_high)
        assert all(s >= 0.1 for s in scores_low)
        assert all(s >= 0.9 for s in scores_high)
        tokenizer.deinit()

    def test_filtered_set_sim_index_cosine_similarity(self, spark_session):
        """Test Cosine similarity search."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        df = spark_session.createDataFrame([
            (1, 'apple banana cherry'),
            (2, 'banana cherry date'),
            (3, 'cherry date elderberry'),
        ], ['_id', 'text'])

        tokenizer.build(df, 'text')
        df_tokens = df.select(
            '_id',
            tokenizer.tokenize_set_spark('text').alias('tokens')
        ).filter('tokens is not null')

        index = FilteredSetSimIndex('cosine', 0.3)
        index.build(df_tokens, 'tokens', '_id')

        tokenizer.init()
        query_tokens = tokenizer.tokenize_set('apple banana cherry')
        scores, ids = index.search(query_tokens, 0.3)

        assert len(scores) > 0
        assert len(ids) > 0
        assert all(0.0 <= s <= 1.0 for s in scores)
        tokenizer.deinit()

    def test_filtered_set_sim_index_empty_query(self, spark_session):
        """Test searching with empty token set."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        df = spark_session.createDataFrame([
            (1, 'apple banana cherry'),
            (2, 'banana cherry date'),
        ], ['_id', 'text'])

        tokenizer.build(df, 'text')
        df_tokens = df.select(
            '_id',
            tokenizer.tokenize_set_spark('text').alias('tokens')
        ).filter('tokens is not null')

        index = FilteredSetSimIndex('jaccard', 0.3)
        index.build(df_tokens, 'tokens', '_id')

        empty_query = np.array([], dtype=np.int32)
        scores, ids = index.search(empty_query, 0.3)

        assert len(scores) == 0
        assert len(ids) == 0
        tokenizer.deinit()

    def test_filtered_set_sim_index_init_deinit(self, spark_session):
        """Test init and deinit methods."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        df = spark_session.createDataFrame([
            (1, 'apple banana cherry'),
            (2, 'banana cherry date'),
        ], ['_id', 'text'])

        tokenizer.build(df, 'text')
        df_tokens = df.select(
            '_id',
            tokenizer.tokenize_set_spark('text').alias('tokens')
        ).filter('tokens is not null')

        index = FilteredSetSimIndex('jaccard', 0.3)
        index.build(df_tokens, 'tokens', '_id')
        index.to_spark()

        assert index._on_spark is True
        assert index.slices is None

        index.init()
        assert index.slices is not None
        assert len(index.slices) > 0

        index.deinit()
        assert index.slices is None

    def test_filtered_set_sim_index_size_in_bytes(self, spark_session):
        """Test size_in_bytes method."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        df = spark_session.createDataFrame([
            (1, 'apple banana cherry'),
            (2, 'banana cherry date'),
        ], ['_id', 'text'])

        tokenizer.build(df, 'text')
        df_tokens = df.select(
            '_id',
            tokenizer.tokenize_set_spark('text').alias('tokens')
        ).filter('tokens is not null')

        index = FilteredSetSimIndex('jaccard', 0.3)
        index.build(df_tokens, 'tokens', '_id')
        index.to_spark()

        size = index.size_in_bytes()
        assert size > 0
        assert isinstance(size, int)
