"""
Tests for lang.predicate.threshold_predicate module.

This module tests threshold predicate functionality.
"""
import pytest
import operator
import pandas as pd
import numpy as np
from delex.lang.predicate import EditDistancePredicate, JaccardPredicate
from delex.tokenizer import QGramTokenizer


@pytest.mark.unit
class TestThresholdPredicate:
    """Tests for ThresholdPredicate class."""

    def test_threshold_predicate_init(self):
        """Test ThresholdPredicate initialization."""
        pred = EditDistancePredicate('col1', 'col2', operator.ge, 0.5)
        assert pred.index_col == 'col1'
        assert pred.search_col == 'col2'
        assert pred.op == operator.ge
        assert pred.val == 0.5
        assert pred.invertable is True
        with pytest.raises(ValueError, match='operator must be in'):
            EditDistancePredicate('col1', 'col2', operator.add, 0.5)
        with pytest.raises(TypeError):
            EditDistancePredicate(1, 'col2', operator.ge, 0.5)
        with pytest.raises(TypeError):
            EditDistancePredicate('col1', 2, operator.ge, 0.5)
        with pytest.raises(TypeError):
            EditDistancePredicate('col1', 'col2', operator.ge, '0.5')

    def test_threshold_predicate_contains(self):
        """Test contains method."""
        pred1 = EditDistancePredicate('col1', 'col2', operator.ge, 0.5)
        pred2 = EditDistancePredicate('col1', 'col2', operator.ge, 0.6)
        pred3 = EditDistancePredicate('col1', 'col2', operator.le, 0.5)
        assert pred1.contains(pred2) is True
        assert pred2.contains(pred1) is False

        pred_le1 = EditDistancePredicate('col1', 'col2', operator.le, 0.5)
        pred_le2 = EditDistancePredicate('col1', 'col2', operator.le, 0.4)
        assert pred_le1.contains(pred_le2) is True
        assert pred_le2.contains(pred_le1) is False

        assert pred1.contains(pred3) is False
        assert pred1.contains(pred1) is True

    def test_threshold_predicate_hash_eq(self):
        """Test __hash__ and __eq__."""
        pred1 = EditDistancePredicate('col1', 'col2', operator.ge, 0.5)
        pred2 = EditDistancePredicate('col1', 'col2', operator.ge, 0.5)
        pred3 = EditDistancePredicate('col1', 'col2', operator.le, 0.5)

        assert pred1 == pred2
        assert pred1 != pred3
        assert hash(pred1) == hash(pred2)

    def test_threshold_predicate_search_batch(self, spark_session):
        """Test search_batch method."""
        pred = JaccardPredicate(
            'name', 'name', QGramTokenizer(q=3), operator.ge, 0.5)
        pred.build(
            for_search=True,
            index_table=spark_session.createDataFrame(
                [(1, 'John Doe'), (2, 'Jane Doe')],
                ['id', 'name']),
            index_id_col='id')
        pred.init()

        queries = pd.Series(['John Doe', None, 'Jane'])
        res = pred.search_batch(queries)

        assert isinstance(res, pd.DataFrame)
        assert len(res) == 3
        assert 'scores' in res.columns
        assert 'id1_list' in res.columns
        assert 'time' in res.columns
        assert len(res.iloc[1]['scores']) == 0
        assert len(res.iloc[1]['id1_list']) == 0
        assert res.iloc[1]['time'] == 0.0
        assert len(res.iloc[0]['scores']) >= 0
        assert len(res.iloc[2]['scores']) >= 0

        pred.deinit()

    def test_threshold_predicate_filter_batch(self, spark_session):
        """Test filter_batch method."""
        pred = EditDistancePredicate('name', 'name', operator.ge, 0.5)
        pred.build(
            for_search=False,
            index_table=spark_session.createDataFrame(
                [(1, 'John Doe'), (2, 'Jane Smith'), (3, 'Bob Johnson')],
                ['id', 'name']),
            index_id_col='id')
        pred.init()

        queries = pd.Series(['John Doe', None, 'Jane'])
        id1_lists = pd.Series([
            np.array([1, 2, 3], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            np.array([1, 2], dtype=np.int64)
        ])

        res = pred.filter_batch(queries, id1_lists)

        assert isinstance(res, pd.DataFrame)
        assert len(res) == 3
        assert 'scores' in res.columns
        assert 'id1_list' in res.columns
        assert 'time' in res.columns
        assert len(res.iloc[0]['scores']) > 0
        assert len(res.iloc[0]['id1_list']) > 0
        assert len(res.iloc[1]['scores']) == 0
        assert res.iloc[1]['time'] == 0.0
        assert len(res.iloc[2]['scores']) >= 0
        pred.deinit()
