"""
Tests for execution.cost_estimation module.

This module tests cost estimation functionality including:
- ScalingModel class
- CostEstimator class
- Cost prediction and optimization
"""
import pytest
import numpy as np
from delex.execution.cost_estimation import ScalingModel, CostEstimator


@pytest.mark.unit
class TestScalingModel:
    """Tests for ScalingModel class."""

    def test_scaling_model_init(self):
        """Test ScalingModel initialization."""
        from scipy.optimize import OptimizeResult
        res = OptimizeResult(x=np.array([1.0, 2.0, 3.0, 4.0]))
        model = ScalingModel(res)
        assert model is not None
        assert model._x is not None
        assert len(model._x) == 4

    def test_scaling_model_predict(self):
        """Test ScalingModel prediction method."""
        from scipy.optimize import OptimizeResult
        res = OptimizeResult(x=np.array([1.0, 2.0, 3.0, 4.0]))
        model = ScalingModel(res)
        prediction = model.predict(size=1000)
        A = [np.sqrt(1000), np.log(1000), 1000, 1.0]
        assert prediction is not None
        assert isinstance(prediction, (float, np.ndarray))
        assert np.allclose(prediction, A @ model._x)

    def test_scaling_model_predict_array(self):
        """Test ScalingModel prediction with array input."""
        from scipy.optimize import OptimizeResult
        res = OptimizeResult(x=np.array([1.0, 2.0, 3.0, 4.0]))
        model = ScalingModel(res)
        prediction = model.predict(size=np.array([1000, 2000, 3000]))
        A = np.array([[np.sqrt(1000), np.log(1000), 1000, 1.0],
                      [np.sqrt(2000), np.log(2000), 2000, 1.0],
                      [np.sqrt(3000), np.log(3000), 3000, 1.0]])
        assert prediction is not None
        assert isinstance(prediction, np.ndarray)
        assert np.allclose(prediction, A @ model._x)

    def test_scaling_model_fit(self):
        """Test ScalingModel.fit class method."""
        sizes = np.array([100, 500, 1000, 2000, 5000])
        times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model = ScalingModel.fit(sizes, times)
        assert model is not None
        assert model._x is not None
        assert len(model._x) == 4
        assert isinstance(model, ScalingModel)


@pytest.mark.unit
class TestCostEstimator:
    """Tests for CostEstimator class."""

    @pytest.mark.requires_spark
    def test_cost_estimator_init(self, table_a, table_b):
        """Test CostEstimator initialization."""
        nthreads = 4
        estimator = CostEstimator(table_a, table_b, nthreads)
        assert estimator is not None
        assert estimator._table_a is not None
        assert estimator._table_a_count == 1000
        assert estimator._table_b is not None
        assert estimator._table_b_count == 1000
        assert estimator._nthreads == nthreads

    def test_cost_estimator_properties(self, table_a, table_b):
        nthreads = 4
        estimator = CostEstimator(table_a, table_b, nthreads)
        assert estimator is not None
        assert estimator.table_a_count == 1000
        assert estimator.nthreads == 4
        assert estimator.table_b_count == 1000

    def test_cost_estimator_times_properties(self, table_a, table_b):
        from delex.lang.predicate import ExactMatchPredicate
        nthreads = 4
        estimator = CostEstimator(table_a, table_b, nthreads)
        pred = ExactMatchPredicate(
            index_col='_id', search_col='_id', invert=False)
        estimator._search_time[pred] = 0.001
        estimator._build_time[pred] = 0.001
        estimator._filter_time[pred.sim] = 0.001
        assert estimator.search_time(pred) == 0.001
        assert estimator.build_time(pred) == 0.001
        assert estimator.filter_time(pred) == 0.001

    @pytest.mark.requires_spark
    def test_cost_estimator_estimate(self, table_a, table_b):
        """Test cost estimation for a node."""
        from delex.lang.predicate import ExactMatchPredicate
        from delex.graph import PredicateNode
        pred = ExactMatchPredicate(
            index_col='_id', search_col='_id', invert=False)
        node = PredicateNode(pred)
        from delex.execution.cost_estimation import CostEstimator
        estimator = CostEstimator(table_a, table_b, nthreads=4)

        # Manually set required values for the test
        estimator._selectivity[pred] = 0.1
        estimator._search_time[pred] = 0.001

        cost = estimator.estimate_plan_cost(node)
        assert cost is not None
        assert isinstance(cost, float)
        assert cost > 0.0

    def test_cost_estimator_compute_estimates(self, table_a, table_b):
        """
       Test cost estimator compute estimates method.
        """
        from delex.lang.predicate import ExactMatchPredicate
        from delex.lang import KeepRule, BlockingProgram
        estimator = CostEstimator(table_a, table_b, nthreads=4)
        pred = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        keep_rule = KeepRule(predicates=[pred])
        simple_blocking_program = BlockingProgram(
            keep_rules=[keep_rule], drop_rules=[])
        estimator.compute_estimates(simple_blocking_program)
        assert estimator is not None
        assert estimator._selectivity[pred] is not None
        assert estimator._search_time[pred] is not None
        assert estimator._build_time[pred] is not None
        assert estimator._filter_time[pred.sim] is not None
        index_key = pred._get_index_key(for_search=True)
        assert estimator._index_size[index_key] is not None
        assert estimator._selectivity[pred] >= 0.0
        assert estimator._search_time[pred] >= 0.0
        assert estimator._build_time[pred] >= 0.0
        assert estimator._filter_time[pred.sim] >= 0.0
        assert isinstance(estimator._index_size[index_key], ScalingModel)
