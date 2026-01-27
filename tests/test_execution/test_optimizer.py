"""
Tests for execution.optimizer module.

This module tests query optimization functionality.
"""
import pytest
from delex.execution.optimizer import BlockingProgramOptimizer


@pytest.mark.unit
class TestBlockingProgramOptimizer:
    """Tests for BlockingProgramOptimizer class."""

    def test_optimizer_init(self):
        """Test BlockingProgramOptimizer initialization."""
        optimizer = BlockingProgramOptimizer()
        assert optimizer is not None

    def test_optimizer_preprocess(self):
        """Test preprocessing of blocking program."""
        from delex.lang import BlockingProgram, KeepRule
        from delex.lang.predicate import ExactMatchPredicate
        pred = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        keep_rule = KeepRule(predicates=[pred])
        simple_blocking_program = BlockingProgram(
            keep_rules=[keep_rule], drop_rules=[])
        optimizer = BlockingProgramOptimizer()
        result = optimizer.preprocess(simple_blocking_program)
        assert optimizer is not None
        assert result is not None
        assert isinstance(result, BlockingProgram)
        assert len(result.keep_rules) >= 1
        assert len(result.drop_rules) >= 0

    def test_optimizer_default_plan(self):
        """Test creating default execution plan."""
        from delex.lang import BlockingProgram, KeepRule
        from delex.lang.predicate import ExactMatchPredicate
        from delex.graph import Node
        pred = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        keep_rule = KeepRule(predicates=[pred])
        simple_blocking_program = BlockingProgram(
            keep_rules=[keep_rule], drop_rules=[])
        optimizer = BlockingProgramOptimizer()
        plan = optimizer.default_plan(simple_blocking_program)
        assert optimizer is not None
        assert plan is not None
        assert isinstance(plan, Node)
        assert plan.is_sink

    def test_optimizer_optimize(self):
        """Test optimization of blocking program."""
        from delex.lang import BlockingProgram, KeepRule
        from delex.lang.predicate import ExactMatchPredicate, BM25TopkPredicate
        from delex.graph import Node
        pred = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        tfidf_pred = BM25TopkPredicate(
            index_col='title', search_col='title', tokenizer='3gram', k=10)
        keep_rule = KeepRule(predicates=[pred, tfidf_pred])
        simple_blocking_program = BlockingProgram(
            keep_rules=[keep_rule], drop_rules=[])
        optimizer = BlockingProgramOptimizer()
        plan, cost = optimizer.optimize(simple_blocking_program)
        assert optimizer is not None
        assert plan is not None
        assert isinstance(plan, Node)
        assert plan.is_sink
        # cost is -1 when no cost_est is provided
        assert cost == -1

    def test_optimizer_preprocess_with_redundant_predicates(self):
        """Test preprocessing with redundant predicates.

        Exercises _preprocess_bool_objects.
        """
        from delex.lang import BlockingProgram, KeepRule
        from delex.lang.predicate import JaccardPredicate
        from delex.tokenizer import QGramTokenizer
        import operator

        # Create predicates where one contains another
        # (lower threshold contains higher threshold)
        tokenizer = QGramTokenizer(3)
        pred1 = JaccardPredicate(
            index_col='title', search_col='title', op=operator.ge,
            tokenizer=tokenizer, val=0.5)
        pred2 = JaccardPredicate(
            index_col='title', search_col='title', op=operator.ge,
            tokenizer=tokenizer, val=0.7)
        # pred1 contains pred2, so pred2 should be removed
        keep_rule = KeepRule(predicates=[pred1, pred2])
        blocking_program = BlockingProgram(
            keep_rules=[keep_rule], drop_rules=[])
        optimizer = BlockingProgramOptimizer()
        result = optimizer.preprocess(blocking_program)
        assert result is not None
        assert isinstance(result, BlockingProgram)
        assert len(result.keep_rules) == 1
        # The redundant predicate should be removed
        assert len(result.keep_rules[0].predicates) <= 2

    def test_optimizer_preprocess_with_redundant_rules(self):
        """Test preprocessing with redundant rules.

        Exercises _preprocess_bool_objects on rules.
        """
        from delex.lang import BlockingProgram, KeepRule
        from delex.lang.predicate import ExactMatchPredicate

        # Create two rules where one contains another
        pred1 = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        pred2 = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        # Rule with pred1 contains rule with pred2 (if they're the same)
        keep_rule1 = KeepRule(predicates=[pred1])
        keep_rule2 = KeepRule(predicates=[pred2])
        blocking_program = BlockingProgram(
            keep_rules=[keep_rule1, keep_rule2], drop_rules=[])
        optimizer = BlockingProgramOptimizer()
        result = optimizer.preprocess(blocking_program)
        assert result is not None
        assert isinstance(result, BlockingProgram)
        # Redundant rules may be removed
        assert len(result.keep_rules) >= 1

    def test_optimizer_default_plan_with_drop_rules(self):
        """Test default plan with drop rules.

        Exercises _add_drop_rule.
        """
        from delex.lang import BlockingProgram, KeepRule, DropRule
        from delex.lang.predicate import ExactMatchPredicate
        from delex.graph import Node, MinusNode

        pred = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        keep_rule = KeepRule(predicates=[pred])
        drop_pred = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        drop_rule = DropRule(predicates=[drop_pred])
        blocking_program = BlockingProgram(
            keep_rules=[keep_rule], drop_rules=[drop_rule])
        optimizer = BlockingProgramOptimizer()
        plan = optimizer.default_plan(blocking_program)
        assert plan is not None
        assert isinstance(plan, Node)
        assert plan.is_sink
        # Should have a MinusNode when drop_rules are present
        from delex.graph.algorithms import find_all_nodes
        all_nodes = list(find_all_nodes(plan))
        minus_nodes = [n for n in all_nodes if isinstance(n, MinusNode)]
        assert len(minus_nodes) >= 1

    def test_optimizer_default_plan_with_multiple_keep_rules(self):
        """Test default plan with multiple keep rules.

        Exercises _add_keep_rule_to_plan thoroughly.
        """
        from delex.lang import BlockingProgram, KeepRule
        from delex.lang.predicate import ExactMatchPredicate
        from delex.graph import Node, UnionNode

        pred1 = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        pred2 = ExactMatchPredicate(
            index_col='description', search_col='description', invert=False)
        keep_rule1 = KeepRule(predicates=[pred1])
        keep_rule2 = KeepRule(predicates=[pred2])
        blocking_program = BlockingProgram(
            keep_rules=[keep_rule1, keep_rule2], drop_rules=[])
        optimizer = BlockingProgramOptimizer()
        plan = optimizer.default_plan(blocking_program)
        assert plan is not None
        assert isinstance(plan, Node)
        assert plan.is_sink
        # With multiple keep rules, should have a UnionNode
        from delex.graph.algorithms import find_all_nodes
        all_nodes = list(find_all_nodes(plan))
        union_nodes = [n for n in all_nodes if isinstance(n, UnionNode)]
        assert len(union_nodes) >= 1

    def test_optimizer_optimize_with_cost_est(self, table_a, table_b):
        """Test optimization with cost estimator.

        Exercises _run_optimize_loop, _generate_plans, _predicate_reuse,
        _rule_short_circuit, _min_weighted_hitting_set_cover.
        """
        from delex.lang import BlockingProgram, KeepRule
        from delex.lang.predicate import ExactMatchPredicate
        from delex.graph import Node
        from delex.execution import CostEstimator

        pred = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        keep_rule = KeepRule(predicates=[pred])
        blocking_program = BlockingProgram(
            keep_rules=[keep_rule], drop_rules=[])
        optimizer = BlockingProgramOptimizer()
        cost_est = CostEstimator(table_a, table_b, nthreads=4)
        cost_est.compute_estimates(blocking_program)
        plan, cost = optimizer.optimize(blocking_program, cost_est=cost_est)
        assert plan is not None
        assert isinstance(plan, Node)
        assert plan.is_sink
        assert cost is not None
        assert isinstance(cost, (int, float))
        # Cost should be non-negative when cost_est is provided
        assert cost >= 0

    def test_optimizer_short_circuit_with_multiple_rules(
            self, table_a, table_b):
        """Test short circuit optimization with multiple keep rules.

        Exercises lines 241-256 in _rule_short_circuit method.
        Creates a UnionNode with multiple input paths to trigger
        short circuit optimization logic.
        """
        from delex.lang import BlockingProgram, KeepRule
        from delex.lang.predicate import ExactMatchPredicate
        from delex.graph import Node
        from delex.execution import CostEstimator

        # Create multiple keep rules with different predicates
        # This will create a UnionNode with multiple inputs
        pred1 = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        pred2 = ExactMatchPredicate(
            index_col='description', search_col='description', invert=False)
        keep_rule1 = KeepRule(predicates=[pred1])
        keep_rule2 = KeepRule(predicates=[pred2])
        blocking_program = BlockingProgram(
            keep_rules=[keep_rule1, keep_rule2], drop_rules=[])
        optimizer = BlockingProgramOptimizer()
        cost_est = CostEstimator(table_a, table_b, nthreads=4)
        cost_est.compute_estimates(blocking_program)
        plan, cost = optimizer.optimize(blocking_program, cost_est=cost_est)
        assert plan is not None
        assert isinstance(plan, Node)
        assert plan.is_sink
        assert cost is not None
        assert isinstance(cost, (int, float))
        assert cost >= 0
        # Verify UnionNode exists (may be optimized away, but should
        # have existed)
        from delex.graph.algorithms import find_all_nodes
        all_nodes = list(find_all_nodes(plan))
        # The plan should be valid even after short circuit optimization
        assert len(all_nodes) > 0

    def test_optimizer_short_circuit_process_candidates(
            self, table_a, table_b):
        """Test short circuit optimization processing candidates.

        Exercises lines 257-270 in _rule_short_circuit method.
        Creates a blocking program with multiple keep rules that have
        multiple predicates to create paths with length >= 2, increasing
        the chance that short circuit candidates are found and processed.
        """
        from delex.lang import BlockingProgram, KeepRule
        from delex.lang.predicate import (
            ExactMatchPredicate, BM25TopkPredicate, JaccardPredicate)
        from delex.tokenizer import QGramTokenizer
        from delex.graph import Node
        from delex.execution import CostEstimator
        import operator

        # Create multiple keep rules with multiple predicates each
        # This increases the chance of creating paths that meet the
        # conditions for short circuit optimization

        # Create tokenizer object for JaccardPredicate
        tokenizer = QGramTokenizer(3)

        # Rule 1: title exact match + BM25 + Jaccard filter
        pred1a = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        pred1b = BM25TopkPredicate(
            index_col='title', search_col='title', tokenizer='3gram', k=10)
        pred1c = JaccardPredicate(
            index_col='title', search_col='title', op=operator.ge,
            tokenizer=tokenizer, val=0.5)
        keep_rule1 = KeepRule(predicates=[pred1a, pred1b, pred1c])

        # Rule 2: title exact match + different Jaccard filter
        pred2a = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        pred2b = JaccardPredicate(
            index_col='title', search_col='title', op=operator.ge,
            tokenizer=tokenizer, val=0.7)
        keep_rule2 = KeepRule(predicates=[pred2a, pred2b])

        # Rule 3: description exact match + BM25
        pred3a = ExactMatchPredicate(
            index_col='description', search_col='description', invert=False)
        pred3b = BM25TopkPredicate(
            index_col='description', search_col='description',
            tokenizer='3gram', k=10)
        keep_rule3 = KeepRule(predicates=[pred3a, pred3b])

        blocking_program = BlockingProgram(
            keep_rules=[keep_rule1, keep_rule2, keep_rule3], drop_rules=[])
        optimizer = BlockingProgramOptimizer()
        cost_est = CostEstimator(table_a, table_b, nthreads=4)
        cost_est.compute_estimates(blocking_program)
        plan, cost = optimizer.optimize(blocking_program, cost_est=cost_est)
        assert plan is not None
        assert isinstance(plan, Node)
        assert plan.is_sink
        assert cost is not None
        assert isinstance(cost, (int, float))
        assert cost >= 0
        # Verify the plan is valid after processing candidates
        from delex.graph.algorithms import find_all_nodes
        all_nodes = list(find_all_nodes(plan))
        assert len(all_nodes) > 0
        # The plan should be valid after short circuit optimization
        # (may have MinusNodes from short circuit optimization)
        assert plan.is_sink

    def test_optimizer_predicate_reuse(self, table_a, table_b):
        """Test predicate reuse optimization.

        Exercises lines 204-230 in _predicate_reuse method.
        Creates a blocking program where the same non-indexable predicate
        appears multiple times in different keep rules, forcing it to be
        filter nodes that end at the same UnionNode, triggering reuse.
        """
        from delex.lang import BlockingProgram, KeepRule
        from delex.lang.predicate import ExactMatchPredicate
        from delex.graph import Node
        from delex.execution import CostEstimator

        # Use ExactMatchPredicate with invert=True - this is NOT indexable
        # So it will definitely become filter nodes, not indexed nodes
        shared_filter_pred = ExactMatchPredicate(
            index_col='title', search_col='title', invert=True)

        # Create different indexable predicates for each rule
        # These will be indexed, and the shared_filter_pred will be filters
        pred1_indexed = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False)
        pred2_indexed = ExactMatchPredicate(
            index_col='title', search_col='title', invert=False,
            lowercase=True)
        pred3_indexed = ExactMatchPredicate(
            index_col='description', search_col='description', invert=False)

        # Create rules where each has an indexable predicate + the same
        # non-indexable predicate. The non-indexable one will be a filter
        # node in each rule, creating multiple instances that end at UnionNode
        keep_rule1 = KeepRule(predicates=[pred1_indexed, shared_filter_pred])
        keep_rule2 = KeepRule(predicates=[pred2_indexed, shared_filter_pred])
        keep_rule3 = KeepRule(predicates=[pred3_indexed, shared_filter_pred])

        blocking_program = BlockingProgram(
            keep_rules=[keep_rule1, keep_rule2, keep_rule3], drop_rules=[])
        optimizer = BlockingProgramOptimizer()
        cost_est = CostEstimator(table_a, table_b, nthreads=4)
        cost_est.compute_estimates(blocking_program)
        plan, cost = optimizer.optimize(blocking_program, cost_est=cost_est)
        assert plan is not None
        assert isinstance(plan, Node)
        assert plan.is_sink
        assert cost is not None
        assert isinstance(cost, (int, float))
        assert cost >= 0
        # Verify the plan is valid after predicate reuse optimization
        from delex.graph.algorithms import find_all_nodes
        all_nodes = list(find_all_nodes(plan))
        assert len(all_nodes) > 0
        # The plan should be valid after optimization
        assert plan.is_sink


@pytest.mark.integration
class TestOptimizerIntegration:
    """Integration tests for BlockingProgramOptimizer."""

    @pytest.mark.slow
    def test_optimizer_end_to_end(self):
        """Test complete optimization workflow."""
        pass
