"""
Tests for execution.plan_executor module.

This module tests execution plan functionality.
"""
import pytest


@pytest.mark.unit
class TestPlanExecutor:
    """Tests for PlanExecutor class."""

    @pytest.mark.requires_spark
    def test_plan_executor_init(self, simple_plan_executor):
        """Test PlanExecutor initialization."""
        assert simple_plan_executor is not None
        assert simple_plan_executor.index_table is not None
        assert simple_plan_executor.search_table is not None
        assert simple_plan_executor.optimize is False
        assert simple_plan_executor.estimate_cost is False

    @pytest.mark.requires_spark
    def test_plan_executor_generate_plan(
            self, simple_plan_executor, simple_blocking_program):
        """Test generating an execution plan."""
        plan, cost_est_time, opt_time = simple_plan_executor.generate_plan(
            simple_blocking_program)
        assert plan is not None
        assert cost_est_time >= 0.0
        assert opt_time >= 0.0

    @pytest.mark.requires_spark
    def test_plan_executor_generate_plan_with_cost_estimation(
            self, table_a, table_b, simple_blocking_program):
        """Test generating plan with cost estimation enabled."""
        from delex.execution.plan_executor import PlanExecutor
        executor = PlanExecutor(
            index_table=table_a,
            search_table=table_b,
            optimize=False,
            estimate_cost=True
        )
        # cost_est should be None initially, will be created in generate_plan
        assert executor.cost_est is None
        plan, cost_est_time, opt_time = executor.generate_plan(
            simple_blocking_program)
        assert plan is not None
        assert cost_est_time >= 0.0
        assert opt_time >= 0.0
        # cost_est should be created after generate_plan
        assert executor.cost_est is not None

    @pytest.mark.requires_spark
    def test_plan_executor_generate_plan_with_optimization(
            self, table_a, table_b, simple_blocking_program):
        """Test generating plan with optimization enabled."""
        from delex.execution.plan_executor import PlanExecutor
        executor = PlanExecutor(
            index_table=table_a,
            search_table=table_b,
            optimize=True,
            estimate_cost=True
        )
        plan, cost_est_time, opt_time = executor.generate_plan(
            simple_blocking_program)
        assert plan is not None
        assert cost_est_time >= 0.0
        assert opt_time >= 0.0

    @pytest.mark.requires_spark
    def test_plan_executor_execute(
            self, simple_plan_executor, simple_blocking_program):
        """Test execute method"""
        df, stats = simple_plan_executor.execute(simple_blocking_program, search_table_id_col='_id')
        assert df is not None
        assert stats is not None
        assert stats.optimize_time >= 0.0
        assert stats.cost_estimation_time >= 0.0
        assert stats.graph_exec_stats is not None
        assert stats.total_time >= 0.0
        assert stats.total_time == (
            stats.graph_exec_stats.total_time +
            stats.optimize_time +
            stats.cost_estimation_time
        )

    @pytest.mark.requires_spark
    def test_plan_executor_execute_rejects_duplicate_index_ids(
            self, table_a, table_b, simple_blocking_program):
        """execute raises when the index table id column has duplicates."""
        from delex.execution.plan_executor import PlanExecutor
        executor = PlanExecutor(
            index_table=table_a.union(table_a),
            search_table=table_b,
            optimize=False,
            estimate_cost=False
        )
        with pytest.raises(ValueError, match="index_table: id column '_id' must be unique"):
            executor.execute(simple_blocking_program, search_table_id_col='_id')

    @pytest.mark.requires_spark
    def test_plan_executor_execute_rejects_duplicate_search_ids(
            self, table_a, table_b, simple_blocking_program):
        """execute raises when the search table id column has duplicates."""
        from delex.execution.plan_executor import PlanExecutor
        executor = PlanExecutor(
            index_table=table_a,
            search_table=table_b.union(table_b),
            optimize=False,
            estimate_cost=False
        )
        with pytest.raises(ValueError, match="search_table: id column '_id' must be unique"):
            executor.execute(simple_blocking_program, search_table_id_col='_id')

    @pytest.mark.requires_spark
    def test_plan_executor_execute_rejects_null_search_ids(
            self, table_a, table_b, simple_blocking_program):
        """execute raises when the search table id column contains nulls."""
        import pyspark.sql.functions as F
        from delex.execution.plan_executor import PlanExecutor
        with_null = table_b.withColumn(
            '_id', F.when(F.col('_id') == 0, None).otherwise(F.col('_id')))
        executor = PlanExecutor(
            index_table=table_a,
            search_table=with_null,
            optimize=False,
            estimate_cost=False
        )
        with pytest.raises(ValueError, match="search_table: nulls are present"):
            executor.execute(simple_blocking_program, search_table_id_col='_id')

    @pytest.mark.requires_spark
    def test_plan_executor_execute_rejects_missing_search_id_column(
            self, simple_plan_executor, simple_blocking_program):
        """execute raises when search_table_id_col is not in the search table."""
        with pytest.raises(ValueError, match="search_table: missing id column 'no_such_col'"):
            simple_plan_executor.execute(
                simple_blocking_program, search_table_id_col='no_such_col')

    @pytest.mark.requires_spark
    def test_plan_executor_execute_rejects_non_integer_index_ids(
            self, table_a, table_b, simple_blocking_program):
        """execute raises when the index table id column is not an integer type."""
        import pyspark.sql.functions as F
        from delex.execution.plan_executor import PlanExecutor
        executor = PlanExecutor(
            index_table=table_a.withColumn('_id', F.col('_id').cast('string')),
            search_table=table_b,
            optimize=False,
            estimate_cost=False
        )
        with pytest.raises(ValueError, match="index_table: id column '_id' must be an integer type"):
            executor.execute(simple_blocking_program, search_table_id_col='_id')

    @pytest.mark.requires_spark
    def test_plan_executor_execute_validate_ids_disabled(
            self, table_a, table_b, simple_blocking_program):
        """execute skips id validation when validate_ids is False."""
        from delex.execution.plan_executor import PlanExecutor
        duplicated = table_b.union(table_b)
        executor = PlanExecutor(
            index_table=table_a,
            search_table=duplicated,
            optimize=False,
            estimate_cost=False
        )
        df, stats = executor.execute(
            simple_blocking_program, search_table_id_col='_id', validate_ids=False)
        assert df.count() == duplicated.count()
