import pyspark
from delex.execution.optimizer import BlockingProgramOptimizer
from delex.execution.cost_estimation import CostEstimator
from delex.execution.graph_executor import GraphExecutor, GraphExecutionStats
from delex.utils.funcs import get_logger, get_num_partitions
from delex.utils.checks import check_id_col
import time 

from pydantic import (
        BaseModel,
        ConfigDict, 
        computed_field
)

logger = get_logger(__name__)

class PlanExecutionStats(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    optimize_time : float
    cost_estimation_time : float
    graph_exec_stats : GraphExecutionStats

    @computed_field
    def total_time(self) -> float:
        return self.graph_exec_stats.total_time + self.optimize_time + self.cost_estimation_time
    

class PlanExecutor(GraphExecutor):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    optimize : bool
    estimate_cost : bool

    def execute(self, prog, search_table_id_col, projection=None, chunk_size=2000, validate_ids=True):
        """
        execute the blocking program `prog` over self.index_table and self.search_table

        Parameters
        ----------
        prog : BlockingProgram
            the blocking program to be executed

        search_table_id_col : str
            the column name for the id of the search table

        projection : Optional[list[str]] = None
            columns to be projected along with the output of `prog`

        chunk_size : int = 2000
            the number of rows per partition used when repartitioning the tables

        validate_ids : bool = True
            check that the id columns of self.index_table and self.search_table
            are integer typed and contain no nulls and no duplicate values

        Raises
        ------
        ValueError
            if self.index_table_id_col is `_rid` or self.index_table has a `_rid` column

        ValueError
            if `validate_ids` is True and either table does not have a valid id column
        """
        self._check_reserved_columns(self.index_table, self.index_table_id_col)
        index_rows = None
        search_rows = None
        if validate_ids:
            index_rows = check_id_col(self.index_table, self.index_table_id_col, 'index_table')
            search_rows = check_id_col(self.search_table, search_table_id_col, 'search_table')

        # before execution, we need to repartition the dataframes
        index_partitions = get_num_partitions(self.index_table, chunk_size=chunk_size, n_rows=index_rows)
        search_partitions = get_num_partitions(self.search_table, chunk_size=chunk_size, n_rows=search_rows)
        self.index_table = self.index_table.repartition(index_partitions, self.index_table_id_col )
        self.search_table = self.search_table.repartition(search_partitions, search_table_id_col)

        plan, cost_estimation_time, optimize_time = self.generate_plan(prog)
        df, stats = super().execute(plan, search_table_id_col, projection, validate_ids=False)
        stats = PlanExecutionStats(
                optimize_time=optimize_time,
                cost_estimation_time=cost_estimation_time,
                graph_exec_stats=stats,
        )
        return df, stats

    def generate_plan(self, prog):
        optimizer = BlockingProgramOptimizer()
        
        prog = optimizer.preprocess(prog)

        t = time.perf_counter()
        if self.estimate_cost:
            # update cost estimator 
            if self.cost_est is None:
                nthreads = pyspark.SparkContext.getOrCreate().defaultParallelism
                self.cost_est = CostEstimator(self.index_table, self.search_table, nthreads)

            self.cost_est.compute_estimates(prog)
        cost_estimation_time = time.perf_counter() - t

        t = time.perf_counter()
        if self.optimize:
            plan, cost = optimizer.optimize(prog, self.cost_est)
        else:
            plan = optimizer.default_plan(prog)
        optimize_time = time.perf_counter() - t

        return plan, cost_estimation_time, optimize_time

    
