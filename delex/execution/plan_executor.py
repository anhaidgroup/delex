import pyspark
from delex.execution.optimizer import BlockingProgramOptimizer
from delex.execution.cost_estimation import CostEstimator
from delex.execution.graph_executor import GraphExecutor, GraphExecutionStats
from delex.utils.funcs import get_logger
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

    def execute(self, prog, search_table_id_col, projection=None):
        plan, cost_estimation_time, optimize_time = self.generate_plan(prog)

        df, stats = super().execute(plan, search_table_id_col, projection)
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

    
