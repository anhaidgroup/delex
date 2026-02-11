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
        logger.warning(f'[PLAN_EXEC] starting execute (optimize={self.optimize}, estimate_cost={self.estimate_cost})')
        plan, cost_estimation_time, optimize_time = self.generate_plan(prog)
        logger.warning(f'[PLAN_EXEC] plan generated (cost_estimation={cost_estimation_time:.2f}s, optimize={optimize_time:.2f}s), executing...')

        df, stats = super().execute(plan, search_table_id_col, projection)
        stats = PlanExecutionStats(
                optimize_time=optimize_time,
                cost_estimation_time=cost_estimation_time,
                graph_exec_stats=stats,
        )
        logger.warning(f'[PLAN_EXEC] execute completed, total_time={stats.total_time:.2f}s')
        return df, stats

    def generate_plan(self, prog):
        optimizer = BlockingProgramOptimizer()
        
        logger.warning('[PLAN_EXEC] preprocessing blocking program...')
        prog = optimizer.preprocess(prog)

        t = time.perf_counter()
        if self.estimate_cost:
            logger.warning('[PLAN_EXEC] computing cost estimates...')
            # update cost estimator 
            if self.cost_est is None:
                nthreads = pyspark.SparkContext.getOrCreate().defaultParallelism
                self.cost_est = CostEstimator(self.index_table, self.search_table, nthreads)

            self.cost_est.compute_estimates(prog)
        cost_estimation_time = time.perf_counter() - t

        t = time.perf_counter()
        if self.optimize:
            logger.warning('[PLAN_EXEC] optimizing plan...')
            plan, cost = optimizer.optimize(prog, self.cost_est)
        else:
            logger.warning('[PLAN_EXEC] using default plan (optimize=False)')
            plan = optimizer.default_plan(prog)
        optimize_time = time.perf_counter() - t

        return plan, cost_estimation_time, optimize_time

    
