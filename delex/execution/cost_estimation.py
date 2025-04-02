from collections import defaultdict
import pyspark
from tqdm import tqdm
import time
from delex.utils import BuildCache, CachedObjectKey
from delex.utils.funcs import persisted, get_logger
import pyspark.sql.functions as F
import pyspark.sql.types as T
import numpy as np
from typing import Iterator, Iterable, Optional
import pandas as pd
from copy import deepcopy
from scipy.optimize import lsq_linear
from delex.graph import Node, UnionNode, MinusNode, IntersectNode, PredicateNode
from delex.graph.algorithms import topological_sort
from delex.execution.dataframe_stream import DataFrameStream
import math
import warnings
from delex import lang

logger = get_logger(__name__)

class ScalingModel:
    """
    a linear model for scaling behavior
    """

    def __init__(self, res):
        self._res = res
        self._x = res.x
    
    def predict(self, size):
        A = self._create_A(size)
        return A @ self._x

    @staticmethod
    def _create_A(size):
        elems = [
                #size**2,
                np.sqrt(size),
                np.log(size),
                size,
                np.ones_like(size)
            ]
        if isinstance(size, np.ndarray):
            return np.stack(elems, axis=1)
        else:
            return np.array(elems)

    @classmethod
    def fit(cls, size : np.ndarray, time: np.ndarray):
        A = cls._create_A(size)
        #atol = max(A.shape) * np.linalg.norm(A, 1) * np.spacing(1.)
        #res = nnls(A, time, maxiter=50*A.shape[0], atol=atol)
        res = lsq_linear(A, time, bounds=(0.0, np.inf))
        if not res.success:
            warnings.warn(f'NNLS terminated with code {res.status} {res.message}')

        return cls(res)

class CostEstimator:
    """
    class for estimating the runtime, working set size, and selectivity
    """

    def __init__(self, table_a: pyspark.sql.DataFrame, table_b: pyspark.sql.DataFrame, nthreads: int):
        self._selectivity = {}
        self._search_time = {}
        self._build_time = {}
        self._filter_time = {}
        self._set_op_time = {}
        self._index_size = {}

        self._table_a = table_a
        self._table_a_count = table_a.count()
        self._table_b = table_b
        self._table_b_count = table_b.count()
        self._index_sample_sizes = (25000, 50000, 100000)
        self._filter_sample_size = 10000
        self._sample_seed = None
        self._nthreads = nthreads
    
    @property
    def table_a_count(self) -> int:
        return self._table_a_count
    
    @property
    def nthreads(self) -> int:
        return self._nthreads

    @property
    def table_b_count(self) -> int:
        return self._table_b_count

    def search_time(self, obj : Node | lang.Predicate) -> float:
        pred = self._ensure_predicate(obj)
        if pred.indexable:
            return self._search_time[pred] 
        else:
            raise ValueError(f'cannot get search time for non-indexable predicate {pred}')

    def build_time(self, obj : Node | lang.Predicate) -> float:
        pred = self._ensure_predicate(obj)
        if pred.indexable:
            return self._build_time[pred]
        else:
            raise ValueError(f'cannot get build time for non-indexable predicate {pred}')

    def filter_time(self, obj : Node | lang.Predicate) -> float:
        if isinstance(obj, (UnionNode, MinusNode, IntersectNode)):
            return self._set_op_time[type(obj)]

        pred = self._ensure_predicate(obj)
        return self._filter_time[pred.sim] 

    def selectivity(self, obj : Node | lang.Predicate) -> float:
        pred = self._ensure_predicate(obj)
        return self._selectivity[pred]

    def working_set_size(self, obj : Node | lang.Predicate, for_search: bool=None, size: int=None) -> float:
        size = self._table_a_count if size is None else size
        
        if isinstance(obj, (UnionNode, MinusNode, IntersectNode)):
            return 0

        elif isinstance(obj, (PredicateNode, lang.Predicate)):

            if isinstance(obj, PredicateNode):
                for_search = obj.for_search
                pred = obj.predicate
            elif for_search is None:
               raise ValueError('working set size depends on if the predicate is index or filter, must specify `for_search` parameter')
            else:
                pred = obj
            total = 0
            for k in pred.index_component_sizes(for_search).keys():
                total += self._index_size[k].predict(size)
            return total
        elif isinstance(obj, CachedObjectKey):
           return self._index_size[obj].predict(size) 
        else:
            raise TypeError(f'expected Node, Predicate or CachedObjectKey, got {type(obj)}')


    def _ensure_predicate(self, obj : PredicateNode | lang.Predicate) -> lang.Predicate:
        if isinstance(obj, PredicateNode):
            pred = obj.predicate
        elif isinstance(obj, lang.Predicate):
            pred = obj
        else:
            raise TypeError(f'{obj=} {type(obj)=}')
        return pred

    
    def _validate_filter_pred(self, p : lang.Predicate) -> bool:
        try:
            self.filter_time(p)
            self.selectivity(p)
            self.working_set_size(p, False)
        except KeyError:
            return False
        else:
            return True

    def _validate_index_pred(self, p : lang.Predicate) -> bool:
        try:
            self.search_time(p)
            self.build_time(p)
            self.selectivity(p)
            self.working_set_size(p, True)
        except KeyError:
            return False
        else:
            return True

    def validate(self, blocking_program : lang.BlockingProgram) -> None:
        for s in self._get_filter_preds(blocking_program).values():
            for p in s:
                if not self._validate_filter_pred(p):
                    raise ValueError('cost estimator missing parameters')


        for p in self._get_index_preds(blocking_program):
            if not self._validate_index_pred(p):
                raise ValueError('cost estimator missing parameters')

    
    def _estimate_set_op(self, op, pool_size: int=1000, sample_size: int=100, reps: int=1000) -> np.ndarray:
        times = []
        ids = np.arange(pool_size, dtype=np.int64)
        for i in range(reps):
            x = np.random.choice(ids, sample_size, replace=False)
            y = np.random.choice(ids, sample_size, replace=False)
            start_t = time.perf_counter()
            op(x,y)
            t = time.perf_counter() - start_t 
            times.append(t)

        return np.array(times)

    def _estimate_set_op_params(self):
        ops = [
                (UnionNode, np.union1d),
                (IntersectNode, np.intersect1d),
                (MinusNode, np.setdiff1d),
        ]
        logger.info('estimating set operations')
        for node, op in ops:
            self._set_op_time[node] = self._estimate_set_op(op).mean()



    def compute_estimates(self, blocking_program : lang.BlockingProgram):
        """
        compute the cost estimates for `blocking_program` 
        """
        # we estimate cost 
        blocking_program = deepcopy(blocking_program)
        self._estimate_set_op_params()
        self._estimate_filter_params(blocking_program)
        self._estimate_index_params(blocking_program)


    def _down_sample(self, table: pyspark.sql.DataFrame, frac: float) -> pyspark.sql.DataFrame:
        if frac < 1:
            nparts = max(math.ceil(table.rdd.getNumPartitions() * frac), self._nthreads * 2)
            #nparts = math.ceil(table.rdd.getNumPartitions() * frac)
            table = table.sample(withReplacement=False, fraction=frac, seed=self._sample_seed)\
                        .coalesce(nparts)

        return table

    def _compute_average_sim_times(self, data : pyspark.sql.DataFrame) -> pd.Series:
        exprs = []
        for pred_id in data.columns:
            p = data[pred_id]
            # compute average time per pair for each feature
            e = (p.getField('time') / F.size(p.getField('scores'))).alias(pred_id)
            exprs.append(e)
        filter_times = data.select(*exprs)\
                            .toPandas()\
                            .median()

        return filter_times

    def _compute_filter_selectivity(self, data: pyspark.sql.DataFrame, col_to_preds: dict[str, lang.Predicate]) -> dict:
        
        exprs = []
        pred_to_agg_col = {}
        for col, preds in col_to_preds.items():
            col = data[col]
            for p in preds:
                agg_col = f'col_{len(pred_to_agg_col)}'
                pred_to_agg_col[p] = agg_col
                # apply the predicate comparison to the scores 
                # and get the average number of pairs
                scores = col.getField('scores')
                sel = F.size( F.filter(scores, lambda x : p.op(x, p.val)) ) / F.size(scores)
                expr = F.mean( F.when(scores.isNotNull(), sel).otherwise( F.lit(0.0) ) )\
                            .alias(agg_col)

                exprs.append(expr)

        res = data.agg(*exprs)\
                    .toPandas()\
                    .iloc[0]

        return {p : res[c] for p,c in pred_to_agg_col.items()}
    
    def _estimate_index_params_for_size(self,
                index_preds: Iterable[lang.Predicate],
                table_b: pyspark.sql.DataFrame,
                sample_size: int):

        logger.info(f'estimating index params for size : {sample_size}')

        index_preds = {p : f'pred_{i}' for i, p in enumerate(deepcopy(index_preds))}
        build_times = {}
        index_sizes = {}
        search_times = {}
        selectivity = {}

        table_a = self._down_sample(self._table_a, sample_size / self._table_a_count)
        with persisted(table_a) as table_a:
            n_indexed_recs = table_a.count()
            for p in index_preds:
                start_t = time.perf_counter()
                p.build(True, table_a)
                t = time.perf_counter() - start_t
                logger.debug(f'{p} : build_time = {t}')
                build_times[p] = t
                index_sizes[p] = list(p.index_component_sizes(True).items())

            data = _execute_preds(table_b, list(index_preds.items()))

            exprs = []
            for pred, col_name in index_preds.items():
                exprs.append( data[col_name].getField('time').alias(col_name + '_time') )
                exprs.append( F.size(data[col_name].getField('ids')).alias(col_name + '_size') )

            res = data.select(*exprs)\
                        .toPandas()

            for pred, col_name in index_preds.items():
                search_time = res[col_name + '_time'].median()
                logger.debug(f'{pred} : search_time={search_time}')
                search_times[pred] = search_time
                selectivity[pred] = res[col_name + '_size'].mean() / n_indexed_recs
        
        df = pd.DataFrame({
                'build' : build_times, 
                'search' : search_times, 
                'selectivity' : selectivity,
                'size' : index_sizes,
        })
        df['sample_size'] = n_indexed_recs
        return df
    
    def _fit_model_and_predict(self, size: np.ndarray, time: np.ndarray) -> ScalingModel:
        model = ScalingModel.fit(size, time)
        return model.predict(self._table_a_count)
        
    def _estimate_index_params(self, blocking_program: lang.BlockingProgram):
        index_preds = self._get_index_preds(blocking_program)
        index_preds = [p for p in index_preds if not self._validate_index_pred(p)]

        if len(index_preds) == 0:
            return 

        table_b = self._down_sample(self._table_b, self._filter_sample_size / self._table_b_count)
        with persisted(table_b) as table_b:
            table_b.count()
            # rescale sample sizes if table_a_count < the largest sample size
            scale = self._table_a_count  / max(self._table_a_count, max(self._index_sample_sizes))

            data = []
            # warm up cluster, results not used for estimates
            self._estimate_index_params_for_size(index_preds, table_b, 10000 * scale)
            for ss in self._index_sample_sizes:
                data.append(
                        self._estimate_index_params_for_size(index_preds, table_b, ss * scale)
                    )

            df = pd.concat(data)
            selectivity = df.loc[df['sample_size'].eq(df['sample_size'].max())]['selectivity'].to_dict()
            self._selectivity.update(selectivity)
            # fit a model to the sample points and predict the 
            # search time based on self._table_a_count

            search_time = {}
            build_time = {}
            index_size = {}
            logger.info('fitting models for search and build time')
            for p in tqdm(index_preds):
                slc = df.loc[p]
                search_time[p] = self._fit_model_and_predict(
                                        slc['sample_size'].to_numpy(),
                                        slc['search'].to_numpy()
                                    )
                build_time[p] = self._fit_model_and_predict(
                                        slc['sample_size'].to_numpy(),
                                        slc['build'].to_numpy()
                                    )


            size_info = df.explode('size')\
                            [['size', 'sample_size']]

            size_info.index = size_info['size'].apply(lambda x : x[0])
            size_info['size_in_bytes'] = size_info['size'].apply(lambda x : x[1])
            for k in size_info.index.unique():
                slc = size_info.loc[k]
                index_size[k] = ScalingModel.fit(
                                        slc['sample_size'].to_numpy(),
                                        slc['size_in_bytes'].to_numpy()
                                    )

            self._search_time.update(search_time)
            self._build_time.update(build_time)
            self._index_size.update(index_size)
                                
    
    def _get_filter_preds(self, blocking_program: lang.BlockingProgram):
        filter_preds = defaultdict(set)
        for r in blocking_program.keep_rules + blocking_program.drop_rules:
            for p in r:
                if p.streamable:
                    filter_preds[p.sim].add(p)

        filter_preds = {k : list(v) for k,v in filter_preds.items()}
        return filter_preds

    def _get_index_preds(self, blocking_program: lang.BlockingProgram):
        index_preds = {p for r in blocking_program.keep_rules for p in r if p.indexable}
        index_preds = sorted(index_preds, key=str, reverse=False)
        return index_preds
    
    def _estimate_filter_pred_working_set_size(self, filter_preds: dict):
        scale = self._table_a_count  / max(self._table_a_count, max(self._index_sample_sizes))

        data = []
        for ss in self._index_sample_sizes:
            ss = ss * scale
            table_a = self._down_sample(self._table_a, ss / self._table_a_count)
            with persisted(table_a) as table_a:
                sample_size = table_a.count()
                for preds in filter_preds.values():
                    p = deepcopy(preds[0])
                    p.build(False, table_a)
                    for c, sz in p.index_component_sizes(False).items():
                        data.append((p.sim, c, sz, sample_size))

        df = pd.DataFrame.from_records(data, columns=['sim', 'key', 'nbytes', 'sample_size'])\
                .set_index('key')
        
        index_size = {}
        for key in df.index.unique():
            slc = df.loc[key]
            index_size[key] = ScalingModel.fit(
                                    slc['sample_size'].to_numpy(),
                                    slc['nbytes'].to_numpy()
                                )

        self._index_size.update(index_size)



    def _estimate_filter_params(self, blocking_program: lang.BlockingProgram):
        all_filter_preds = self._get_filter_preds(blocking_program)
        filter_preds = {}

        for sim, preds in all_filter_preds.items():
            preds = [p for p in preds if not self._validate_filter_pred(p)]
            if len(preds):
                filter_preds[sim] = preds
        


        logger.info(f'estimating filter params for {len(filter_preds)} predicates')
        if len(filter_preds) == 0:
            return 
        cache = BuildCache()

        self._estimate_filter_pred_working_set_size(filter_preds)
        
        pred_id_to_type = {f'pred_{i}' : t for i,t in enumerate(filter_preds)}

        table_a = self._down_sample(self._table_a, self._filter_sample_size / self._table_a_count)
        table_b = self._down_sample(self._table_b, self._filter_sample_size / self._table_b_count)

        with persisted(table_a) as table_a, persisted(table_b) as table_b:

            for pred_type, preds in filter_preds.items():
                p = preds[0]
                p.build(False, table_a, cache=cache)
        
            a_ids = table_a.select('_id').toPandas()['_id'].to_numpy()
            a_id_sample_size = 10

            @F.pandas_udf(T.ArrayType(T.LongType()))
            def a_id_sample(itr : Iterator[pd.Series]) -> Iterator[pd.Series]:
                for s in itr:
                    yield s.apply(lambda x : np.random.choice(a_ids, a_id_sample_size, replace=False))
            
            table_b = table_b.withColumn('id_list', a_id_sample('_id'))

            pred_pairs = [(filter_preds[pred_type][0], pred_id) for pred_id, pred_type in pred_id_to_type.items()]

            data = _execute_preds(table_b, pred_pairs, 'id_list')
            data = data.select(*[t[1] for t in pred_pairs])

            with persisted(data) as data:
                data.count()
                logger.info('computing filter times')
                filter_times = self._compute_average_sim_times(data)
                self._filter_time.update({pred_id_to_type[pred_id] : time for pred_id, time in filter_times.items()})
                # compute selectivity of filter predicates,
                # we will overwrite these if they are indexed since this gives a better 
                # estimate than taking a random sample (which is what we are doing here)
                logger.info('computing filter selectivity')
                pred_id_to_preds = { pred_id : filter_preds[t] for pred_id, t in pred_id_to_type.items() }
                self._selectivity.update(self._compute_filter_selectivity(data, pred_id_to_preds))
               
    
    def _compute_node_selectivity(self, node: Node, sel_map: dict[Node, float]):
        # assume that all inputs are independent
        if isinstance(node, UnionNode):
            inputs = np.array([sel_map[n] for n in node.iter_in()])
            # P(x OR y OR z) = NOT (NOT x AND NOT y NOT AND NOT z)
            sel = 1 - np.prod( 1 - inputs)

        elif isinstance(node, IntersectNode):
            inputs = np.array([sel_map[n] for n in node.iter_in()])
            # P(x AND y AND z) = P(x) * P(y) * P(z)
            sel = np.prod(inputs)

        elif isinstance(node, MinusNode):
            left = sel_map[node.left]
            right = sel_map[node.right]
            # P(x AND NOT y) = x * (1 - y)
            # TODO factor in if the nodes are from the same 
            # set, meaning that you would just get left - right
            sel = left * (1 - right)

        elif isinstance(node, PredicateNode):
            # source node
            if node.is_source:
                sel = self.selectivity(node)
            elif node.in_degree == 1:
                sel = sel_map[next(node.iter_in())] * self.selectivity(node)
            else:
                raise ValueError('unable to compute selectivity of PredicateNode with multiple inputs, probably invalid plan')
        else:
            raise TypeError('unknown node type to compute selectivity {type(node)=}')

        return sel

    def estimate_plan_cost(self, node: Node) -> float:
        if not node.is_sink:
            raise ValueError('node must be sink')

        srted_nodes = topological_sort(node)
        sel_map = {}
        for n in srted_nodes:
            sel_map[n] = self._compute_node_selectivity(n, sel_map)

        cross_prod_size = self._table_b_count * self._table_a_count
        cost = 0.0
        for n in srted_nodes:
            if n.is_source:
                c = self.search_time(n) * self._table_b_count / self._nthreads
            else:
                # the selectivity of the inputs * the op cost
                i = sum(sel_map[n] for n in node.iter_in())
                c = i * self.filter_time(n) * cross_prod_size / self._nthreads

            cost += c

        return cost

_OUTPUT_TYPE = T.StructType([
        T.StructField('scores', T.ArrayType(T.FloatType())),
        T.StructField('ids', T.ArrayType(T.LongType())),
        T.StructField('time', T.FloatType()),
])

def _flat_to_nested_exprs(schema: T.StructType, prefix: str=''):
    select = []
    for f in schema.fields:
        if isinstance(f.dataType, T.StructType):
            expr = F.struct(_flat_to_nested_exprs(f.dataType, f'{prefix}{f.name}.'))\
                    .alias(f.name)
        else:
            expr = F.col(f'`{prefix}{f.name}`').alias(f.name)
        select.append(expr)

    return select

def _create_search_stream(itr, schema: T.StructType, pred_pairs: list[tuple[lang.Predicate, str]]):
    """
    create a dataframe stream that performs search for the predicates
    """
    stream = DataFrameStream.from_arrow_iter(itr, schema)
    for pred, output_col in pred_pairs:
        input_cols = [pred.search_col]
        stream = stream.apply(pred.search_batch, input_cols, output_col, _OUTPUT_TYPE)

    return stream

def _create_filter_stream(itr, schema: T.StructType, pred_pairs: list[tuple[lang.Predicate, str]], input_col: str):
    """
    create a dataframe stream that filters `input_col` 
    """
    stream = DataFrameStream.from_arrow_iter(itr, schema)
    for pred, output_col in pred_pairs:
        input_cols = [pred.search_col, input_col]
        f = _compute_scores_lambda(pred)
        stream = stream.apply(f, input_cols, output_col, _OUTPUT_TYPE)

    return stream

def _run_stream(itr, schema: T.StructType, pred_pairs: list[tuple[lang.Predicate, str]], input_col: Optional[str]=None):
    for pred, _ in pred_pairs:
        pred.init()

    if input_col is None:
        stream = _create_search_stream(itr, schema, pred_pairs)
    else:
        stream = _create_filter_stream(itr, schema, pred_pairs, input_col)

    yield from stream.to_arrow_stream()

    for pred, _ in pred_pairs:
        pred.deinit()

def _execute_preds(df: pyspark.sql.DataFrame, pred_pairs: list[tuple[lang.Predicate, str]], input_col: str=None):
    """
    execute the predicates, if `input_col` is provided, the predicates will be used for filtering
    else they are used for search.
    """
    in_schema = T.StructType([deepcopy(f) for f in df.schema])
    if input_col is None:
        stream = _create_search_stream([], in_schema, pred_pairs)
    else:
        stream = _create_filter_stream([], in_schema, pred_pairs, input_col)

    flat_schema = stream.spark_schema(flat=True)
    search_table = df.mapInArrow(lambda x : _run_stream(x, in_schema, pred_pairs, input_col), schema=flat_schema)
    search_table = search_table.select(_flat_to_nested_exprs(stream.spark_schema()))
    return search_table

def _compute_scores_lambda(predicate: lang.Predicate):
    """
    create a function which computes scores for predicate
    """

    def f(queries, id1_lists) -> Iterator[pd.DataFrame]:
        res = []
        for query, id_list in zip(queries, id1_lists):
            if query is None or id_list is None:
                res.append( (None, None, None) )
            else:
                start_t = time.perf_counter()
                scores = predicate.compute_scores(query, id_list)
                t = time.perf_counter() - start_t
                res.append( (scores, id_list, t) ) 

        return pd.DataFrame(res, columns=['scores', 'ids', 'time'])

    return f   



