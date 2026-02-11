from copy import deepcopy
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
from typing import Iterator, Tuple, Any, List, Optional
import pyarrow as pa
import pandas as pd
from delex.execution.partitioner import DataFramePartitioner
from delex.execution.cost_estimation import CostEstimator
from delex.execution.dataframe_stream import DataFrameStream
from delex.utils.funcs import persisted, get_logger, human_format_bytes
from delex.graph import PredicateNode, Node
from delex.graph.utils import nodes_to_dot
from delex.graph.algorithms import topological_sort
from delex.utils.funcs import type_check_call
from joblib import Parallel, delayed
from functools import cached_property
import time 


from delex.utils import BuildCache
from pydantic import (
        BaseModel,
        ConfigDict, 
        PositiveInt, 
        PrivateAttr, 
        computed_field,
        field_serializer
)

logger = get_logger(__name__)



class PartitionExecutionStats(BaseModel):
    """
    execution statistics for a single partition 
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    partitioner : DataFramePartitioner | None
    part_num : int | None
    build_time : float
    exec_time : float
    working_set_size : int

class SubGraphExecutionStats(BaseModel):
    """
    execution statistics for a subgraph
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    nodes : list[Node]
    partition_stats : list[PartitionExecutionStats]

    @computed_field
    def exec_time(self) -> float:
        return sum((s.exec_time for s in self.partition_stats))

    @computed_field
    def build_time(self) -> float:
        return sum((s.build_time for s in self.partition_stats))

    @computed_field
    def working_set_size(self) -> float:
        return sum((s.working_set_size for s in self.partition_stats))

    @computed_field
    def total_time(self) -> float:
        return self.build_time + self.exec_time
    
    @field_serializer('nodes')
    def _serialize_nodes(self, nodes, _info) -> list[str]:
        return [str(n) for n in nodes]

class GraphExecutionStats(BaseModel):
    """
    execution statistics for an entire execution plan
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    nodes : list[Node]
    sub_graph_stats : list[SubGraphExecutionStats]
    dot_graph : str

    @computed_field
    def exec_time(self) -> float:
        return sum((s.exec_time for s in self.sub_graph_stats))

    @computed_field
    def build_time(self) -> float:
        return sum((s.build_time for s in self.sub_graph_stats))

    @computed_field
    def working_set_size(self) -> float:
        return sum((s.working_set_size for s in self.sub_graph_stats))

    @computed_field
    def total_time(self) -> float:
        return self.build_time + self.exec_time

    @field_serializer('nodes')
    def _serialize_nodes(self, nodes, _info) -> list[str]:
        return [str(n) for n in nodes]



class GraphExecutor(BaseModel):
    """
    a class for executing a execution graph over two dataframes
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    index_table : pyspark.sql.DataFrame 
    search_table : pyspark.sql.DataFrame
    build_parallelism  : PositiveInt = 4
    index_table_id_col : str = '_id'
    ram_size_in_bytes : PositiveInt | None = None
    cost_est : CostEstimator | None = None
    # build cache
    _build_caches : Any = PrivateAttr(default_factory=dict)

    @computed_field
    def use_cost_estimation(self) -> bool:
        return self.cost_est is not None
    
    @computed_field
    def use_chunking(self) -> bool:
        return self.ram_size_in_bytes is not None and self.use_cost_estimation

    @computed_field
    @cached_property
    def index_table_count(self) -> int:
        return self.index_table.count()

    def _working_set_size(self, nodes, size=None):
        """
        get the estimated working set size of a list of nodes for a 
        given dataframe size 
        """
        components = {k for n in nodes for k in n.working_set_size()}
        return sum(self.cost_est.working_set_size(c, size=size) for c in  components)
    


    def _execute_with_chunking(self,
                               sorted_nodes: List[Node],
                               index_table: pyspark.sql.DataFrame,
                               search_table: pyspark.sql.DataFrame
                            ) -> Tuple[pyspark.sql.DataFrame, List[SubGraphExecutionStats]]:
        """
        execute a graph over `index_table` and `search_table` with 
        chunking enabled

        Parameters
        ----------
        sorted_nodes : List[Node]
            nodes to be executed, sorted in topological order

        index_table : pyspark.sql.DataFrame
            the table that will be indexed 

        search_table : pyspark.sql.DataFrame
            the table that will be used for probing

        Returns
        -------
        pyspark.sql.DataFrame
            `search_table` with any columns produced by sinks in `sorted_nodes` appended

        List[SubGraphExecutionStats]
        """
        # if we have enabled chunking and our working set size is too big
        topk_nodes = [n for n in sorted_nodes if not n.streamable]
        non_topk_nodes = [n for n in sorted_nodes if n.streamable]
        stats = []

        logger.warning(f'[EXEC_CHUNKING] splitting nodes: {len(topk_nodes)} topk (non-streamable), {len(non_topk_nodes)} streamable')
        for idx, n in enumerate(topk_nodes):
            logger.warning(f'[EXEC_CHUNKING] executing topk node {idx+1}/{len(topk_nodes)}: {n}')
            nodes = [n]
            search_table, sub_stats = self._exec_sub_graph(nodes, self.index_table, search_table)
            stats.append(sub_stats)
            logger.warning(f'[EXEC_CHUNKING] topk node {idx+1}/{len(topk_nodes)} completed')

        if len(non_topk_nodes) > 0:
            logger.warning(f'[EXEC_CHUNKING] executing {len(non_topk_nodes)} streamable nodes: {[str(n) for n in non_topk_nodes]}')
        search_table, sub_stats = self._exec_sub_graph(non_topk_nodes, self.index_table, search_table)
        stats.append(sub_stats)
        logger.warning(f'[EXEC_CHUNKING] all nodes completed')

        return search_table, stats

    @type_check_call
    def execute(self, sink: Node, search_table_id_col: str, projection: Optional[list[str]]=None):
        """
        execute the graph `sink` over self.index_table and self.search_table
        optionally, projecting columns `projection` along with the output of executing `sink`
        always projects the search table id column, renamed as 'id2'

        Parameters
        ----------
        sink : Node
            the sink of the execution graph

        search_table_id_col : str
            the column name for the id of the search table

        projection : Optional[list[str]] = None
            columns to be projected along with the output of `sink`

        Raises
        ------
        ValueError
            if `sink` is not a sink in the graph
        """
        if not sink.is_sink:
            raise ValueError('must pass sink')
        # compute this early to fail fast 
        dot_graph = nodes_to_dot(sink)
        sorted_nodes = topological_sort(sink)
        logger.warning(f'[EXECUTE] starting execution with {len(sorted_nodes)} nodes: {[str(n) for n in sorted_nodes]}')
        logger.warning(f'[EXECUTE] use_chunking={self.use_chunking}, use_cost_estimation={self.use_cost_estimation}')
        working_set_size = None
        if self.use_cost_estimation:
            working_set_size = self._working_set_size(sorted_nodes)
            logger.warning(f'[EXECUTE] estimated total working set size = {human_format_bytes(working_set_size)}')

        search_table = self.search_table

        if projection is None:
            projection = [search_table_id_col]
        elif search_table_id_col not in projection:
            projection = projection + [search_table_id_col]
            
        missing = [c for c in projection if c not in search_table.columns]
        if len(missing):
            raise ValueError(f'missing columns from projection {missing}')
        # project out unused columns
        deps = set(projection) | {d for n in sorted_nodes for d in n.iter_dependencies()}
        select = [c for c in search_table.columns if c in deps]
        logger.info(f'projecting {select}')
        search_table = search_table.select(*select)
        
        exec_start_t = time.perf_counter()
        if self.use_chunking and working_set_size >= self.ram_size_in_bytes:
            logger.warning(f'[EXECUTE] taking CHUNKING path (working_set_size={human_format_bytes(working_set_size)} >= ram_size={human_format_bytes(self.ram_size_in_bytes)})')
            search_table, stats = self._execute_with_chunking(sorted_nodes, self.index_table, search_table)
        else:
            logger.warning(f'[EXECUTE] taking NO-CHUNKING path')
            search_table, sub_stats = self._exec_sub_graph(sorted_nodes, self.index_table, search_table)
            stats = [sub_stats]

        logger.warning(f'[EXECUTE] execution completed in {time.perf_counter() - exec_start_t:.2f}s')
        graph_exec_stats = GraphExecutionStats(
                nodes=sorted_nodes,
                sub_graph_stats=stats,
                dot_graph=dot_graph
        )
        
        if projection is None:
            drop_cols = {n.output_col for n in sorted_nodes}
            select_exprs = [c for c in search_table.columns if c not in drop_cols]
        else:
            select_exprs = list(projection)

        output_expr = search_table[sorted_nodes[-1].output_col].getField('id1_list').alias('id1_list')
        select_exprs.append(output_expr)
        search_table = search_table.select(*select_exprs)
        search_table = search_table.withColumnRenamed(search_table_id_col, 'id2')
        return search_table, graph_exec_stats

    @staticmethod
    def _exp_search(n_recs, ram_size, f):
        """
        perform exponential search
        """
        n = 1 
        i = 1
        while f(n_recs / n) > ram_size:
            n += i
            i *= 2
        i //= 2
        while i > 0:
            if f(n_recs / (n - i)) <= ram_size:
                n -= i
            i //= 2

        return n


    def _get_num_chunks(self, nodes: List[Node]):
        """
        get the number of chunks such that the estimated working set size is <= self.ram_size_in_bytes
        """
        if self.use_chunking:
            f = lambda x: self._working_set_size(nodes, size=x)
            nchunks = self._exp_search(self.index_table_count, self.ram_size_in_bytes, f)
        else:
            nchunks = 1
        return nchunks

    def _execute_sub_graph_without_chunking(self,
                                            sorted_nodes: List[Node],
                                            index_table: pyspark.sql.DataFrame,
                                            search_table: pyspark.sql.DataFrame
                                        ) -> Tuple[pyspark.sql.DataFrame, SubGraphExecutionStats]:
        """
        execute a subgraph over `index_table` and `search_table` with 
        chunking disabled

        Parameters
        ----------
        sorted_nodes : List[Node]
            nodes to be executed, sorted in topological order
        index_table : pyspark.sql.DataFrame
            the table that will be indexed 
        search_table : pyspark.sql.DataFrame
            the table that will be used for probing

        Returns
        -------
        pyspark.sql.DataFrame
            `search_table` with any columns produced by sinks in `sorted_nodes` appended

        SubGraphExecutionStats
        """
        start_t = time.perf_counter()

        logger.warning(f'[NO_CHUNK] calling _exec_sub_graph_part (builds indexes, sets up mapInArrow)...')
        search_table, build_time, wss = self._exec_sub_graph_part(sorted_nodes, index_table, search_table, None, None, None)
        logger.warning(f'[NO_CHUNK] _exec_sub_graph_part returned, build_time={build_time:.2f}s, working_set_size={human_format_bytes(wss)}')
        logger.warning(f'[NO_CHUNK] calling persist()...')
        search_table.persist()
        logger.warning(f'[NO_CHUNK] calling count() -- this triggers all Spark computation (mapInArrow over entire search table)...')
        count_start_t = time.perf_counter()
        search_table.count()
        count_elapsed = time.perf_counter() - count_start_t
        logger.warning(f'[NO_CHUNK] count() completed in {count_elapsed:.2f}s')

        exec_time = time.perf_counter() - start_t - build_time
        stats = SubGraphExecutionStats(
                nodes=sorted_nodes,
                partition_stats = [PartitionExecutionStats(
                    partitioner=None, 
                    part_num=None, 
                    build_time=build_time,
                    exec_time=exec_time,
                    working_set_size=wss
                )]
        )


        return search_table, stats

    def _execute_sub_graph_with_chunking(self,
                                         sorted_nodes : List[Node],
                                         index_table : pyspark.sql.DataFrame, 
                                         search_table: pyspark.sql.DataFrame,
                                         nchunks: int,
                                         sinks: List[Node]
                                        )-> Tuple[pyspark.sql.DataFrame, SubGraphExecutionStats]:
        """
        execute a subgraph over `index_table` and `search_table` with 
        chunking enabled

        Parameters
        ----------
        sorted_nodes : List[Node]
            nodes to be executed, sorted in topological order

        index_table : pyspark.sql.DataFrame
            the table that will be indexed 

        search_table : pyspark.sql.DataFrame
            the table that will be used for probing

         nchunks: int
            the number of chunks to execute the subgraph in

         sinks: List[Node]
            a list of sinks for this sub graph

        Returns
        -------
        pyspark.sql.DataFrame
            `search_table` with any columns produced by `sinks` in `sorted_nodes` appended
            
        List[SubGraphExecutionStats]
        """
        partitioner = DataFramePartitioner(self.index_table_id_col, nchunks)
        out_cols = {n.output_col for n in sorted_nodes}
        in_cols = {x.output_col for n in sorted_nodes for x in n.iter_in()}
        # a list of columns that have been created by a different execution call
        external_columns = {f'{c}_TEMP_FULL' : c for c in in_cols - out_cols}
        stats = []
        # accumulator columns
        acc_cols = {s : f'{s.output_col}_TEMP_ACC' for s in sinks}

        for temp_col, c in external_columns.items():
            search_table = search_table.withColumnRenamed(c, temp_col)

        logger.warning(f'[CHUNKED] starting chunked execution: {nchunks} chunks, {len(external_columns)} external columns')
        total_chunk_start_t = time.perf_counter()
        for i in range(nchunks):
            logger.warning(f'[CHUNKED] === chunk {i+1}/{nchunks} starting ===')
            start_t = time.perf_counter()
            logger.warning(f'[CHUNKED] chunk {i+1}/{nchunks}: calling _exec_sub_graph_part...')
            stable, build_time, wss = self._exec_sub_graph_part(sorted_nodes, index_table, search_table, partitioner, i, external_columns)
            logger.warning(f'[CHUNKED] chunk {i+1}/{nchunks}: _exec_sub_graph_part returned, build_time={build_time:.2f}s, working_set_size={human_format_bytes(wss)}')
            # while this may check conditions multiple times, 
            # it is simplier and shouldn't affect runtime in any real way
            for s in sinks:
                if i == 0:
                    # first iter just rename
                    stable = stable.withColumnRenamed(s.output_col, acc_cols[s])

                elif isinstance(s, PredicateNode) and s.predicate.is_topk:
                    # merge the topk 
                    stable = stable.withColumn(acc_cols[s], GraphExecutor._merge_topk(stable, [acc_cols[s], s.output_col], s.predicate.k))\
                                    .drop(s.output_col)
                else:
                    # concat
                    stable = stable.withColumn(acc_cols[s], GraphExecutor._concat_structs(stable, [acc_cols[s], s.output_col]))\
                                    .drop(s.output_col)

            logger.warning(f'[CHUNKED] chunk {i+1}/{nchunks}: calling persist() + count()...')
            stable = stable.persist()
            count_start_t = time.perf_counter()
            stable.count()
            count_elapsed = time.perf_counter() - count_start_t
            logger.warning(f'[CHUNKED] chunk {i+1}/{nchunks}: count() completed in {count_elapsed:.2f}s')
            search_table.unpersist()
            search_table = stable

            exec_time = time.perf_counter() - start_t - build_time
            logger.warning(f'[CHUNKED] chunk {i+1}/{nchunks}: total chunk time={time.perf_counter() - start_t:.2f}s (build={build_time:.2f}s, exec={exec_time:.2f}s)')

            stats.append(
                    PartitionExecutionStats(
                        partitioner=partitioner, 
                        part_num=i, 
                        build_time=build_time,
                        exec_time=exec_time,
                        working_set_size=wss,
                    )
                )

        logger.warning(f'[CHUNKED] all {nchunks} chunks completed in {time.perf_counter() - total_chunk_start_t:.2f}s')

        for temp_col, c in external_columns.items():
            search_table = search_table.withColumnRenamed(temp_col, c)
            
        # do final rename of accumulated results 
        for s in sinks:
            search_table = search_table.withColumnRenamed(acc_cols[s], s.output_col)

        stats = SubGraphExecutionStats(
                nodes=sorted_nodes,
                partition_stats = stats
        )

        return search_table, stats


    def _exec_sub_graph(self,
                         sorted_nodes : List[Node],
                         index_table : pyspark.sql.DataFrame, 
                         search_table: pyspark.sql.DataFrame,
                        ):
        """
        execute a subgraph over `index_table` and `search_table`

        Parameters
        ----------
        sorted_nodes : List[Node]
            nodes to be executed, sorted in topological order

        index_table : pyspark.sql.DataFrame
            the table that will be indexed 

        search_table : pyspark.sql.DataFrame
            the table that will be used for probing

        Returns
        -------
        pyspark.sql.DataFrame
            `search_table` with any columns produced by sinks in `sorted_nodes` appended

        SubGraphExecutionStats
            stats for this call 
        """
        node_set = set(sorted_nodes)
        sinks = [n for n in sorted_nodes if n.is_sink or not any((i in node_set) for i in n.iter_out())]
        if len(sinks) == 0:
            raise ValueError('NO SINKS')

        nchunks = self._get_num_chunks(sorted_nodes)
        logger.warning(f'[SUB_GRAPH] executing nodes={[str(n) for n in sorted_nodes]}, sinks={[str(s) for s in sinks]}, nchunks={nchunks}')
        if nchunks == 1:
            logger.warning(f'[SUB_GRAPH] entering _execute_sub_graph_without_chunking')
            search_table, stats = self._execute_sub_graph_without_chunking(sorted_nodes, index_table, search_table)
        else:
            logger.warning(f'[SUB_GRAPH] entering _execute_sub_graph_with_chunking with {nchunks} chunks')
            search_table, stats = self._execute_sub_graph_with_chunking(sorted_nodes, index_table, search_table, nchunks, sinks)

        return search_table, stats


    @staticmethod
    def _build_df_stream(itr : Iterator[pa.RecordBatch], nodes: list[Node], schema: T.StructType) -> DataFrameStream:
        """
        create a DataFrameStream that applys the operations in `nodes`
        """
        stream = DataFrameStream.from_arrow_iter(itr, schema)
        columns_with_deps = {}
        for node in nodes:
            drop = []
            # apply operation
            stream = node.execute(stream)
            # update reference count
            columns_with_deps[node.output_col] = node.out_degree

            # garbage collection
            for dep in node.iter_in():
                dep_col = dep.output_col
                if dep_col not in columns_with_deps:
                    continue
                ref_count = columns_with_deps[dep_col] - 1
                if ref_count == 0:
                    drop.append(dep_col)
                else:
                    columns_with_deps[dep_col] = ref_count

            if len(drop):
                stream = stream.drop(drop)

        return stream

    @staticmethod
    def _exec_sub_graph_part_stream(itr : Iterator[pa.RecordBatch], nodes: list[Node], schema: T.StructType) -> Iterator[pa.RecordBatch]:
        """
        execute a subgraph using a DataFrameStream, return the result as a generator of arrow batches
        """
        import logging as _logging
        _worker_logger = _logging.getLogger('delex.worker')
        _worker_logger.warning(f'[WORKER] initializing {len(nodes)} nodes...')
        for node in nodes:
            _worker_logger.warning(f'[WORKER] init node: {node}')
            node.init()
        _worker_logger.warning(f'[WORKER] all nodes initialized, processing batches...')

        stream = GraphExecutor._build_df_stream(itr, nodes, schema)
        batch_idx = 0
        batch_start_t = time.perf_counter()
        for batch in stream.to_arrow_stream():
            batch_idx += 1
            elapsed = time.perf_counter() - batch_start_t
            _worker_logger.warning(f'[WORKER] batch {batch_idx} completed ({batch.num_rows} rows, {elapsed:.2f}s)')
            batch_start_t = time.perf_counter()
            yield batch
        _worker_logger.warning(f'[WORKER] all {batch_idx} batches completed')

    @staticmethod
    def _flat_to_nested_exprs(schema: T.StructType, prefix: str='') -> list:
        """
        create a pyspark select expression to convert a flat schema into a nested schema
        of structs.
        """
        select = []
        for f in schema.fields:
            if isinstance(f.dataType, T.StructType):
                expr = F.struct(GraphExecutor._flat_to_nested_exprs(f.dataType, f'{prefix}{f.name}.'))\
                        .alias(f.name)
            else:
                expr = F.col(f'`{prefix}{f.name}`').alias(f.name)
            select.append(expr)

        return select

    def _get_or_create_cache(self, partitioner: DataFramePartitioner, partition_num: int) -> BuildCache:
        """
        get or create a BuildCache for a particular dataframe partition
        """
        key = (partitioner, partition_num)
        if key not in self._build_caches:
            self._build_caches[key] = BuildCache()
        return self._build_caches[key]
    
    @staticmethod
    def _build_node(node: Node, index_table: pyspark.sql.DataFrame, id_col: str, cache: BuildCache):
        """
        build a node for execution using index table and id_col
        """
        logger.warning(f'[BUILD] building node: {node} (is_source={node.is_source})...')
        build_start = time.perf_counter()
        result = node.build(index_table, id_col, cache)
        logger.warning(f'[BUILD] node built: {node} in {time.perf_counter() - build_start:.2f}s')
        return result

    def _build_nodes(self,
            partitioner: DataFramePartitioner | None,
            part_num: int | None,
            index_table: pyspark.sql.DataFrame,
            nodes: list[Node]
        ) -> float:
        """
        build a list of nodes in parallel
        """
        logger.warning(f'[BUILD] building {len(nodes)} nodes with parallelism={self.build_parallelism}, partitioner={"None" if partitioner is None else f"part {part_num}"}')
        for node in nodes:
            logger.warning(f'[BUILD]   node: {node} (is_source={node.is_source})')
        start_t = time.perf_counter()
        cache = self._get_or_create_cache(partitioner, part_num)
        pool = Parallel(n_jobs=self.build_parallelism, backend='threading')

        if partitioner is not None:
            logger.warning(f'[BUILD] persisting index_table partition {part_num}...')
            with persisted(partitioner.get_partition(index_table, part_num)) as index_table:
                logger.warning(f'[BUILD] partition persisted, building nodes...')
                _ = pool(delayed(self._build_node)(node, index_table, self.index_table_id_col, cache) for node in nodes)
        else:
            logger.warning(f'[BUILD] building nodes over FULL index_table (no partitioning)...')
            _ = pool(delayed(self._build_node)(node, index_table, self.index_table_id_col, cache) for node in nodes)

        elapsed = time.perf_counter() - start_t
        logger.warning(f'[BUILD] all nodes built in {elapsed:.2f}s')
        return elapsed

    def _exec_sub_graph_part(self, 
                             sorted_nodes: List[Node],
                             index_table : pyspark.sql.DataFrame,
                             search_table : pyspark.sql.DataFrame,
                             partitioner : DataFramePartitioner | None,
                             partition_num : int | None,
                             external_columns : dict[str, str] | None
                            ):
        """
        Execute a subgraph over a partition of `index_table` using `partitioner`

        Parameters
        ----------
        sorted_nodes : List[Node]
            nodes to be executed, sorted in topological order

        index_table : pyspark.sql.DataFrame
            the table that will be indexed 

        search_table : pyspark.sql.DataFrame
            the table that will be used for probing

        partitioner : DataFramePartitioner | None
            the partitioner for `index_table`, if None execute over all of `index_table`

        partition_num : int | None
            the partition number for `index_table`

        external_columns : dict[str, str] | None
            a mapping of temporary column names to expected input columns generated by another call to 
            _exec_sub_graph_part. These columns are needed as input to nodes in `sorted_nodes` during execution.

        """
        logger.warning(f'[PART] building nodes (partitioner={"None" if partitioner is None else f"partition {partition_num}"})...')
        build_time = self._build_nodes(partitioner, partition_num, index_table, sorted_nodes)
        logger.warning(f'[PART] nodes built in {build_time:.2f}s')
        components = {}
        for n in sorted_nodes:
            components |= n.working_set_size()
        wss = sum(components.values())
        logger.warning(f'[PART] working set size = {human_format_bytes(wss)}')

        if external_columns is not None and len(external_columns):
            columns = list(search_table.columns)
            # filter the external columns using DataFramePartitioner so that 
            # only ids from this partition are fed into `sorted_nodes` for this call
            for temp_name, c in external_columns.items():
                fields = [f.name for f in search_table.schema[temp_name].dataType if f.dataType.simpleString().startswith('array')]
                temp = F.col(temp_name)
                expr = F.struct(*[
                                    partitioner.filter_array(
                                        temp.getField('id1_list'),
                                        temp.getField(f),
                                        partition_num
                                    ).alias(f) for f in fields
                                ]
                            ).alias(c)

                columns.append(expr)
            search_table = search_table.select(*columns)

        in_schema = T.StructType([deepcopy(f) for f in search_table.schema])
        stream = GraphExecutor._build_df_stream([], sorted_nodes, in_schema)
        flat_schema = stream.spark_schema(flat=True)
        logger.warning(f'[PART] setting up mapInArrow (actual computation deferred to persist/count)...')
        search_table = search_table.mapInArrow(
                lambda x : GraphExecutor._exec_sub_graph_part_stream(x, sorted_nodes, in_schema),
                schema=flat_schema
            )

        # drop the sliced columns
        if external_columns is not None and len(external_columns):
            stream = stream.drop(list(external_columns.values()))

        search_table = search_table.select(*GraphExecutor._flat_to_nested_exprs(stream.spark_schema()))
        logger.warning(f'[PART] DataFrame transformations set up, returning (not yet materialized)')
        return search_table, build_time, wss


    @staticmethod
    def _merge_topk(df: pyspark.sql.DataFrame, struct_cols: list[str], k: int):
        """
        merge the topk results in spark
        """
        cols = [df[c] for c in struct_cols]
        id_lists = F.concat(*[c.getField('id1_list') for c in cols])
        scores = F.concat(*[c.getField('scores') for c in cols])
        expr = GraphExecutor._topk_spark(scores, id_lists, k)
        return expr

    @staticmethod
    def _topk_spark(scores_col: list[pyspark.sql.Column], ids_col: list[pyspark.sql.Column], k : int) -> pyspark.sql.Column:
        if not isinstance(k, int):
            raise TypeError(f'k must be int, {type(k)=}')
        if k < 1:
            raise ValueError(f'k must be >= 1, {k=}')


        @F.pandas_udf('scores array<float>, id1_list array<long>')
        def _topk_impl(itr: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.DataFrame]:
            res = []
            for pair in itr:
                res.clear()
                for scores, id1_list in zip(pair[0], pair[1]):
                    if len(id1_list) > k:
                        indexes = scores.argsort()[::-1][:k]
                        id1_list = id1_list[indexes]
                        scores = scores[indexes]
                    res.append( (scores, id1_list) )

                yield pd.DataFrame(res, columns=['scores', 'id1_list'])

        return _topk_impl(scores_col, ids_col)


    @staticmethod
    def _concat_structs(df: pyspark.sql.DataFrame, struct_cols: list[str]) -> pyspark.sql.Column:
        """
        concatentate structs with array fields
        """
        cols = [df[c] for c in struct_cols]
        id_lists = F.concat(*[c.getField('id1_list') for c in cols]).alias('id1_list')
        fields = [id_lists]
        if all(('scores' in df.schema[c].dataType.fieldNames()) for c in struct_cols):
            scores = F.concat(*[c.getField('scores') for c in cols]).alias('scores')
            fields.append(scores)

        expr = F.struct(*fields)
        return expr

