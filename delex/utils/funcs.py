from pathlib import Path
from contextlib import contextmanager
from pyspark import StorageLevel
import math
from pydantic import validate_call, ConfigDict
import logging
import numpy as np
import pyarrow as pa
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark import SparkContext, TaskContext

try:
    import lucene
except ImportError:
    #warnings.warn("Unable to import pylucene, this may cause the program to fail", UserWarning)
    pass

type_check_call = validate_call(config=ConfigDict(arbitrary_types_allowed=True))

@type_check_call
def size_in_bytes(f: Path, /) -> int:
    """
    get the size on disk in bytes of `f`

    Parameters
    ----------
    f : Path
        path to the file or directory on the local filesystem

    Returns
    -------
    int
        if `f` is a file, return the size of the single file
        else get total size in bytes of all files in the directory similar to `du` utility

    Raises
    ------
    FileNotFoundError
        if `f` doesn't exist
    """
    if not f.exists():
        raise FileNotFoundError(f'cannot take size of file that doesn\'t exist {f}')

    if f.is_file():
        return f.stat().st_size
    else:
        return sum(x.stat().st_size for x in f.glob('**/*') if x.is_file())

def init_jvm(vmargs=[]):
    if not lucene.getVMEnv():
        lucene.initVM(vmargs=['-Djava.awt.headless=true'] + vmargs)


def attach_current_thread_jvm():
    env = lucene.getVMEnv()
    env.attachCurrentThread()

def get_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger

def human_format_bytes(n):
    units = ['', 'KiB', 'MiB', 'GiB', 'TiB']
    k = 2**10
    magnitude = int(math.floor(math.log(n, k)))
    if magnitude == 0:
        return str(n)
    return '%.4f%s' % (n / k**magnitude, units[magnitude])

def type_check(var, var_name, expected):
    """
    type checking utility, throw a type error if the var isn't the expected type
    """
    if not isinstance(var, expected):
        raise TypeError(f'{var_name} must be type {expected} (got {type(var)})')

@contextmanager
def persisted(df, storage_level=StorageLevel.MEMORY_AND_DISK):
    if df is not None and not is_persisted(df):
        df = df.persist(storage_level) 
    try:
        yield df
    finally:
        if df is not None:
            df.unpersist()

def is_persisted(df):
    sl = df.storageLevel
    return sl.useMemory or sl.useDisk


def with_rid(df, rid_col, storage_level=StorageLevel.MEMORY_AND_DISK):
    """
    return a persisted copy of `df` with `rid_col` appended, a dense unique
    id in the range [0, df.count())

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        the dataframe to be assigned ids
    rid_col : str
        the name of the id column to be appended
    storage_level : StorageLevel
        the storage level used to persist the result

    Returns
    -------
    pyspark.sql.DataFrame
        `df` with `rid_col` appended, persisted with `storage_level`
    """
    cache_input = not is_persisted(df)
    if cache_input:
        df = df.persist(storage_level)
    try:
        counts = df.groupBy(F.spark_partition_id().alias('pid')).count().collect()
        offsets = {}
        start = 0
        for row in sorted(counts, key=lambda r: r['pid']):
            offsets[row['pid']] = start
            start += row['count']

        schema = T.StructType(df.schema.fields + [T.StructField(rid_col, T.LongType(), False)])

        def append_rid(itr):
            start = offsets.get(TaskContext.get().partitionId(), 0)
            for batch in itr:
                rids = pa.array(np.arange(start, start + batch.num_rows, dtype=np.int64))
                start += batch.num_rows
                yield pa.RecordBatch.from_arrays(batch.columns + [rids], names=batch.schema.names + [rid_col])

        out = df.mapInArrow(append_rid, schema).persist(storage_level)
        out.count()
    finally:
        if cache_input:
            df.unpersist()
    return out


def get_num_partitions(df, chunk_size=2000, n_rows=None):
    # get the number of partitions based on the size of the dataframe
    # chunk_size is the number of rows per partition
    logger = get_logger(__name__)
    num_cores = SparkContext.getOrCreate().defaultParallelism
    if n_rows is None:
        n_rows = df.count()
    chunks = n_rows / chunk_size
    partitions = max(num_cores * 2, math.ceil(chunks))
    logger.info(f"Repartitioning {n_rows} rows into {partitions} partitions")
    return partitions
