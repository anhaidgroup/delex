import numpy as np
from delex.storage import MemmapArray, StringIntHashMap
import pyspark
import pyspark.sql.functions as F
from delex.utils.traits import SparkDistributable
from delex.utils.funcs import type_check_call
from delex.utils.build_cache import CachedObjectKey
from pydantic.dataclasses import dataclass
from typing import Union

class HashIndex(SparkDistributable):
    """
    a memory mapped hash index to be used on Spark
    """

    @dataclass(frozen=True)
    class CacheKey(CachedObjectKey):
        index_col : str
        lowercase : bool
    
    def __init__(self):
        self._offset_arr = None
        self._string_to_idx = None
        self._id_lists = None
    
    def init(self):
        self._id_lists.init()
        self._offset_arr.init()
        self._string_to_idx.init()

    def deinit(self):
        self._id_lists.deinit()
        self._offset_arr.deinit()
        self._string_to_idx.deinit()

    def to_spark(self):
        self._id_lists.to_spark()
        self._offset_arr.to_spark()
        self._string_to_idx.to_spark()

    def size_in_bytes(self):
        return self._id_lists.size_in_bytes() +\
                self._offset_arr.size_in_bytes() +\
                self._string_to_idx.size_in_bytes()
    
    @type_check_call
    def build(self, index_table: pyspark.sql.DataFrame, index_col: str, id_col: str='_id'):
        """
        build the index over `index_col` of `index_table` using `id_col` as a unique id, 

        Parameters
        ----------

        index_table : pyspark.sql.DataFrame
            the dataframe that will be preprocessed / indexed

        index_col : str
            the name of the string column to be indexes

        id_col : str
            the name of the unique id column in `index_table`
        """
        strings = index_table.select(F.col(index_col).alias('key'), id_col)\
                        .filter(F.col('key').isNotNull())\
                        .groupBy('key')\
                        .agg(F.collect_list(id_col).alias('ids'))\
                        .select('key', F.array_sort(F.col('ids')).alias('ids'))\
                        .toPandas()
    
        keys = strings['key']
        id_lists = strings['ids']
        size_arr = np.fromiter(map(len, id_lists), dtype=np.int32, count=len(id_lists))
        self._id_lists = MemmapArray(np.concatenate(id_lists))
        self._offset_arr = MemmapArray(np.cumsum(np.concatenate([np.zeros(1, dtype=np.int32), size_arr]))) 
        self._string_to_idx = StringIntHashMap.build(
                keys, 
                np.arange(len(keys)),
                load_factor=.5
            )

    def fetch(self, key: str) -> Union[np.ndarray, None]:
        """
        fetch all records with `key`, return None if entry doesn't exist in index

        Parameters
        ----------
        key : str
            the key to retrieve 

        Returns
        -------
        a numpy array of ids if `key` is found else None
        """
        idx = self._string_to_idx[key]
        if idx < 0:
            return None
        else:
            start = self._offset_arr.values[idx]
            end = self._offset_arr.values[idx+1]
            return self._id_lists.values[start:end]
