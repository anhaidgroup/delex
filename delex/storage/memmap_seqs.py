import pyspark.sql.functions as F
import pyspark
import numpy as np
from tempfile import mkstemp
from pathlib import Path
from delex.storage import MemmapArray, IdOffsetHashMap
from delex.utils.traits import SparkDistributable
from delex.utils.funcs import type_check_call

class MemmapSeqs(SparkDistributable):
    """
    a class to hold arbitrary sequences of elements 
    e.g. strings, arrays of ints, etc.
    """
    def __init__(self):
        self._offset_arr = None
        self._seq_arr = None
        self._id_to_offset_map = None
    
    def size_in_bytes(self) -> int:
        """
        return the size in bytes on disk
        """
        return self._seq_arr.size_in_bytes()\
                + self._id_to_offset_map.size_in_bytes()\
                + self._offset_arr.size_in_bytes()

    def init(self):
        self._seq_arr.init()
        self._offset_arr.init()
        self._id_to_offset_map.init()

    def deinit(self):
        self._seq_arr.deinit()
        self._offset_arr.deinit()
        self._id_to_offset_map.deinit()
    
    
    def delete(self):
        self._seq_arr.delete()
        self._offset_arr.delet()
        self._id_to_offset_map.delete()

    def to_spark(self):
        self._seq_arr.to_spark()
        self._id_to_offset_map.to_spark()
        self._offset_arr.to_spark()


    @classmethod
    @type_check_call
    def build(cls, df: pyspark.sql.DataFrame, seq_col: str, dtype: type, id_col: str='_id'):
        """
        create a MemmapSeqs instance from a spark dataframe 

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            the dataframe containing the sequences and ids
        seq_col : str
            the name of the column in `df` that contains the sequences, e.g. strings, arrays
        dtype : type
            the dtype of the elements in `seq_col`
        id_col : str
            the name of the column in `df` that contains the ids for retrieving the sequences

        Returns
        -------
        MemmapSeqs
        """
        dtype = np.dtype(dtype)
        obj = cls()
        is_array_col = df.schema[seq_col].dataType.typeName().startswith('array')
        # pyspark has different functions for array vs string or binary length, because.... idk
        size_func = F.size if is_array_col else F.length
        df = df.filter(df[seq_col].isNotNull())\
                .select(id_col, seq_col, size_func(seq_col).alias('size'))

        pdf = df.toPandas()
        #assert np.all(id_and_size[id_col].values > 0)

        local_mmap_file = Path(mkstemp(suffix='.mmap_arr')[1])
        # buffer 256MB at a time
        with open(local_mmap_file, 'wb', buffering=(2**20) * 256) as ofs:
            seqs = pdf[seq_col]
            if is_array_col and not isinstance(seqs.iloc[0], np.ndarray):
                # create arrays out of lists if lists were returned
                for seq in seqs:
                    seq = np.array(seq, dtype=dtype)
                    ofs.write(memoryview(seq))
            else:
                for seq in seqs:
                    ofs.write(memoryview(seq))


        id_arr = pdf[id_col].to_numpy(dtype=np.uint64)
        size_arr = pdf['size'].to_numpy(dtype=np.uint64) 
        
        obj._id_to_offset_map = IdOffsetHashMap.build(
                id_arr,
                np.arange(len(id_arr), dtype=np.int32)
        )

        obj._offset_arr = MemmapArray(np.cumsum(np.concatenate([np.zeros(1, dtype=np.uint64), size_arr]))) 
        seq_arr = np.memmap(local_mmap_file, shape=obj._offset_arr.values[-1], dtype=dtype, mode='r+')
        
        obj._seq_arr = MemmapArray(seq_arr)
        return obj

    def fetch(self, i: int, /) -> np.ndarray | None:
        """
        retrieve the sequence associated with `i`

        Returns 
        -------
        np.ndarray if `i` is found, else None
        """
        idx = self._id_to_offset_map[i]
        if idx == -1:
            return None
        else:
            start = self._offset_arr.values[idx]
            end = self._offset_arr.values[idx+1]
            return self._seq_arr.values[start:end]


