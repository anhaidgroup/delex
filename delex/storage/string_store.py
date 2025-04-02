import pyspark.sql.functions as F
import numpy as np
from delex.storage.memmap_seqs import MemmapSeqs
from pydantic.dataclasses import dataclass
from delex.utils import CachedObjectKey

class MemmapStrings(MemmapSeqs):

    @dataclass(frozen=True)
    class CacheKey(CachedObjectKey):
        index_col : str

    @classmethod
    def build(cls, df, col, id_col='_id'):
        df = df.select(id_col, F.encode(col, 'utf-8').alias('bin_string'))\
                .filter(F.col('bin_string').isNotNull())
        return super().build(df, 'bin_string', np.uint8, id_col)
    
    def fetch_bytes(self, i):
        x = super().fetch(i)
        if x is None:
            return None
        else:
            return x.tobytes()

    def fetch(self, i):
        x = super().fetch(i)
        if x is None:
            return None
        else:
            return x.tobytes().decode('utf-8')
