import pyspark.sql.functions as F
from pydantic import PositiveInt
from pydantic.dataclasses import dataclass
import pyspark
from delex.utils.funcs import type_check_call

@dataclass(frozen=True)
class DataFramePartitioner:
    """
    A simple class for hash paritioning dataframes
    using the xxhash64 implementation in pyspark
    """
    column : str
    nparts : PositiveInt

    @type_check_call
    def get_partition(self, df: pyspark.sql.DataFrame, pnum: int) -> pyspark.sql.DataFrame:
        """
        get partition `pnum` of `df`

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            the dataframe to be partitioned
        pnum : int
            the partition number

        Returns
        -------
        pyspark.sql.DataFrame
            the parition of `df`

        Raises
        ------
        ValueError
            if pnum < 0 or pnum >= self.nparts:
        """
        if pnum < 0 or pnum >= self.nparts:
            raise ValueError(str(pnum))

        if self.nparts == 1:
            return df
        else:
            return df.filter(F.abs(F.xxhash64(self.column) % F.lit(self.nparts)) == F.lit(pnum))

    @type_check_call
    def filter_array(self, ids: str | pyspark.sql.Column, arr: str | pyspark.sql.Column | None, pnum: int):
        """
        filter an array column based on ids

        Parameters
        ----------
        ids : str | pyspark.sql.Column
            array<long> column used to partition the dataframe
        arr : Optional[str | pyspark.sql.Column]
            the array column that will be filtered, and returned, if not provided, `ids` will be filtered and returned
        pnum : int
            the partition number

        Returns
        -------
        pyspark.sql.Column
            a column expression for the filtered array

        Raises
        ------
        ValueError
            if pnum < 0 or pnum >= self.nparts:
        """
        if pnum < 0 or pnum >= self.nparts:
            raise ValueError(str(pnum))

        if arr is None:
            return F.filter(ids, lambda x : F.abs(F.xxhash64(x) % F.lit(self.nparts)) == F.lit(pnum))
        else:
            return F.filter(arr, lambda x, i : F.abs(F.xxhash64(ids.getItem(i)) % F.lit(self.nparts)) == F.lit(pnum))
