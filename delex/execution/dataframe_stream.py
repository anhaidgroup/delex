import pyspark.sql.types as T
import pandas as pd
from copy import deepcopy
import pyarrow.parquet
import pyarrow as pa
from typing import Iterable, Callable, Iterator


_PYSPARK_TO_PA_TYPE = {
        T.DoubleType() : pa.float64(),
        T.FloatType() : pa.float32(),
        T.LongType() : pa.int64(),
        T.IntegerType() : pa.int32(),
        T.StringType() : pa.string(),
        T.ArrayType(T.LongType()) : pa.list_(pa.int64()),
        T.ArrayType(T.FloatType()) : pa.list_(pa.float32()),
        T.TimestampType() : pa.timestamp('us')
}

class DataFrameStream:
    """
    a stream of dataframes on which tranformations can be applied
    this replaces pyspark based execution for resource allocation issues
    """
    def __init__(self, stream: Iterable, schema: dict):
        self._stream = stream
        self._schema: dict = schema

    def __iter__(self):
        return iter(self._stream)
    
    def spark_schema(self, flat: bool=False) -> T.StructType:
        """
        return schema as a pyspark schema 

        Parameters
        ----------
        flat : bool = False
            if True return the schema in flattened format

        Returns
        -------
        T.StructType
        """
        schema = self._schema if not flat else self._flatten_dict(self.schema)
        return self._dict_to_pyspark_schema(schema)

    def pyarrow_schema(self, flat: bool=False) -> pa.Schema:
        """
        return schema as a pyarrow schema 

        Parameters
        ----------
        flat : bool = False
            if True return the schema in flattened format

        Returns
        -------
        pa.Schema
        """
        schema = self._schema if not flat else self._flatten_dict(self.schema)
        schema = self._dict_to_pyarrow_schema(schema)
        return pa.schema(list(schema))
    
    @staticmethod
    def _record_batch_to_dict(batch):
        out = {}
        schema = batch.schema
        for field in schema:
            if isinstance(field.type, pa.StructType):
                sub_batch = pa.RecordBatch.from_struct_array(batch.column(field.name))
                out[field.name] = DataFrameStream._record_batch_to_dict(sub_batch)
            else:
                out[field.name] = batch.column(field.name).to_numpy(zero_copy_only=False)

        return out

    @staticmethod
    def _dict_to_record_batch(data, schema):
        out = {}
        for field in schema:
            if isinstance(field.type, pa.StructType):
                batch = DataFrameStream._dict_to_record_batch(data[field.name], field)
                out[field.name] = batch.to_struct_array()
            else:
                out[field.name] = pa.array(data[field.name], field.type)
        
        return pd.RecordBatch.from_pydict(out, schema=schema)

    @staticmethod
    def _dict_to_pyarrow_schema(schema):
        if isinstance(schema, dict):
            return pa.struct([pa.field(n, DataFrameStream._dict_to_pyarrow_schema(t)) for n,t in schema.items()])
        else:
            return _PYSPARK_TO_PA_TYPE[schema]

    @property
    def schema(self) -> dict:
        return self._schema
    
    @staticmethod
    def _pyspark_schema_to_dict(schema):
        if isinstance(schema, T.StructType):
            return {f.name : DataFrameStream._pyspark_schema_to_dict(f.dataType) for f in schema.fields}
        else:
            return schema

    @staticmethod
    def _get_column(data, col):
        if isinstance(col, str):
            return data[col]
        elif isinstance(col, tuple) and len(col) == 2:
            return data[col[0]][col[1]]
        else:
            raise ValueError(f'{col=}')

    @staticmethod
    def _drop_column(data, col):
        if isinstance(col, str):
            return data.pop(col)
        elif isinstance(col, tuple) and len(col) == 2:
            return data[col[0]].pop(col[1])
        else:
            raise ValueError(f'{col=}')

    @staticmethod
    def _flatten_dict(d):
        out = {}
        for key, val in d.items():
            if isinstance(val, dict):
                sub = DataFrameStream._flatten_dict(val)
                out.update({f'{key}.{k}' :v for k,v in sub.items()})
            else:
                out[key] = val
        return out

    @staticmethod
    def _dict_to_pyspark_schema(schema):
        if isinstance(schema, dict):
            return T.StructType([T.StructField(n, DataFrameStream._dict_to_pyspark_schema(t)) for n,t in schema.items()])
        else:
            return schema

    @classmethod
    def from_pandas_iter(cls, itr: Iterable[pd.DataFrame], schema: T.StructType):
        """
        create a DataFrameStream from an iterable of pd.DataFrames and a pyspark Schema
        """
        schema = cls._pyspark_schema_to_dict(schema)
        stream = ({c : df[c].values for c in df} for df in itr)
        return cls(stream, schema)

    @classmethod
    def from_arrow_iter(cls, itr: Iterable[pa.RecordBatch], schema: pa.Schema):
        """
        create a DataFrameStream from an iterable of pyarrow RecordBatchs and a pyarrow Schema
        """
        schema = cls._pyspark_schema_to_dict(schema)
        
        stream = map(cls._record_batch_to_dict, itr)
        return cls(stream, schema)

    def _exists_or_raise(self, columns):
        try:
            for c in columns:
                self._get_column(self._schema, c)
        except KeyError:
            raise RuntimeError(f'unresolved column {c}')

    def _apply(self, func, input_cols: list[str], out_name : str, out_dtype : T.DataType | dict):

        if isinstance(out_dtype, dict):
            for data in self._stream:
                out = func(*(self._get_column(data, c) for c in input_cols))
                data[out_name] = {c : out[c].values for c in out_dtype}
                yield data
        else:
            for data in self._stream:
                out = func(*(self._get_column(data, c) for c in input_cols))
                data[out_name] = out
                yield data
            
 
    def apply(self, func: Callable, input_cols: list[str | tuple], out_name : str, out_dtype : T.DataType):
        """
        apply `func` with `input_cols` to this stream and append the result to the stream as `out_name` with 
        data type `out_dtype` return a new DataFrameStream

        Parameters
        ----------
        func : Callable
            the function that will be executed over the stream

        input_cols : list[str | tuple]
            the list of input columns for `func`, if nested tuples are provided

        out_name : str
            the name of the output column to be added to the stream

        out_dtype  : T.DataType
            the type returned by `func`


        Returns
        -------
        DataFrameStream
            a new dataframe stream

        Raises
        ------
        KeyError 
            if any of `input_cols` cannot be resolved
        """
        self._exists_or_raise(input_cols)
        schema = deepcopy(self._schema)
        out_dtype = self._pyspark_schema_to_dict(out_dtype)
        schema[out_name] = out_dtype
        stream = self._apply(func, input_cols, out_name, out_dtype)

        return DataFrameStream(stream, schema)
 
    def _drop(self, columns: list[str | tuple]):
        for data in self._stream:
            for c in columns:
                self._drop_column(data, c)
            yield data

    def drop(self, columns):
        """
        drop `columns` from this dataframe stream, return a new stream
        Parameters
        ----------
        columns : list[str | tuple]
            the list of input columns for `func`, if nested tuples are provided

        Returns
        -------
        DataFrameStream
            a new dataframe stream with `columns` removed

        Raises
        ------
        KeyError 
            if any of `columns` cannot be resolved
        """
        self._exists_or_raise(columns)

        schema = deepcopy(self._schema)
        for c in columns:
            self._drop_column(schema, c)

        stream  = self._drop(columns)
        return DataFrameStream(stream, schema)

    def to_pandas_stream(self) -> Iterator[pd.DataFrame]:
        """
        convert this dataframe stream into an iterator of pandas DataFrames
        """
        for data in self._stream:
            yield pd.DataFrame(data)

    def to_arrow_stream(self) -> Iterator[pa.RecordBatch]:
        """
        convert this dataframe stream into an iterator of pyarrow RecordBatchs
        """
        schema = self.pyarrow_schema(flat=True)
        cols = []
        for f in schema:
            if '.' in f.name:
                col = (tuple(f.name.split('.')), f)
            else:
                col = (f.name, f)
            cols.append(col)

        for data in self._stream:
            arrs = [pa.array(self._get_column(data, c), f.type) for c,f in cols]
            yield pa.RecordBatch.from_arrays(arrs, schema=schema)

