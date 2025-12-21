"""
Tests for execution.dataframe_stream module.

This module tests DataFrame streaming functionality.
"""
import pytest
from delex.execution.dataframe_stream import DataFrameStream


@pytest.mark.unit
class TestDataFrameStream:
    """Tests for DataFrameStream class."""

    def test_dataframe_stream_init(self):
        """Test DataFrameStream initialization."""
        stream = DataFrameStream(stream=range(10), schema={})
        assert stream is not None
        assert stream._stream is not None
        assert stream._schema is not None
        assert len(stream._stream) == 10
        assert len(stream._schema) == 0

    def test_dataframe_stream_iter(self):
        """Test DataFrameStream iteration."""
        import pandas as pd
        stream = DataFrameStream(stream=[pd.DataFrame({'a': range(10)}), pd.DataFrame({'b': range(10)})], schema={'a': 'int', 'b': 'int'})
        assert stream is not None
        assert stream._stream is not None
        assert stream._schema is not None
        assert len(stream._stream) == 2
        assert len(stream._schema) == 2
        for df in stream:
            assert df is not None
            assert isinstance(df, pd.DataFrame)

    def test_dataframe_stream_from_pandas_iter(self):
        """Test creating DataFrameStream from pandas iterable."""
        import pandas as pd
        from typing import Iterator
        stream = DataFrameStream.from_pandas_iter(itr=[pd.DataFrame({'a': range(10)}), pd.DataFrame({'b': range(10)})], schema={'a': 'int', 'b': 'int'})
        assert stream is not None
        assert stream._stream is not None
        assert stream._schema is not None
        assert isinstance(stream._stream, Iterator)
        assert len(stream._schema) == 2
        for df in stream:
            assert df is not None
            assert isinstance(df, dict)

    def test_dataframe_stream_from_arrow_iter(self):
        """Test creating DataFrameStream from arrow iterable."""
        import pyarrow as pa
        from typing import Iterator
        stream = DataFrameStream.from_arrow_iter(itr=[pa.RecordBatch.from_pydict({'a': range(10)}), pa.RecordBatch.from_pydict({'b': range(10)})], schema={'a': 'int', 'b': 'int'})
        assert stream is not None
        assert stream._stream is not None
        assert stream._schema is not None
        assert isinstance(stream._stream, Iterator)
        assert len(stream._schema) == 2
        for df in stream:
            assert df is not None
            assert isinstance(df, dict)

    def test_dataframe_stream_spark_schema(self):
        """Test spark_schema method."""
        import pyspark.sql.types as T
        from pyspark.sql.types import StructType
        schema = T.StructType([T.StructField('a', T.IntegerType()), T.StructField('b', T.IntegerType())])
        stream = DataFrameStream(stream=[{'a': 1, 'b': 2}, {'a': 3, 'b': 4}], schema=schema)
        assert stream is not None
        assert stream._stream is not None
        assert stream._schema is not None
        assert isinstance(stream._schema, StructType)
        assert len(stream._schema) == 2
        assert stream.spark_schema() == schema

    def test_dataframe_stream_pyarrow_schema(self):
        """Test pyarrow_schema method."""
        import pyarrow as pa
        schema = pa.schema([pa.field('a', pa.int32()), pa.field('b', pa.int32())])
        stream = DataFrameStream.from_arrow_iter(itr=[pa.RecordBatch.from_pydict({'a': [1, 2, 3], 'b': [4, 5, 6]}), pa.RecordBatch.from_pydict({'a': [7, 8, 9], 'b': [10, 11, 12]})], schema=schema)
        assert stream is not None
        assert stream._stream is not None
        assert stream._schema is not None
        print(list(stream._stream))
        print(stream._schema)
        assert isinstance(stream._schema, pa.Schema)
        assert len(stream._schema) == 2
    
    def test_dataframe_stream_to_pandas_stream(self):
        """Test to_pandas_stream method."""
        import pandas as pd
        import pyspark.sql.types as T
        import numpy as np

        schema = T.StructType([
            T.StructField('a', T.IntegerType()),
            T.StructField('b', T.IntegerType())
        ])
        schema_dict = DataFrameStream._pyspark_schema_to_dict(schema)
        stream = DataFrameStream(
            stream=[
                {'a': np.array([1]), 'b': np.array([2])},
                {'a': np.array([3]), 'b': np.array([4])}
            ],
            schema=schema_dict
        )
        for df in stream.to_pandas_stream():
            assert df is not None
            assert isinstance(df, pd.DataFrame)

    def test_dataframe_stream_to_arrow_stream(self):
        """Test to_arrow_stream method."""
        import pyarrow as pa
        import pyspark.sql.types as T
        import numpy as np

        schema = T.StructType([
            T.StructField('a', T.IntegerType()),
            T.StructField('b', T.IntegerType())
        ])
        schema_dict = DataFrameStream._pyspark_schema_to_dict(schema)
        stream = DataFrameStream(
            stream=[
                {'a': np.array([1]), 'b': np.array([2])},
                {'a': np.array([3]), 'b': np.array([4])}
            ],
            schema=schema_dict
        )
        for batch in stream.to_arrow_stream():
            assert batch is not None
            assert isinstance(batch, pa.RecordBatch)

    def test_dataframe_stream_drop(self):
        """Test drop method."""
        import pyspark.sql.types as T
        schema = T.StructType([
            T.StructField('a', T.IntegerType()),
            T.StructField('b', T.IntegerType())
        ])
        schema_dict = DataFrameStream._pyspark_schema_to_dict(schema)
        stream = DataFrameStream(stream=[{'a': 1, 'b': 2}, {'a': 3, 'b': 4}], schema=schema_dict)
        stream = stream.drop(['a'])
        for data in stream._stream:
            assert 'a' not in data
            assert 'b' in data

    def test_dataframe_stream_apply(self):
        """Test apply method."""
        import pyspark.sql.types as T
        schema = T.StructType([
            T.StructField('a', T.IntegerType()),
            T.StructField('b', T.IntegerType())
        ])
        schema_dict = DataFrameStream._pyspark_schema_to_dict(schema)
        # Use arrays to match how .apply unpacks to positional arguments
        import numpy as np
        stream = DataFrameStream(
            stream=[
                {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])},
                {'a': np.array([7, 8, 9]), 'b': np.array([10, 11, 12])}
            ],
            schema=schema_dict
        )
        # Lambda must take two arguments ('a', 'b') not one dict
        stream = stream.apply(lambda a, b: a + b, ['a', 'b'], 'c', T.IntegerType())
        for data in stream._stream:
            assert 'c' in data
            np.testing.assert_array_equal(
                data['c'],
                data['a'] + data['b']
            )