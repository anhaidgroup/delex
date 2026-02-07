import pandas as pd
import pyspark.sql as sql
import pyspark.sql.functions as F


def check_tables(table_a : sql.DataFrame, id_col_table_a : str, table_b : sql.DataFrame, id_col_table_b : str):
    """
    Check that table_a and table_b have valid id columns.

    Parameters
    ----------
    table_a : sql.DataFrame
        The table A to be indexed.
    id_col_table_a : str
        The column name of the id column in table A.
    table_b : sql.DataFrame
        The table B to be searched.
    id_col_table_b : str
        The column name of the id column in table B.

    Raises
    ------
    ValueError
        If table_a or table_b do not have a valid id column.
    """
    # --- spark ---
    if isinstance(table_a, sql.DataFrame):
        if not isinstance(table_b, sql.DataFrame):
            raise TypeError("table_a and table_b must both be Spark DataFrames")

        _check_id_spark(table_a, id_col_table_a, "table_a")
        _check_id_spark(table_b, id_col_table_b, "table_b")

        return

    raise TypeError("table_a must be Spark DataFrame")


def _check_id_spark(df: sql.DataFrame, id_col: str, name: str) -> None:
    """
    Check that the id column is in the dataframe and is a valid id column.
    """
    if id_col not in df.columns:
        raise ValueError(f"{name}: missing id column '{id_col}'")
    if len(df.take(1)) == 0:
        raise ValueError(f"{name}: empty dataframe")
    if dict(df.dtypes)[id_col] not in ("int", "bigint", "smallint", "tinyint"):
        raise ValueError(f"{name}: id column '{id_col}' must be an integer type")
    if df.filter(F.col(id_col).isNull()).limit(1).count() > 0:
        raise ValueError(f"{name}: nulls are presentin the id column '{id_col}'")
    if df.select(id_col).distinct().count() != df.count():
        raise ValueError(f"{name}: id column '{id_col}' must be unique")
