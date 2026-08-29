import pyspark.sql as sql
import pyspark.sql.functions as F

from delex.utils.funcs import get_logger

logger = get_logger(__name__)


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
        
        logger.warning("check_tables: table_a and table_b formats are correct")

        return

    raise TypeError("table_a must be Spark DataFrame")


def check_id_col(df: sql.DataFrame, id_col: str, name: str) -> int:
    """
    Check that `id_col` is an integer column of `df` containing no nulls
    and no duplicate values.

    Parameters
    ----------
    df : sql.DataFrame
        The table to be checked.
    id_col : str
        The column name of the id column in `df`.
    name : str
        The name of `df` used in error messages.

    Returns
    -------
    int
        The number of rows in `df`.

    Raises
    ------
    ValueError
        If `id_col` is missing, not an integer type, contains nulls or
        contains duplicate values.
    """
    _check_id_schema_spark(df, id_col, name)
    return _check_id_values_spark(df, id_col, name)


def _check_id_schema_spark(df: sql.DataFrame, id_col: str, name: str) -> None:
    """
    Check that the id column is in the dataframe and is an integer type.
    """
    if id_col not in df.columns:
        raise ValueError(f"{name}: missing id column '{id_col}'.  Available columns: {df.columns}")
    if dict(df.dtypes)[id_col] not in ("int", "bigint", "smallint", "tinyint"):
        raise ValueError(f"{name}: id column '{id_col}' must be an integer type")


def _check_id_values_spark(df: sql.DataFrame, id_col: str, name: str) -> int:
    """
    Check that the id column contains no nulls and no duplicate values
    using a single aggregation over the id column. Returns the row count.
    """
    n_rows, n_non_null, n_distinct = df.select(
            F.count(F.lit(1)),
            F.count(id_col),
            F.countDistinct(id_col),
    ).first()
    if n_non_null < n_rows:
        raise ValueError(f"{name}: nulls are present in the id column '{id_col}'")
    if n_distinct < n_non_null:
        raise ValueError(f"{name}: id column '{id_col}' must be unique")
    return n_rows


def _check_id_spark(df: sql.DataFrame, id_col: str, name: str) -> None:
    """
    Check that the id column is in the dataframe and is a valid id column.
    """
    if check_id_col(df, id_col, name) == 0:
        raise ValueError(f"{name}: empty dataframe")
