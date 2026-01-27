"""
Pytest configuration and shared fixtures.
"""
import pytest
import sys
import tempfile
import shutil
import os
import uuid
from pathlib import Path
from typing import Generator
import numpy as np
import pandas as pd
import random

# Add the parent directory to the path so we can import delex modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path."""
    return Path(__file__).parent.parent


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test files.
    Automatically cleaned up after test.
    """
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """
    Create a temporary file in a temp directory.
    Automatically cleaned up after test.
    """
    temp_file_path = temp_dir / "test_file.tmp"
    temp_file_path.touch()
    try:
        yield temp_file_path
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()


@pytest.fixture
def sample_numpy_array():
    """Create a sample numpy array for testing."""
    return np.array([1, 2, 3, 4, 5], dtype=np.int32)


@pytest.fixture
def sample_numpy_2d_array():
    """Create a sample 2D numpy array for testing."""
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)


@pytest.fixture
def sample_pandas_dataframe():
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'value': [10.5, 20.3, 30.1, 40.7, 50.9]
    })


@pytest.fixture
def sample_string_list():
    """Create a sample list of strings for testing."""
    return ['apple', 'banana', 'cherry', 'date', 'elderberry']


@pytest.fixture(scope="function")
def spark_session():
    """
    Create a Spark session for testing.
    Uses function scope to ensure each test gets a fresh Spark context
    with a clean file distribution cache, preventing file conflicts.
    This is slower than session scope but necessary to avoid Spark file
    distribution conflicts.
    """
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder \
            .appName("delex-tests") \
            .getOrCreate()

        yield spark

        spark.stop()
    except ImportError:
        pytest.skip("PySpark not available")


@pytest.fixture
def table_a(spark_session):
    """Create table_a (index table) for testing."""
    # List of random words to use for the 'title' column
    words = [
        "apple", "banana", "cherry", "dog", "elephant", "fire", "grape",
        "house", "ink", "jungle", "kite", "lemon", "mountain", "notebook",
        "orange", "parrot", "queen", "river", "sun", "tree", "umbrella",
        "violet", "wolf", "xylophone", "yellow", "zebra"
    ]
    data = [(int(i), random.choice(words), random.choice(words))
            for i in range(1000)]
    a = spark_session.createDataFrame(data, ['_id', 'title', 'description'])
    return a


@pytest.fixture
def table_b(spark_session):
    """Create table_b (search table) for testing."""
    words = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
        "hotel", "india", "juliet", "kilo", "lima", "mango", "november",
        "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
        "victor", "whiskey", "xray", "yankee", "zulu"
    ]
    data = [(int(i), random.choice(words), random.choice(words))
            for i in range(1000)]
    b = spark_session.createDataFrame(data, ['_id', 'title', 'description'])
    return b


@pytest.fixture
def simple_blocking_program():
    """Create a simple blocking program for testing."""
    from delex.lang import BlockingProgram, KeepRule
    from delex.lang.predicate import ExactMatchPredicate

    pred = ExactMatchPredicate(
        index_col='title', search_col='title', invert=False)
    keep_rule = KeepRule(predicates=[pred])
    return BlockingProgram(keep_rules=[keep_rule], drop_rules=[])


@pytest.fixture
def simple_plan_executor(spark_session, table_a, table_b):
    """Create a simple PlanExecutor for testing."""
    from delex.execution.plan_executor import PlanExecutor

    return PlanExecutor(
        index_table=table_a,
        search_table=table_b,
        optimize=False,
        estimate_cost=False
    )


@pytest.fixture(autouse=True)
def unique_temp_files(monkeypatch):
    """
    Monkeypatch mkstemp to ensure unique filenames across tests.
    Spark's file distribution uses filenames as keys, so if two tests create
    files with the same name (possible with mkstemp), Spark will conflict.
    By adding a UUID to each filename, we ensure uniqueness.
    """
    original_mkstemp = tempfile.mkstemp

    def unique_mkstemp(*args, **kwargs):
        """Wrapper around mkstemp that ensures unique filenames."""
        fd, path = original_mkstemp(*args, **kwargs)
        # Add UUID to filename to ensure uniqueness
        path_obj = Path(path)
        unique_id = uuid.uuid4().hex[:8]
        unique_path = (path_obj.parent /
                       f"{path_obj.stem}_{unique_id}{path_obj.suffix}")
        # Rename to unique name (fd stays valid as it's tied to the inode)
        os.rename(path, unique_path)
        return fd, str(unique_path)

    monkeypatch.setattr(tempfile, 'mkstemp', unique_mkstemp)
    yield
    monkeypatch.setattr(tempfile, 'mkstemp', original_mkstemp)


@pytest.fixture(autouse=True)
def reset_state(spark_session):
    """
    Fixture that runs before and after each test to reset any global state.
    Cleans up temporary files and ensures Spark's file cache doesn't conflict.
    """
    # Before test: ensure clean state
    yield
    # After test: cleanup
    # Force Spark to complete any pending operations
    try:
        # Small operation to ensure Spark processes any pending
        # file distributions
        spark_context = spark_session.sparkContext
        _ = spark_context.parallelize([1]).collect()
    except Exception:
        pass
