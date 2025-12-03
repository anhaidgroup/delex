"""
Pytest configuration and shared fixtures.
"""
import pytest
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator
import numpy as np
import pandas as pd

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


@pytest.fixture(scope="session")
def spark_session():
    """
    Create a Spark session for testing.
    Uses local mode with minimal resources for fast tests.
    """
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder \
            .appName("delex-tests") \
            .master("local[2]") \
            .config("spark.sql.shuffle.partitions", "2") \
            .config("spark.driver.memory", "1g") \
            .config("spark.executor.memory", "1g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()

        yield spark

        spark.stop()
    except ImportError:
        pytest.skip("PySpark not available")


@pytest.fixture
def sample_spark_dataframe(spark_session):
    """Create a sample Spark DataFrame for testing."""
    from pyspark.sql import Row

    data = [
        Row(id=1, name="Alice", value=10.5),
        Row(id=2, name="Bob", value=20.3),
        Row(id=3, name="Charlie", value=30.1),
    ]

    return spark_session.createDataFrame(data)


@pytest.fixture(autouse=True)
def reset_state():
    """
    Fixture that runs before each test to reset any global state.
    Override this in specific test modules if needed.
    """
    yield
    # Add any cleanup logic here if needed
