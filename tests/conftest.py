"""Pytest fixtures and test configurations for AquaTrade MLOps."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.config import AppConfig
from src.infrastructure.db.duckdb_client import DuckDBClient
from src.infrastructure.repositories.trade_repository import DuckDBTradeRepository
from src.use_cases.etl_pipeline import ETLService


@pytest.fixture
def temp_dir():
    """Provides an isolated temporary directory for test artifacts."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Generates synthetic raw trade data matching raw CSV schema."""
    data = []
    dates = [
        (2020, 1), (2020, 2), (2020, 3), (2020, 4),
        (2021, 1), (2021, 2), (2021, 3), (2021, 4),
    ]
    commodities = ["Atlantic Salmon", "Trout"]
    geographies = [(1, "World"), (1220, "Canada")]

    for y, m in dates:
        for c in commodities:
            for g_code, g_desc in geographies:
                # Add 4 attributes
                data.append({"SOURCE_ID": 63, "HS_CODE": 302110000.0, "COMMODITY_DESC": c, "GEOGRAPHY_CODE": g_code, "GEOGRAPHY_DESC": g_desc, "ATTRIBUTE_DESC": "US Import, QTY", "UNIT_DESC": "KG", "YEAR_ID": y, "TIMEPERIOD_ID": m, "AMOUNT": 1000.0 + m * 50})
                data.append({"SOURCE_ID": 63, "HS_CODE": 302110000.0, "COMMODITY_DESC": c, "GEOGRAPHY_CODE": g_code, "GEOGRAPHY_DESC": g_desc, "ATTRIBUTE_DESC": "US Import, VLU", "UNIT_DESC": "$", "YEAR_ID": y, "TIMEPERIOD_ID": m, "AMOUNT": 5000.0 + m * 200})
                data.append({"SOURCE_ID": 63, "HS_CODE": 302110000.0, "COMMODITY_DESC": c, "GEOGRAPHY_CODE": g_code, "GEOGRAPHY_DESC": g_desc, "ATTRIBUTE_DESC": "US Export, QTY", "UNIT_DESC": "KG", "YEAR_ID": y, "TIMEPERIOD_ID": m, "AMOUNT": 800.0 + m * 30})
                data.append({"SOURCE_ID": 63, "HS_CODE": 302110000.0, "COMMODITY_DESC": c, "GEOGRAPHY_CODE": g_code, "GEOGRAPHY_DESC": g_desc, "ATTRIBUTE_DESC": "US Export, VLU", "UNIT_DESC": "$", "YEAR_ID": y, "TIMEPERIOD_ID": m, "AMOUNT": 6400.0 + m * 250})

    return pd.DataFrame(data)


@pytest.fixture
def sample_transformed_df(sample_raw_df) -> pd.DataFrame:
    """Transforms raw sample DataFrame into standard pivoted fact DataFrame."""
    etl = ETLService()
    return etl.transform(sample_raw_df)


@pytest.fixture
def test_duckdb(temp_dir) -> DuckDBClient:
    """Provides a fresh DuckDBClient pointing to a temporary test database."""
    db_file = temp_dir / "test_aquaculture.duckdb"
    return DuckDBClient(db_file)


@pytest.fixture
def test_repository(test_duckdb, sample_transformed_df) -> DuckDBTradeRepository:
    """Provides a DuckDBTradeRepository pre-loaded with sample transformed data."""
    repo = DuckDBTradeRepository(test_duckdb)
    repo.save_facts(sample_transformed_df)
    return repo


@pytest.fixture
def synthetic_time_series() -> pd.DataFrame:
    """Generates 48 months of continuous monthly trade time-series data for testing forecaster."""
    dates = pd.date_range(start="2010-01-01", periods=48, freq="MS")
    # Trend + Seasonality + Noise
    trend = np.linspace(1000, 3000, 48)
    seasonal = 300 * np.sin(2 * np.pi * np.arange(48) / 12)
    noise = np.random.RandomState(42).normal(0, 50, 48)
    values = np.maximum(100.0, trend + seasonal + noise)

    return pd.DataFrame({"date": dates, "value": values})
