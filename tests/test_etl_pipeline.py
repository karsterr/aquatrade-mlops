"""Unit tests for ETL transformation pipeline and schema validations."""

import pytest
import pandas as pd

from src.use_cases.etl_pipeline import ETLService
from src.domain.exceptions import DataQualityError
from src.domain.validation import DataValidator


def test_etl_transform_structure(sample_raw_df):
    etl = ETLService()
    df_transformed = etl.transform(sample_raw_df)

    assert "date" in df_transformed.columns
    assert "import_qty" in df_transformed.columns
    assert "import_vlu" in df_transformed.columns
    assert "export_qty" in df_transformed.columns
    assert "export_vlu" in df_transformed.columns
    assert "import_unit_price" in df_transformed.columns
    assert "export_unit_price" in df_transformed.columns
    assert "trade_balance_vlu" in df_transformed.columns

    # Verify length
    assert len(df_transformed) == 32  # 8 dates * 2 commodities * 2 geographies


def test_etl_unit_price_calculation(sample_raw_df):
    etl = ETLService()
    df_transformed = etl.transform(sample_raw_df)

    row = df_transformed.iloc[0]
    expected_import_price = row["import_vlu"] / row["import_qty"]
    expected_export_price = row["export_vlu"] / row["export_qty"]

    assert row["import_unit_price"] == pytest.approx(expected_import_price)
    assert row["export_unit_price"] == pytest.approx(expected_export_price)


def test_etl_missing_columns_raises_error():
    invalid_df = pd.DataFrame({"dummy_col": [1, 2, 3]})
    etl = ETLService()
    with pytest.raises(DataQualityError):
        etl.transform(invalid_df)


def test_data_validator_report(sample_transformed_df):
    report = DataValidator.generate_quality_report(sample_transformed_df)
    assert report.total_records == len(sample_transformed_df)
    assert report.is_valid is True
    assert report.negative_value_count == 0
