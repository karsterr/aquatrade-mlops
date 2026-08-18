"""Domain validation rules and schema verification for AquaTrade datasets."""

from typing import List, Tuple
import pandas as pd
import numpy as np

from src.domain.models import DataQualityReport
from src.domain.exceptions import DataQualityError


class DataValidator:
    """Validates raw and transformed trade datasets against business rules."""

    REQUIRED_RAW_COLUMNS = [
        "YEAR_ID",
        "TIMEPERIOD_ID",
        "GEOGRAPHY_CODE",
        "GEOGRAPHY_DESC",
        "COMMODITY_DESC",
        "ATTRIBUTE_DESC",
        "AMOUNT",
    ]

    REQUIRED_TRANSFORMED_COLUMNS = [
        "date",
        "GEOGRAPHY_CODE",
        "GEOGRAPHY_DESC",
        "COMMODITY_DESC",
        "import_qty",
        "import_vlu",
        "export_qty",
        "export_vlu",
        "import_unit_price",
        "export_unit_price",
    ]

    @classmethod
    def validate_raw_schema(cls, df: pd.DataFrame) -> None:
        """Checks if all required raw columns exist."""
        missing = [col for col in cls.REQUIRED_RAW_COLUMNS if col not in df.columns]
        if missing:
            raise DataQualityError(f"Raw dataset is missing required columns: {missing}")

    @classmethod
    def validate_transformed_schema(cls, df: pd.DataFrame) -> None:
        """Checks if transformed dataset contains required columns."""
        missing = [col for col in cls.REQUIRED_TRANSFORMED_COLUMNS if col not in df.columns]
        if missing:
            raise DataQualityError(f"Transformed dataset is missing required columns: {missing}")

    @classmethod
    def generate_quality_report(cls, df: pd.DataFrame) -> DataQualityReport:
        """Inspects DataFrame and returns a comprehensive DataQualityReport."""
        cls.validate_transformed_schema(df)

        total_records = len(df)
        warnings: List[str] = []

        if total_records == 0:
            return DataQualityReport(
                total_records=0,
                unique_dates=0,
                min_date=pd.to_datetime("1970-01-01").date(),
                max_date=pd.to_datetime("1970-01-01").date(),
                unique_commodities=0,
                unique_geographies=0,
                missing_import_qty=0,
                missing_import_vlu=0,
                missing_export_qty=0,
                missing_export_vlu=0,
                negative_value_count=0,
                outlier_count=0,
                is_valid=False,
                warnings=["Dataset is empty."],
            )

        unique_dates = int(df["date"].nunique())
        min_date = pd.to_datetime(df["date"]).min().date()
        max_date = pd.to_datetime(df["date"]).max().date()
        unique_commodities = int(df["COMMODITY_DESC"].nunique())
        unique_geographies = int(df["GEOGRAPHY_DESC"].nunique())

        missing_import_qty = int(df["import_qty"].isna().sum())
        missing_import_vlu = int(df["import_vlu"].isna().sum())
        missing_export_qty = int(df["export_qty"].isna().sum())
        missing_export_vlu = int(df["export_vlu"].isna().sum())

        # Check negative values
        num_cols = ["import_qty", "import_vlu", "export_qty", "export_vlu"]
        negative_count = 0
        for col in num_cols:
            if col in df.columns:
                negative_count += int((df[col].dropna() < 0).sum())

        if negative_count > 0:
            warnings.append(f"Found {negative_count} negative trade volume/value records.")

        # Check outliers using IQR on unit prices
        outlier_count = 0
        for price_col in ["import_unit_price", "export_unit_price"]:
            if price_col in df.columns:
                series = df[price_col].dropna()
                if len(series) > 10:
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    upper_bound = q3 + 3.0 * iqr
                    outliers = int((series > upper_bound).sum())
                    outlier_count += outliers
                    if outliers > 0:
                        warnings.append(f"Detected {outliers} extreme outliers in {price_col} (> ${upper_bound:.2f}/unit).")

        is_valid = (negative_count == 0) and (total_records > 0)

        return DataQualityReport(
            total_records=total_records,
            unique_dates=unique_dates,
            min_date=min_date,
            max_date=max_date,
            unique_commodities=unique_commodities,
            unique_geographies=unique_geographies,
            missing_import_qty=missing_import_qty,
            missing_import_vlu=missing_import_vlu,
            missing_export_qty=missing_export_qty,
            missing_export_vlu=missing_export_vlu,
            negative_value_count=negative_count,
            outlier_count=outlier_count,
            is_valid=is_valid,
            warnings=warnings,
        )
