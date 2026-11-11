"""ETL Transformation and Data Cleansing Use Case."""

import pandas as pd
import numpy as np

from src.domain.validation import DataValidator
from src.domain.exceptions import DataQualityError


class ETLService:
    """Transforms raw aquaculture trade records into a structured, pivoted analytical dataset."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans, standardizes, pivots, and enriches raw trade records."""
        DataValidator.validate_raw_schema(df)

        df_clean = df.copy()
        amounts = pd.to_numeric(df_clean["AMOUNT"], errors="raise")
        if (amounts < 0).any() or np.isinf(amounts).any():
            raise DataQualityError("Raw trade amounts must be finite and nonnegative.")
        df_clean["AMOUNT"] = amounts

        # Build ISO monthly date from YEAR_ID and TIMEPERIOD_ID
        year_str = df_clean["YEAR_ID"].astype(str)
        month_str = df_clean["TIMEPERIOD_ID"].astype(str).str.zfill(2)
        date_str = year_str + "-" + month_str + "-01"
        df_clean["date"] = pd.to_datetime(date_str, errors="coerce")

        # Drop any records with unparseable dates
        df_clean = df_clean.dropna(subset=["date"])

        # Pivot trade attributes into dedicated columns
        index_cols = [
            "date",
            "GEOGRAPHY_CODE",
            "GEOGRAPHY_DESC",
            "HS_CODE",
            "COMMODITY_DESC",
        ]

        df_pivot = (
            df_clean.groupby(index_cols + ["ATTRIBUTE_DESC"], dropna=False)["AMOUNT"]
            .sum(min_count=1)
            .unstack("ATTRIBUTE_DESC")
            .reset_index()
        )

        df_pivot.columns.name = None

        # Standardize attribute column names
        rename_map = {
            "US Import, QTY": "import_qty",
            "US Import, VLU": "import_vlu",
            "US Export, QTY": "export_qty",
            "US Export, VLU": "export_vlu",
        }
        df_pivot = df_pivot.rename(columns=rename_map)

        # Ensure all 4 key metric columns exist even if some are absent in raw slice
        for col in ["import_qty", "import_vlu", "export_qty", "export_vlu"]:
            if col not in df_pivot.columns:
                df_pivot[col] = np.nan

        # Reject invalid raw measurements rather than silently clipping them.
        base = ["import_qty", "import_vlu", "export_qty", "export_vlu"]
        if (df_pivot[base] < 0).any().any() or np.isinf(df_pivot[base]).any().any():
            raise DataQualityError("Trade values and quantities must be finite and nonnegative.")

        # Calculate unit prices ($ / unit)
        df_pivot["import_unit_price"] = df_pivot["import_vlu"] / df_pivot["import_qty"].replace(
            {0: np.nan}
        )
        df_pivot["export_unit_price"] = df_pivot["export_vlu"] / df_pivot["export_qty"].replace(
            {0: np.nan}
        )

        # Calculate trade balances
        df_pivot["trade_balance_vlu"] = df_pivot["export_vlu"].fillna(0) - df_pivot[
            "import_vlu"
        ].fillna(0)
        df_pivot["trade_balance_qty"] = df_pivot["export_qty"].fillna(0) - df_pivot[
            "import_qty"
        ].fillna(0)

        # Sort chronologically and by commodity
        df_pivot = df_pivot.sort_values(
            by=["date", "COMMODITY_DESC", "GEOGRAPHY_DESC"]
        ).reset_index(drop=True)

        DataValidator.validate_transformed_schema(df_pivot)
        return df_pivot
