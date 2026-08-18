"""Data inspection and statistical profiling presenter."""

import pandas as pd
from typing import Dict, Any

from src.domain.validation import DataValidator
from src.domain.models import DataQualityReport


class DataInspector:
    """Profiles and presents dataset statistics, schema information, and quality reports."""

    def inspect(self, df: pd.DataFrame) -> DataQualityReport:
        """Runs quality analysis and prints a detailed summary."""
        print("\n" + "=" * 60)
        print("[REPORT] AQUATRADE DATASET QUALITY & PROFILE REPORT")
        print("=" * 60)
        print(f"Total Rows:     {len(df):,}")
        print(f"Total Columns:  {len(df.columns)}")
        print(f"Columns:        {', '.join(df.columns.tolist())}")

        if "date" in df.columns and not df.empty:
            min_date = pd.to_datetime(df["date"]).min()
            max_date = pd.to_datetime(df["date"]).max()
            print(f"Temporal Span:  {min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}")

        print("\nMissing Values Breakdown:")
        missing = df.isnull().sum()
        for col, count in missing.items():
            pct = (count / len(df)) * 100 if len(df) > 0 else 0
            print(f" - {col:<20} : {count:>8,} missing ({pct:>5.1f}%)")

        print("\nNumeric Distribution Summary:")
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns
        if len(num_cols) > 0:
            print(df[num_cols].describe().round(2).T.to_string())

        report = DataValidator.generate_quality_report(df)
        if report.warnings:
            print("\n[WARNING] Data Quality Warnings:")
            for w in report.warnings:
                print(f"  * {w}")
        else:
            print("\n[OK] All data quality checks passed successfully.")

        print("=" * 60 + "\n")
        return report
