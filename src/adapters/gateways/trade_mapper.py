"""Data mapper gateway converting DataFrames and dictionaries to Domain entities."""

from typing import List, Generator, Optional
import pandas as pd
from datetime import datetime

from src.domain.models import TradeRecord
from src.domain.exceptions import DomainValidationError


class TradeMapper:
    """Maps tabular data rows to TradeRecord domain entities."""

    @staticmethod
    def from_dataframe_row(row: pd.Series) -> TradeRecord:
        """Converts a single DataFrame series into a TradeRecord entity."""
        raw_date = row.get("date")
        if isinstance(raw_date, pd.Timestamp):
            record_date = raw_date.date()
        elif isinstance(raw_date, datetime):
            record_date = raw_date.date()
        else:
            record_date = pd.to_datetime(raw_date).date()

        import_qty = float(row["import_qty"]) if pd.notna(row.get("import_qty")) else None
        import_vlu = float(row["import_vlu"]) if pd.notna(row.get("import_vlu")) else None
        export_qty = float(row["export_qty"]) if pd.notna(row.get("export_qty")) else None
        export_vlu = float(row["export_vlu"]) if pd.notna(row.get("export_vlu")) else None
        hs_code = float(row["HS_CODE"]) if pd.notna(row.get("HS_CODE")) else None

        return TradeRecord(
            date=record_date,
            geography_code=int(row["GEOGRAPHY_CODE"]),
            geography_desc=str(row["GEOGRAPHY_DESC"]),
            commodity_desc=str(row["COMMODITY_DESC"]),
            hs_code=hs_code,
            import_qty=import_qty,
            import_vlu=import_vlu,
            export_qty=export_qty,
            export_vlu=export_vlu,
        )

    @staticmethod
    def from_dataframe(df: pd.DataFrame, limit: Optional[int] = None) -> List[TradeRecord]:
        """Converts a DataFrame into a list of TradeRecord domain entities with error resilience."""
        target_df = df.head(limit) if limit is not None else df
        records: List[TradeRecord] = []
        for _, row in target_df.iterrows():
            try:
                record = TradeMapper.from_dataframe_row(row)
                records.append(record)
            except (DomainValidationError, ValueError, TypeError):
                continue
        return records

    @staticmethod
    def iter_dataframe(df: pd.DataFrame) -> Generator[TradeRecord, None, None]:
        """Memory-efficient generator yielding TradeRecord entities."""
        for _, row in df.iterrows():
            try:
                yield TradeMapper.from_dataframe_row(row)
            except (DomainValidationError, ValueError, TypeError):
                continue
