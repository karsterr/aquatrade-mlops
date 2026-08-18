"""CSV Data Ingestion Driver."""

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from src.domain.exceptions import DataQualityError


class CSVDataLoader:
    """Robust CSV Loader for raw aquaculture trade data."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)

    def load_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Loads data from CSV file into a pandas DataFrame."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Source trade dataset not found at: {self.file_path}")

        try:
            df = pd.read_csv(
                self.file_path,
                nrows=nrows,
                dtype={
                    "SOURCE_ID": "int64",
                    "HS_CODE": "float64",
                    "COMMODITY_DESC": "str",
                    "GEOGRAPHY_CODE": "int64",
                    "GEOGRAPHY_DESC": "str",
                    "ATTRIBUTE_DESC": "str",
                    "UNIT_DESC": "str",
                    "YEAR_ID": "int64",
                    "TIMEPERIOD_ID": "int64",
                    "AMOUNT": "float64",
                },
                low_memory=False,
            )
            return df
        except Exception as e:
            raise DataQualityError(f"Failed to parse CSV trade dataset at '{self.file_path}': {e}") from e
