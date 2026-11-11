"""Repository pattern implementation for AquaTrade analytical data access."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime

from src.infrastructure.db.duckdb_client import DuckDBClient
from src.domain.models import ForecastResult


class ITradeRepository(ABC):
    """Abstract interface defining trade data repository operations."""

    @abstractmethod
    def save_facts(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def get_time_series(
        self, commodity_desc: str, geography_desc: str = "World", metric: str = "export_vlu"
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_top_commodities(self, n: int = 10) -> List[str]:
        pass

    @abstractmethod
    def get_top_trading_partners(self, n: int = 10) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_forecasts(self, results: List[ForecastResult]) -> None:
        pass

    @abstractmethod
    def get_forecasts(self, commodity_desc: Optional[str] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_summary_stats(self) -> Dict[str, Any]:
        pass


class DuckDBTradeRepository(ITradeRepository):
    """DuckDB implementation of the Trade repository."""

    def __init__(self, db_client: DuckDBClient):
        self.client = db_client

    def save_facts(self, df: pd.DataFrame) -> None:
        """Persists transformed fact data into fact_trade table."""
        self.client.save_dataframe(df, "fact_trade", mode="overwrite")
        self.client.initialize_star_schema()

    def get_time_series(
        self, commodity_desc: str, geography_desc: str = "World", metric: str = "export_vlu"
    ) -> pd.DataFrame:
        """Retrieves a clean monthly time series for a specific commodity, geography, and metric."""
        allowed = {
            "import_qty",
            "import_vlu",
            "export_qty",
            "export_vlu",
            "import_unit_price",
            "export_unit_price",
            "trade_balance_vlu",
            "trade_balance_qty",
        }
        if metric not in allowed:
            raise ValueError(f"Unsupported metric: {metric}")
        query = f"""
            SELECT 
                date,
                {metric} AS value
            FROM fact_trade
            WHERE COMMODITY_DESC = ?
              AND GEOGRAPHY_DESC = ?
              AND date IS NOT NULL
            ORDER BY date ASC;
        """
        df = self.client.execute_query(query, [commodity_desc, geography_desc])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            # Ensure full monthly frequency (preserve missing observations for explicit validation)
            full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="MS")
            df = df.reindex(full_idx)
            df.index.name = "date"
            df = df.reset_index()
        return df

    def get_top_commodities(self, n: int = 10) -> List[str]:
        """Returns top N commodities ranked by combined trade value."""
        query = """
            SELECT 
                COMMODITY_DESC,
                SUM(COALESCE(export_vlu, 0) + COALESCE(import_vlu, 0)) AS total_val
            FROM fact_trade
            WHERE COMMODITY_DESC IS NOT NULL
            GROUP BY COMMODITY_DESC
            ORDER BY total_val DESC
            LIMIT ?;
        """
        df = self.client.execute_query(query, [n])
        return df["COMMODITY_DESC"].tolist() if not df.empty else []

    def get_top_trading_partners(self, n: int = 10) -> pd.DataFrame:
        """Returns top N trading partner countries by trade volume in USD."""
        query = "SELECT * FROM v_top_trading_partners LIMIT ?;"
        return self.client.execute_query(query, [n])

    def save_forecasts(self, results: List[ForecastResult]) -> None:
        """Appends generated forecasts and evaluation metrics to fact_forecasts."""
        records = []
        for res in results:
            for pred in res.predictions:
                records.append(
                    {
                        "series_id": res.series_id,
                        "commodity_desc": res.commodity_desc,
                        "geography_desc": res.geography_desc,
                        "target_metric": res.target_metric,
                        "champion_model": res.champion_model,
                        "horizon_months": res.horizon_months,
                        "forecast_date": pred.date,
                        "predicted_value": float(pred.predicted_value),
                        "lower_bound": float(pred.lower_bound),
                        "upper_bound": float(pred.upper_bound),
                        "model_mae": float(res.evaluation.mae),
                        "model_mape": float(res.evaluation.mape),
                        "model_rmse": float(res.evaluation.rmse),
                        "created_at": datetime.now(),
                    }
                )

        if records:
            df_forecast = pd.DataFrame(records)
            self.client.save_dataframe(df_forecast, "fact_forecasts", mode="append")

    def get_forecasts(self, commodity_desc: Optional[str] = None) -> pd.DataFrame:
        """Retrieves stored forecast predictions."""
        if commodity_desc:
            query = (
                "SELECT * FROM fact_forecasts WHERE commodity_desc = ? ORDER BY forecast_date ASC"
            )
            return self.client.execute_query(query, [commodity_desc])
        return self.client.execute_query("SELECT * FROM fact_forecasts ORDER BY forecast_date ASC")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates global summary figures from DuckDB."""
        query = """
            SELECT
                COUNT(*) AS total_records,
                MIN(date) AS min_date,
                MAX(date) AS max_date,
                SUM(export_vlu) AS total_export_usd,
                SUM(import_vlu) AS total_import_usd,
                COUNT(DISTINCT COMMODITY_DESC) AS total_commodities,
                COUNT(DISTINCT GEOGRAPHY_DESC) AS total_geographies
            FROM fact_trade;
        """
        df = self.client.execute_query(query)
        if not df.empty:
            row = df.iloc[0].to_dict()
            return row
        return {}
