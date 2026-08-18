"""DuckDB analytical database client and schema manager."""

from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import duckdb
import pandas as pd

from src.domain.exceptions import RepositoryError


class DuckDBClient:
    """Manages connection, schema creation, data insertion, and analytical queries in DuckDB."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Returns a connection to the DuckDB database."""
        try:
            return duckdb.connect(str(self.db_path))
        except Exception as e:
            raise RepositoryError(f"Failed to connect to DuckDB at {self.db_path}: {e}") from e

    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """Executes a SQL query and returns the result as a pandas DataFrame."""
        with self.get_connection() as conn:
            try:
                if params:
                    return conn.execute(query, params).fetchdf()
                return conn.execute(query).fetchdf()
            except Exception as e:
                raise RepositoryError(f"Query execution failed: {query} | Error: {e}") from e

    def execute_statement(self, sql: str, params: Optional[List[Any]] = None) -> None:
        """Executes a SQL DDL/DML statement without returning results."""
        with self.get_connection() as conn:
            try:
                if params:
                    conn.execute(sql, params)
                else:
                    conn.execute(sql)
            except Exception as e:
                raise RepositoryError(f"Statement execution failed: {sql} | Error: {e}") from e

    def save_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        mode: str = "overwrite"
    ) -> None:
        """Saves a pandas DataFrame to DuckDB table."""
        with self.get_connection() as conn:
            try:
                if mode == "overwrite":
                    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
                elif mode == "append":
                    # Check if table exists
                    table_exists = conn.execute(
                        f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
                    ).fetchone()[0] > 0

                    if table_exists:
                        conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
                    else:
                        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
                else:
                    raise ValueError(f"Unsupported mode: {mode}. Use 'overwrite' or 'append'.")
            except Exception as e:
                raise RepositoryError(f"Failed to save DataFrame into table '{table_name}': {e}") from e

    def initialize_star_schema(self) -> None:
        """Builds dimensional views and summary aggregates if fact_trade exists."""
        with self.get_connection() as conn:
            try:
                # Dim Commodity View
                conn.execute("""
                    CREATE OR REPLACE VIEW dim_commodity AS
                    SELECT DISTINCT
                        HS_CODE AS hs_code,
                        COMMODITY_DESC AS commodity_desc
                    FROM fact_trade
                    WHERE COMMODITY_DESC IS NOT NULL;
                """)

                # Dim Geography View
                conn.execute("""
                    CREATE OR REPLACE VIEW dim_geography AS
                    SELECT DISTINCT
                        GEOGRAPHY_CODE AS geography_code,
                        GEOGRAPHY_DESC AS geography_desc
                    FROM fact_trade
                    WHERE GEOGRAPHY_DESC IS NOT NULL;
                """)

                # Monthly Commodity Summary View
                conn.execute("""
                    CREATE OR REPLACE VIEW v_monthly_commodity_summary AS
                    SELECT
                        date,
                        COMMODITY_DESC AS commodity_desc,
                        SUM(import_qty) AS total_import_qty,
                        SUM(import_vlu) AS total_import_vlu,
                        SUM(export_qty) AS total_export_qty,
                        SUM(export_vlu) AS total_export_vlu,
                        CASE 
                            WHEN SUM(import_qty) > 0 THEN SUM(import_vlu) / SUM(import_qty)
                            ELSE NULL 
                        END AS weighted_import_price,
                        CASE 
                            WHEN SUM(export_qty) > 0 THEN SUM(export_vlu) / SUM(export_qty)
                            ELSE NULL 
                        END AS weighted_export_price
                    FROM fact_trade
                    GROUP BY date, COMMODITY_DESC;
                """)

                # Top Partner View
                conn.execute("""
                    CREATE OR REPLACE VIEW v_top_trading_partners AS
                    SELECT
                        GEOGRAPHY_DESC AS geography_desc,
                        SUM(export_vlu) AS total_export_value,
                        SUM(import_vlu) AS total_import_value,
                        SUM(COALESCE(export_vlu, 0) + COALESCE(import_vlu, 0)) AS total_trade_volume_usd
                    FROM fact_trade
                    WHERE GEOGRAPHY_DESC NOT IN ('World', 'Unknown')
                    GROUP BY GEOGRAPHY_DESC
                    ORDER BY total_trade_volume_usd DESC;
                """)

                # Forecast Table Schema
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fact_forecasts (
                        series_id VARCHAR,
                        commodity_desc VARCHAR,
                        geography_desc VARCHAR,
                        target_metric VARCHAR,
                        champion_model VARCHAR,
                        horizon_months INTEGER,
                        forecast_date DATE,
                        predicted_value DOUBLE,
                        lower_bound DOUBLE,
                        upper_bound DOUBLE,
                        model_mae DOUBLE,
                        model_mape DOUBLE,
                        model_rmse DOUBLE,
                        created_at TIMESTAMP
                    );
                """)
            except Exception as e:
                raise RepositoryError(f"Failed to initialize star schema & analytical views: {e}") from e
