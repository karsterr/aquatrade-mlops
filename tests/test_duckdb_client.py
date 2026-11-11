"""Unit tests for DuckDBClient and DuckDBTradeRepository."""

import pandas as pd
from datetime import date

from src.infrastructure.db.duckdb_client import DuckDBClient
from src.domain.models import ForecastResult, ForecastPoint, ModelEvaluation


def test_duckdb_client_save_and_query(temp_dir):
    db_path = temp_dir / "test.duckdb"
    client = DuckDBClient(db_path)

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    client.save_dataframe(df, "test_table")

    res = client.execute_query("SELECT COUNT(*) AS total FROM test_table")
    assert res["total"][0] == 3


def test_repository_save_and_summary_stats(test_repository):
    stats = test_repository.get_summary_stats()
    assert stats["total_records"] > 0
    assert stats["total_commodities"] == 2
    assert stats["total_geographies"] == 2


def test_repository_get_time_series(test_repository):
    ts = test_repository.get_time_series(
        commodity_desc="Atlantic Salmon", geography_desc="World", metric="export_vlu"
    )
    assert not ts.empty
    assert "date" in ts.columns
    assert "value" in ts.columns


def test_repository_get_top_commodities(test_repository):
    top_c = test_repository.get_top_commodities(n=5)
    assert len(top_c) == 2
    assert "Atlantic Salmon" in top_c
    assert "Trout" in top_c


def test_repository_save_and_get_forecasts(test_repository):
    eval_m = ModelEvaluation("RidgeRegression", 5.0, 7.0, 0.05, 0.04, 0.9, 10)
    point = ForecastPoint(date(2025, 1, 1), 1000.0, 900.0, 1100.0)
    result = ForecastResult(
        series_id="Atlantic Salmon__export_vlu__World",
        commodity_desc="Atlantic Salmon",
        geography_desc="World",
        target_metric="export_vlu",
        champion_model="RidgeRegression",
        horizon_months=1,
        evaluation=eval_m,
        predictions=[point],
    )

    test_repository.save_forecasts([result])
    df_fc = test_repository.get_forecasts(commodity_desc="Atlantic Salmon")
    assert not df_fc.empty
    assert len(df_fc) == 1
    assert df_fc["champion_model"].iloc[0] == "RidgeRegression"
