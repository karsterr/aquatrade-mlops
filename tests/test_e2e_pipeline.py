"""End-to-end integration tests for AquaTrade MLOps pipeline."""

import pytest
import pandas as pd
from pathlib import Path

from src.config import AppConfig
from src.use_cases.pipeline_orchestrator import AquaTradePipeline


def test_pipeline_e2e_execution(temp_dir, sample_raw_df):
    # Prepare temporary CSV and DuckDB paths
    csv_path = temp_dir / "sample_raw.csv"
    sample_raw_df.to_csv(csv_path, index=False)

    db_path = temp_dir / "test_aqua.duckdb"

    cfg = AppConfig(
        base_dir=temp_dir,
        raw_data_path=csv_path,
        processed_db_path=db_path,
        models_dir=temp_dir / "models",
        experiments_dir=temp_dir / "experiments",
        default_top_n_commodities=2,
        default_forecast_horizon=3,
        min_series_length_for_forecast=4,  # low threshold for synthetic sample
        default_test_split_months=2,
    )
    cfg.resolve_paths()

    pipeline = AquaTradePipeline(cfg)
    summary = pipeline.run_autopilot()

    assert summary["run_id"].startswith("run_")
    assert summary["summary_stats"]["total_records"] == 32
    assert summary["summary_stats"]["total_commodities"] == 2
    assert summary["quality_report"]["is_valid"] is True
