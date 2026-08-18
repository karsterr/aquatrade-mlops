"""Central configuration for AquaTrade MLOps pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class AppConfig:
    # Root and data paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    raw_data_path: Path = field(default_factory=lambda: Path("data/raw/Aquaculture_Trade_Filtered.csv"))
    processed_db_path: Path = field(default_factory=lambda: Path("data/processed/aquaculture.duckdb"))
    models_dir: Path = field(default_factory=lambda: Path("data/models"))
    experiments_dir: Path = field(default_factory=lambda: Path("data/experiments"))
    reports_dir: Path = field(default_factory=lambda: Path("data/reports"))

    # Table names
    fact_table_name: str = "fact_trade"
    dim_commodity_table_name: str = "dim_commodity"
    dim_geography_table_name: str = "dim_geography"
    fact_forecast_table_name: str = "fact_forecasts"

    # Forecasting configurations
    default_forecast_horizon: int = 12  # months
    default_test_split_months: int = 24  # backtest evaluation window
    min_series_length_for_forecast: int = 36  # minimum monthly points required
    confidence_level: float = 0.95
    candidate_models: List[str] = field(
        default_factory=lambda: [
            "Naive",
            "SeasonalNaive",
            "MovingAverage",
            "LinearTrend",
            "RidgeRegression",
            "RandomForest",
            "GradientBoosting",
        ]
    )

    # Top commodities to automatically train/forecast in autopilot mode
    default_top_n_commodities: int = 5
    default_target_metrics: List[str] = field(
        default_factory=lambda: [
            "export_vlu",
            "export_qty",
            "import_vlu",
            "import_qty",
            "export_unit_price",
            "import_unit_price",
        ]
    )

    def resolve_paths(self):
        """Ensure paths are resolved relative to base_dir if relative."""
        if not self.raw_data_path.is_absolute():
            self.raw_data_path = self.base_dir / self.raw_data_path
        if not self.processed_db_path.is_absolute():
            self.processed_db_path = self.base_dir / self.processed_db_path
        if not self.models_dir.is_absolute():
            self.models_dir = self.base_dir / self.models_dir
        if not self.experiments_dir.is_absolute():
            self.experiments_dir = self.base_dir / self.experiments_dir
        if not self.reports_dir.is_absolute():
            self.reports_dir = self.base_dir / self.reports_dir

        # Create needed directories
        self.processed_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


# Global default configuration instance
config = AppConfig()
config.resolve_paths()
