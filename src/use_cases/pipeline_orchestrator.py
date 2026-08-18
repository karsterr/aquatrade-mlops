"""End-to-End Pipeline Orchestrator coordinating Ingestion, ETL, Validation, DuckDB, Forecasting, and MLOps."""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from src.config import AppConfig, config as default_config
from src.infrastructure.data_loaders.csv_loader import CSVDataLoader
from src.infrastructure.db.duckdb_client import DuckDBClient
from src.infrastructure.repositories.trade_repository import DuckDBTradeRepository
from src.use_cases.etl_pipeline import ETLService
from src.use_cases.forecast_service import ForecastService
from src.adapters.presenters.data_inspector import DataInspector
from src.adapters.presenters.console_reporter import ConsoleReporter
from src.mlops.tracker import ExperimentTracker
from src.mlops.registry import ModelRegistry
from src.mlops.monitoring import DataDriftMonitor
from src.domain.models import ForecastResult, DataQualityReport


class AquaTradePipeline:
    """Orchestrates the entire AquaTrade MLOps lifecycle."""

    def __init__(self, cfg: Optional[AppConfig] = None):
        self.config = cfg or default_config
        self.config.resolve_paths()

        # Infrastructure
        self.loader = CSVDataLoader(self.config.raw_data_path)
        self.db_client = DuckDBClient(self.config.processed_db_path)
        self.repository = DuckDBTradeRepository(self.db_client)

        # Services & Presenters
        self.etl = ETLService()
        self.inspector = DataInspector()
        self.reporter = ConsoleReporter()
        self.forecaster = ForecastService(
            test_split_months=self.config.default_test_split_months,
            min_series_length=self.config.min_series_length_for_forecast,
            confidence_level=self.config.confidence_level,
        )

        # MLOps
        self.tracker = ExperimentTracker(self.config.experiments_dir)
        self.registry = ModelRegistry(self.config.models_dir)
        self.drift_monitor = DataDriftMonitor()

    def run_etl(self) -> pd.DataFrame:
        """Loads raw CSV and executes ETL transformation."""
        self.reporter.print_step(1, f"Ingesting raw dataset from {self.config.raw_data_path}")
        df_raw = self.loader.load_data()
        print(f"Loaded {len(df_raw):,} raw records.")

        self.reporter.print_step(2, "Executing ETL transformations, metric calculations, and pivoting")
        df_transformed = self.etl.transform(df_raw)
        print(f"Transformed into {len(df_transformed):,} monthly aggregated trade facts.")
        return df_transformed

    def run_inspection(self, df_transformed: pd.DataFrame) -> DataQualityReport:
        """Profiles and validates data quality."""
        self.reporter.print_step(3, "Profiling dataset and validating business invariants")
        report = self.inspector.inspect(df_transformed)
        return report

    def run_persistence(self, df_transformed: pd.DataFrame) -> Dict[str, Any]:
        """Saves facts to DuckDB star schema and retrieves summary stats."""
        self.reporter.print_step(4, f"Persisting fact_trade into DuckDB ({self.config.processed_db_path.name})")
        self.repository.save_facts(df_transformed)

        summary_stats = self.repository.get_summary_stats()
        self.reporter.print_summary_stats(summary_stats)
        return summary_stats

    def run_forecasting(
        self,
        top_n: int = 5,
        horizon_months: int = 12,
        target_metrics: Optional[List[str]] = None,
    ) -> List[ForecastResult]:
        """Runs model tournament and generates multi-step forecasts for top commodities."""
        self.reporter.print_step(5, f"Running Time-Series Forecasting Tournament (Top {top_n} Commodities)")
        top_commodities = self.repository.get_top_commodities(n=top_n)
        metrics = target_metrics or ["export_vlu", "import_vlu"]

        all_results: List[ForecastResult] = []

        for comm in top_commodities:
            for metric in metrics:
                series_id = f"{comm}__{metric}__World"
                try:
                    ts_df = self.repository.get_time_series(
                        commodity_desc=comm,
                        geography_desc="World",
                        metric=metric
                    )

                    if len(ts_df) >= self.config.min_series_length_for_forecast:
                        result, champion_model_obj = self.forecaster.run_tournament(
                            time_series_df=ts_df,
                            series_id=series_id,
                            commodity_desc=comm,
                            target_metric=metric,
                            horizon_months=horizon_months,
                            geography_desc="World",
                        )
                        all_results.append(result)

                        # Register model in Model Registry
                        self.registry.register_model(
                            model_object=champion_model_obj,
                            series_id=series_id,
                            commodity_desc=comm,
                            target_metric=metric,
                            champion_model_name=result.champion_model,
                            metrics=result.evaluation.to_dict(),
                            version="v1.0.0",
                        )
                except Exception as e:
                    print(f"Skipping {series_id}: {e}")

        # Persist forecasts to DuckDB
        if all_results:
            self.repository.save_forecasts(all_results)
            self.reporter.print_leaderboard(all_results)
            # Show detailed forecast for top champion
            self.reporter.print_forecast_table(all_results[0], limit=horizon_months)

        return all_results

    def run_drift_check(self, df_transformed: pd.DataFrame) -> Any:
        """Compares historical baseline vs recent split to detect distribution shifts."""
        self.reporter.print_step(6, "Evaluating Data Distribution & Drift (PSI)")
        split_date = pd.to_datetime("2010-01-01")
        baseline_df = df_transformed[df_transformed["date"] < split_date]
        target_df = df_transformed[df_transformed["date"] >= split_date]

        drift_report = self.drift_monitor.evaluate_drift(baseline_df, target_df)
        print(f"Overall Data Drift Status: {drift_report.overall_status}")
        for m in drift_report.metrics:
            print(f" - {m.feature_name:<20}: PSI={m.psi_score:<6} ({m.status}) | Baseline Mean: {m.baseline_mean:>10.2f} -> Target Mean: {m.target_mean:>10.2f} ({m.mean_change_pct:+.1f}%)")
        return drift_report

    def run_autopilot(self) -> Dict[str, Any]:
        """Executes full pipeline seamlessly from end-to-end with MLOps tracking."""
        start_time = time.time()
        self.reporter.print_banner("AquaTrade MLOps - Automated End-to-End Execution")

        run_id = self.tracker.start_run(
            experiment_name="Aquaculture_Trade_Full_Pipeline",
            tags={"mode": "autopilot", "version": "1.0.0"}
        )
        self.tracker.log_param("forecast_horizon", self.config.default_forecast_horizon)
        self.tracker.log_param("top_n_commodities", self.config.default_top_n_commodities)

        try:
            # Step 1 & 2: ETL
            df_transformed = self.run_etl()
            self.tracker.log_dataset_metadata(df_transformed, name="transformed_fact_trade")

            # Step 3: Inspection
            quality_report = self.run_inspection(df_transformed)
            self.tracker.log_metric("total_records", quality_report.total_records)
            self.tracker.log_metric("outlier_count", quality_report.outlier_count)

            # Step 4: DuckDB Storage
            summary_stats = self.run_persistence(df_transformed)

            # Step 5: Forecasting Tournament
            forecast_results = self.run_forecasting(
                top_n=self.config.default_top_n_commodities,
                horizon_months=self.config.default_forecast_horizon,
                target_metrics=["export_vlu", "import_vlu"],
            )
            if forecast_results:
                avg_mape = float(sum(r.evaluation.mape for r in forecast_results) / len(forecast_results))
                self.tracker.log_metric("avg_champion_mape", avg_mape)
                self.tracker.log_metric("total_forecasted_series", len(forecast_results))

            # Step 6: Data Drift
            drift_report = self.run_drift_check(df_transformed)
            self.tracker.log_param("drift_status", drift_report.overall_status)

            elapsed_sec = round(time.time() - start_time, 2)
            self.tracker.log_metric("pipeline_runtime_sec", elapsed_sec)

            completed_run = self.tracker.end_run(status="SUCCESS")

            self.reporter.print_banner(f"Pipeline Completed Successfully in {elapsed_sec}s | Run ID: {run_id}")

            return {
                "run_id": run_id,
                "elapsed_sec": elapsed_sec,
                "summary_stats": summary_stats,
                "total_forecasts": len(forecast_results),
                "quality_report": quality_report.to_dict(),
            }

        except Exception as e:
            self.tracker.end_run(status=f"FAILED: {e}")
            raise
