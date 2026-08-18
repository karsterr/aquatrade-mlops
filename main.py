"""AquaTrade MLOps - Main Execution Entrypoint and CLI."""

import argparse
import sys
import json
from pathlib import Path

from src.config import config, AppConfig
from src.use_cases.pipeline_orchestrator import AquaTradePipeline
from src.mlops.registry import ModelRegistry
from src.mlops.tracker import ExperimentTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="AquaTrade MLOps: End-to-end Aquaculture Trade Data & Forecasting System."
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "etl", "inspect", "forecast", "stats", "models", "runs"],
        help="Pipeline step to execute (default: all / autopilot mode)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=config.default_top_n_commodities,
        help="Number of top commodities to model and forecast (default: 5)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=config.default_forecast_horizon,
        help="Forecasting horizon in months (default: 12)",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default=str(config.raw_data_path),
        help="Path to raw CSV dataset",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(config.processed_db_path),
        help="Path to DuckDB database file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    custom_config = AppConfig(
        raw_data_path=Path(args.raw_path),
        processed_db_path=Path(args.db_path),
        default_forecast_horizon=args.horizon,
        default_top_n_commodities=args.top_n,
    )
    custom_config.resolve_paths()

    pipeline = AquaTradePipeline(custom_config)

    if args.step == "all":
        pipeline.run_autopilot()

    elif args.step == "etl":
        df_transformed = pipeline.run_etl()
        pipeline.run_persistence(df_transformed)
        print(f"ETL completed successfully. Saved {len(df_transformed):,} rows into DuckDB.")

    elif args.step == "inspect":
        df_transformed = pipeline.run_etl()
        pipeline.run_inspection(df_transformed)

    elif args.step == "forecast":
        pipeline.run_forecasting(
            top_n=args.top_n,
            horizon_months=args.horizon,
        )

    elif args.step == "stats":
        stats = pipeline.repository.get_summary_stats()
        pipeline.reporter.print_summary_stats(stats)
        print("\nTop 5 Trading Partners (USD Volume):")
        partners = pipeline.repository.get_top_trading_partners(n=5)
        print(partners.to_string(index=False))

    elif args.step == "models":
        registry = ModelRegistry(custom_config.models_dir)
        models = registry.list_models()
        print("\n" + "=" * 60)
        print("[MODELS] REGISTERED ML MODELS")
        print("=" * 60)
        if not models:
            print("No models registered yet. Run 'python main.py --step forecast' first.")
        else:
            for s_id, data in models.items():
                print(f"Series: {s_id}")
                print(f" - Commodity: {data.get('commodity_desc')}")
                print(f" - Latest Version: {data.get('latest_version')}")
                ver_meta = data.get("versions", {}).get(data.get("latest_version"), {})
                print(f" - Champion Model: {ver_meta.get('champion_model_name')}")
                print(f" - Metrics: {ver_meta.get('metrics')}")
                print("-" * 60)

    elif args.step == "runs":
        tracker = ExperimentTracker(custom_config.experiments_dir)
        runs = tracker.get_runs()
        print("\n" + "=" * 60)
        print("[RUNS] EXPERIMENT RUNS HISTORY")
        print("=" * 60)
        if not runs:
            print("No experiment runs found.")
        else:
            for r in runs:
                print(f"Run ID: {r.get('run_id')} | Status: {r.get('status')} | Start: {r.get('start_time')}")
                print(f" - Experiment: {r.get('experiment_name')}")
                print(f" - Metrics: {r.get('metrics')}")
                print("-" * 60)


if __name__ == "__main__":
    main()
