"""Console reporting and visualization presenter for pipeline, forecasting, and MLOps metrics."""

from typing import List, Dict, Any, Optional
import pandas as pd
from src.domain.models import ForecastResult


class ConsoleReporter:
    """Formats and prints executive summaries, forecast leaderboards, and MLOps metrics."""

    @staticmethod
    def print_banner(title: str) -> None:
        print("\n" + "#" * 70)
        print(f"  {title.upper()}")
        print("#" * 70)

    @staticmethod
    def print_step(step_num: int, title: str) -> None:
        print(f"\n[STEP {step_num}] {title}")
        print("-" * 50)

    @staticmethod
    def print_leaderboard(results: List[ForecastResult]) -> None:
        """Prints a comparison table of champion models across commodities."""
        print("\n" + "=" * 85)
        print("[LEADERBOARD] AQUATRADE TIME-SERIES FORECASTING CHAMPIONS")
        print("=" * 85)
        headers = f"{'Commodity':<32} {'Target':<14} {'Champion Model':<16} {'MAE':>8} {'RMSE':>8} {'MAPE':>7} {'R2':>6}"
        print(headers)
        print("-" * 85)

        for res in results:
            short_comm = (res.commodity_desc[:29] + "...") if len(res.commodity_desc) > 32 else res.commodity_desc
            eval_m = res.evaluation
            print(
                f"{short_comm:<32} "
                f"{res.target_metric:<14} "
                f"{res.champion_model:<16} "
                f"{eval_m.mae:>8.2f} "
                f"{eval_m.rmse:>8.2f} "
                f"{eval_m.mape*100:>6.1f}% "
                f"{eval_m.r2:>6.2f}"
            )
        print("=" * 85)

    @staticmethod
    def print_forecast_table(res: ForecastResult, limit: int = 12) -> None:
        """Prints multi-step forecasts with confidence intervals for a single series."""
        print(f"\n[FORECAST] Multi-Month Horizon: '{res.commodity_desc}' ({res.target_metric})")
        print(f"   Model: {res.champion_model} | Backtest MAE: {res.evaluation.mae:.2f} | MAPE: {res.evaluation.mape*100:.1f}%")
        print(f"   {'Month':<12} {'Forecast Value':>18} {'Lower 95% CI':>18} {'Upper 95% CI':>18}")
        print("   " + "-" * 70)
        for point in res.predictions[:limit]:
            month_str = point.date.strftime("%Y-%m")
            print(f"   {month_str:<12} {point.predicted_value:>18,.2f} {point.lower_bound:>18,.2f} {point.upper_bound:>18,.2f}")
        print()

    @staticmethod
    def print_summary_stats(stats: Dict[str, Any]) -> None:
        """Prints overall DuckDB warehouse statistics."""
        print("\n" + "=" * 60)
        print("[METRICS] DUCKDB ANALYTICAL WAREHOUSE METRICS")
        print("=" * 60)
        for k, v in stats.items():
            key_clean = k.replace("_", " ").title()
            if isinstance(v, float):
                print(f" - {key_clean:<26}: {v:>20,.2f}")
            elif isinstance(v, int):
                print(f" - {key_clean:<26}: {v:>20,}")
            else:
                print(f" - {key_clean:<26}: {str(v):>20}")
        print("=" * 60)
