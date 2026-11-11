"""Unit tests for FeatureEngineer, candidate model fitting, and ForecastService."""

import pytest
import pandas as pd

from src.use_cases.forecast_service import FeatureEngineer, ForecastService
from src.domain.exceptions import InsufficientDataError


def test_feature_engineering_lags_and_rolling(synthetic_time_series):
    series = synthetic_time_series.set_index("date")["value"]
    feat_df = FeatureEngineer.create_features(series)

    assert "trend" in feat_df.columns
    assert "month_sin" in feat_df.columns
    assert "month_cos" in feat_df.columns
    assert "quarter" in feat_df.columns
    assert "lag_1" in feat_df.columns
    assert "lag_12" in feat_df.columns
    assert "rolling_mean_3" in feat_df.columns
    assert "rolling_std_6" in feat_df.columns


def test_forecast_service_tournament(synthetic_time_series):
    forecaster = ForecastService(test_split_months=12, min_series_length=24)
    result, champion_obj = forecaster.run_tournament(
        time_series_df=synthetic_time_series,
        series_id="TestSeries__export_vlu__World",
        commodity_desc="TestSalmon",
        target_metric="export_vlu",
        horizon_months=6,
        geography_desc="World",
    )

    assert result.champion_model in [
        "LinearTrend",
        "RidgeRegression",
        "RandomForest",
        "GradientBoosting",
        "Naive",
        "SeasonalNaive",
        "MovingAverage",
    ]
    assert len(result.predictions) == 6
    assert result.evaluation.mae >= 0
    assert result.predictions[0].lower_bound <= result.predictions[0].upper_bound


def test_forecast_service_insufficient_data():
    forecaster = ForecastService(min_series_length=36)
    short_df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="MS"),
            "value": [10.0] * 10,
        }
    )
    with pytest.raises(InsufficientDataError):
        forecaster.run_tournament(
            time_series_df=short_df,
            series_id="Short__val__World",
            commodity_desc="Short",
            target_metric="val",
        )
