"""Forecast-origin and numerical regression tests using synthetic observations."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from src.use_cases.forecast_service import ForecastService
from src.domain.exceptions import InsufficientDataError, ModelTrainingError


class LagOne(RegressorMixin, BaseEstimator):
    seen = []

    def fit(self, X, y):
        return self

    def predict(self, X):
        type(self).seen.extend(X["lag_1"].tolist())
        return X["lag_1"].to_numpy()


def run(service, frame, horizon=6):
    return service.run_tournament(frame, "synthetic", "fish", "export_vlu", horizon)


def test_recursive_backtest_never_reads_future_targets(synthetic_time_series, monkeypatch):
    service = ForecastService(test_split_months=12)
    monkeypatch.setattr(service, "_get_candidate_models", lambda: {"LagOne": LagOne()})
    LagOne.seen = []
    result, _ = run(service, synthetic_time_series)
    # Origins at 36 and 42. Every step must reuse predictions, never test observations.
    assert LagOne.seen[:6] == [synthetic_time_series.value.iloc[35]] * 6
    assert LagOne.seen[6:12] == [synthetic_time_series.value.iloc[41]] * 6
    assert {e.test_samples for e in result.candidate_evaluations} == {12}
    assert len(result.predictions) == 6


def test_future_mutation_cannot_change_fixed_origin_predictions(synthetic_time_series):
    history = synthetic_time_series.set_index("date").value.iloc[:36]
    first = ForecastService._recursive_forecast(LagOne(), history, 6)
    modified = synthetic_time_series.copy()
    modified.loc[36:, "value"] = 1e9
    second = ForecastService._recursive_forecast(
        LagOne(), modified.set_index("date").value.iloc[:36], 6
    )
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "nan", "negative", "infinite", "midmonth"]
)
def test_invalid_monthly_series_rejected(synthetic_time_series, damage):
    frame = synthetic_time_series.copy()
    if damage == "missing":
        frame = frame.drop(index=20)
    elif damage == "duplicate":
        frame.loc[20, "date"] = frame.loc[19, "date"]
    elif damage == "midmonth":
        frame.loc[20, "date"] += pd.Timedelta(days=1)
    else:
        frame.loc[20, "value"] = {"nan": np.nan, "negative": -1, "infinite": np.inf}[damage]
    with pytest.raises(ValueError):
        run(ForecastService(), frame)


@pytest.mark.parametrize("horizon", [0, -1])
def test_invalid_horizon(synthetic_time_series, horizon):
    with pytest.raises(ValueError):
        run(ForecastService(), synthetic_time_series, horizon)


def test_full_horizon_required(synthetic_time_series):
    with pytest.raises(InsufficientDataError):
        run(ForecastService(test_split_months=3), synthetic_time_series, 6)


def test_metrics_zero_and_constant_targets():
    service = ForecastService()
    zero = service._calculate_metrics(np.zeros(3), np.ones(3), "test")
    assert zero.mae == zero.rmse == 1
    assert np.isnan(zero.wape) and np.isnan(zero.mape) and np.isnan(zero.r2)
    mixed = service._calculate_metrics(np.array([0, 2]), np.array([1, 1]), "test")
    assert mixed.mape == 0.5 and mixed.wape == 1
    with pytest.raises(ValueError):
        service._calculate_metrics([np.nan], [0], "test")


def test_baseline_selection_and_determinism(synthetic_time_series):
    frame = synthetic_time_series.copy()
    frame["value"] = 10.0
    first, _ = run(ForecastService(), frame)
    second, _ = run(ForecastService(), frame)
    assert first.champion_model == "Naive"
    assert first.evaluation.to_dict()["mae"] == second.evaluation.mae == 0
    assert [p.predicted_value for p in first.predictions] == [10.0] * 6


def test_candidate_failure_is_visible(synthetic_time_series, monkeypatch):
    service = ForecastService()
    monkeypatch.setattr(
        service, "_fit", lambda *args: (_ for _ in ()).throw(RuntimeError("broken"))
    )
    with pytest.raises(ModelTrainingError):
        run(service, synthetic_time_series)


@pytest.mark.parametrize(
    "strategy,expected",
    [
        ("Naive", [24] * 14),
        ("MovingAverage", [21.5] * 14),
        ("SeasonalNaive", list(range(13, 25)) + [13, 14]),
    ],
)
def test_baselines(strategy, expected):
    actual = ForecastService()._forecast_baseline(pd.Series(range(1, 25)), 14, strategy)
    np.testing.assert_allclose(actual, expected)


def test_seeded_estimator_and_negative_recursion(synthetic_time_series):
    service = ForecastService()
    history = synthetic_time_series.set_index("date").value.iloc[:36]
    estimator = service._get_candidate_models()["RandomForest"]
    one = service._recursive_forecast(service._fit(estimator, history), history, 3)
    two = service._recursive_forecast(service._fit(estimator, history), history, 3)
    np.testing.assert_array_equal(one, two)

    class Negative:
        def predict(self, features):
            return [-1.0]

    np.testing.assert_array_equal(service._recursive_forecast(Negative(), history, 3), np.zeros(3))


def test_long_horizons_remain_full_and_disjoint(monkeypatch):
    frame = pd.DataFrame(
        {"date": pd.date_range("2000-01-01", periods=72, freq="MS"), "value": np.arange(72.0)}
    )
    service = ForecastService(test_split_months=36)
    monkeypatch.setattr(service, "_get_candidate_models", lambda: {})
    result, _ = run(service, frame, 18)
    assert len(result.predictions) == 18
    assert result.predictions[0].date == pd.Timestamp("2006-01-01").date()
    assert result.predictions[-1].date == pd.Timestamp("2007-06-01").date()
    assert {e.test_samples for e in result.candidate_evaluations} == {36}
    assert result.evaluation_protocol["origins"] == ["2002-12-01T00:00:00", "2004-06-01T00:00:00"]
