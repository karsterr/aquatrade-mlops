"""ML-based time-series forecasting service with automated model selection and backtesting."""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statistics import NormalDist

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.domain.models import ForecastPoint, ModelEvaluation, ForecastResult
from src.domain.exceptions import InsufficientDataError, ModelTrainingError


class FeatureEngineer:
    """Creates lag, rolling, seasonal, and trend features for time-series forecasting."""

    @staticmethod
    def create_features(
        series: pd.Series,
        lags: Tuple[int, ...] = (1, 2, 3, 6, 12),
        rolling_windows: Tuple[int, ...] = (3, 6, 12),
    ) -> pd.DataFrame:
        df = pd.DataFrame({"target": series.values}, index=series.index)

        # Time Trend
        df["trend"] = np.arange(len(df))

        # Calendar / Cyclical Seasonality
        months = df.index.month
        df["month_sin"] = np.sin(2 * np.pi * months / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * months / 12.0)
        df["quarter"] = df.index.quarter

        # Lag Features
        for lag in lags:
            df[f"lag_{lag}"] = df["target"].shift(lag)

        # Rolling Window Statistics
        for w in rolling_windows:
            df[f"rolling_mean_{w}"] = df["target"].shift(1).rolling(window=w, min_periods=1).mean()
            df[f"rolling_std_{w}"] = (
                df["target"].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
            )
            df[f"rolling_min_{w}"] = df["target"].shift(1).rolling(window=w, min_periods=1).min()
            df[f"rolling_max_{w}"] = df["target"].shift(1).rolling(window=w, min_periods=1).max()

        return df


class ForecastService:
    """Manages model training, tournament evaluation, backtesting, and recursive inference."""

    def __init__(
        self,
        test_split_months: int = 24,
        min_series_length: int = 36,
        confidence_level: float = 0.95,
    ):
        if test_split_months < 1 or min_series_length < 1 or not 0 < confidence_level < 1:
            raise ValueError("Invalid evaluation configuration.")
        self.test_split_months = test_split_months
        self.min_series_length = min_series_length
        self.confidence_level = confidence_level

    def _get_candidate_models(self) -> Dict[str, Any]:
        """Returns initialized candidate forecasting estimators."""
        return {
            "LinearTrend": LinearRegression(),
            "RidgeRegression": make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42)),
            "RandomForest": RandomForestRegressor(n_estimators=60, max_depth=6, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=50, max_depth=4, random_state=42
            ),
        }

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str
    ) -> ModelEvaluation:
        """Calculates standard regression and percentage metrics."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.shape != y_pred.shape or not y_true.size:
            raise ValueError("Metrics require matching nonempty arrays.")
        if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
            raise ValueError("Metrics require finite observations and predictions.")
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        denominator = float(np.abs(y_true).sum())
        # Undefined percentages are explicit NaN, never epsilon-adjusted scores.
        wape = float(np.abs(y_true - y_pred).sum() / denominator) if denominator else float("nan")
        nonzero = y_true != 0
        mape = (
            float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])))
            if nonzero.any()
            else float("nan")
        )
        r2 = (
            float(r2_score(y_true, y_pred))
            if len(y_true) > 1 and np.var(y_true) > 0
            else float("nan")
        )

        if not np.isfinite([mae, rmse]).all():
            raise ValueError(
                "Metric arithmetic overflowed; rescale target units before evaluation."
            )

        return ModelEvaluation(
            model_name=model_name,
            mae=mae,
            rmse=rmse,
            mape=mape,
            wape=wape,
            r2=r2,
            test_samples=len(y_true),
        )

    def _forecast_baseline(
        self, train_series: pd.Series, horizon: int, strategy: str = "Naive"
    ) -> np.ndarray:
        """Generates baseline benchmark forecasts."""
        vals = train_series.values
        if len(vals) == 0:
            return np.zeros(horizon)

        if strategy == "Naive":
            return np.full(horizon, vals[-1])
        elif strategy == "SeasonalNaive":
            if len(vals) >= 12:
                last_12 = vals[-12:]
                reps = int(np.ceil(horizon / 12.0))
                return np.tile(last_12, reps)[:horizon]
            return np.full(horizon, vals[-1])
        elif strategy == "MovingAverage":
            k = min(6, len(vals))
            return np.full(horizon, float(np.mean(vals[-k:])))
        return np.full(horizon, vals[-1])

    @staticmethod
    def _fit(estimator, history):
        features = FeatureEngineer.create_features(history).dropna()
        model = clone(estimator)
        model.fit(features.drop(columns="target"), features["target"])
        return model

    @staticmethod
    def _recursive_forecast(model, history, horizon):
        extended = history.copy()
        predictions = []
        for _ in range(horizon):
            extended.loc[extended.index[-1] + pd.offsets.MonthBegin(1)] = np.nan
            features = FeatureEngineer.create_features(extended).drop(columns="target").iloc[-1:]
            value = float(model.predict(features)[0])
            if not np.isfinite(value):
                raise ValueError("Model produced a nonfinite forecast.")
            value = max(0.0, value)
            extended.iloc[-1] = value
            predictions.append(value)
        return np.asarray(predictions)

    def run_tournament(
        self,
        time_series_df: pd.DataFrame,
        series_id: str,
        commodity_desc: str,
        target_metric: str,
        horizon_months: int = 12,
        geography_desc: str = "World",
    ) -> Tuple[ForecastResult, Any]:
        """Conducts a model tournament over backtest split and generates future forecasts."""
        if not isinstance(horizon_months, int) or horizon_months < 1:
            raise ValueError("horizon_months must be a positive integer.")
        if self.test_split_months < horizon_months:
            raise InsufficientDataError(
                "Evaluation window must contain at least one full forecast horizon."
            )
        ts = time_series_df.copy()
        ts["date"] = pd.to_datetime(ts["date"], errors="raise")
        if ts["date"].isna().any() or not ts["date"].dt.is_month_start.all():
            raise ValueError("Dates must be valid month starts.")
        ts = ts.sort_values("date").set_index("date")
        target_series = ts["value"].astype(float)
        expected = pd.date_range(ts.index.min(), ts.index.max(), freq="MS") if len(ts) else []
        if ts.index.has_duplicates or not ts.index.equals(pd.DatetimeIndex(expected)):
            raise ValueError(
                "Series must have unique, consecutive monthly observations; missing months are not zeros."
            )
        if not np.isfinite(target_series).all() or (target_series < 0).any():
            raise ValueError(
                "Targets must be finite and nonnegative; missing values require an explicit data policy."
            )
        if len(ts) < self.min_series_length:
            raise InsufficientDataError(
                f"Series '{series_id}' has {len(ts)} points, minimum is {self.min_series_length}."
            )
        # Right-aligned, nonoverlapping full horizons; never shorten the requested horizon.
        blocks = min(self.test_split_months // horizon_months, (len(ts) - 12) // horizon_months)
        if blocks < 1:
            raise InsufficientDataError(
                "Need at least 12 training months and one full evaluation horizon."
            )
        origins = list(range(len(ts) - blocks * horizon_months, len(ts), horizon_months))
        candidate_models = self._get_candidate_models()
        baselines = ["Naive", "SeasonalNaive", "MovingAverage"]
        names = baselines + (list(candidate_models) if origins[0] - 12 > 10 else [])
        actuals = np.concatenate(
            [target_series.iloc[o : o + horizon_months].values for o in origins]
        )
        candidate_evals = []
        for name in names:
            predictions = []
            for origin in origins:
                history = target_series.iloc[:origin]
                try:
                    if name in baselines:
                        pred = self._forecast_baseline(history, horizon_months, name)
                    else:
                        model = self._fit(candidate_models[name], history)
                        pred = self._recursive_forecast(model, history, horizon_months)
                    predictions.extend(pred)
                except Exception as exc:
                    raise ModelTrainingError(
                        f"Candidate {name} failed at origin {history.index[-1]}."
                    ) from exc
            candidate_evals.append(self._calculate_metrics(actuals, np.asarray(predictions), name))
        # WAPE and MAE give identical rankings within a series; MAE also handles all-zero targets.
        candidate_evals.sort(key=lambda evaluation: evaluation.mae)
        champion_eval = candidate_evals[0]
        champion_name = champion_eval.model_name
        future_dates = pd.date_range(
            target_series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS"
        ).date
        if champion_name in baselines:
            future_preds = self._forecast_baseline(target_series, horizon_months, champion_name)
            champion_model_obj = {"type": "baseline", "strategy": champion_name}
        else:
            champion_model_obj = self._fit(candidate_models[champion_name], target_series)
            future_preds = self._recursive_forecast(
                champion_model_obj, target_series, horizon_months
            )

        # Heuristic uncertainty bands: nominal multiplier is not calibrated coverage.
        z = NormalDist().inv_cdf((1 + self.confidence_level) / 2)
        forecast_points: List[ForecastPoint] = []
        base_sigma = max(champion_eval.rmse, champion_eval.mae * 1.25, 0.1)

        for step_i, (f_date, f_val) in enumerate(zip(future_dates, future_preds), start=1):
            # Uncertainty expands slightly with horizon: sqrt(step)
            step_uncertainty = z * base_sigma * np.sqrt(step_i / 2.0)
            lower_ci = max(0.0, float(f_val - step_uncertainty))
            upper_ci = max(0.0, float(f_val + step_uncertainty))

            forecast_points.append(
                ForecastPoint(
                    date=f_date,
                    predicted_value=round(float(f_val), 2),
                    lower_bound=round(lower_ci, 2),
                    upper_bound=round(upper_ci, 2),
                )
            )

        result = ForecastResult(
            series_id=series_id,
            commodity_desc=commodity_desc,
            geography_desc=geography_desc,
            target_metric=target_metric,
            champion_model=champion_name,
            horizon_months=horizon_months,
            evaluation=champion_eval,
            predictions=forecast_points,
            candidate_evaluations=candidate_evals,
            evaluation_protocol={
                "strategy": "expanding-window, nonoverlapping full horizons",
                "horizon_months": horizon_months,
                "origins": [target_series.index[o - 1].isoformat() for o in origins],
                "selection_metric": "MAE (same ranking as WAPE when defined)",
                "score_role": "selection validation, not independent test performance",
                "uncertainty": "uncalibrated heuristic bands",
            },
        )

        return result, champion_model_obj
