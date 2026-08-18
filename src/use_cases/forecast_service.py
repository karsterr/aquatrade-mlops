"""ML-based time-series forecasting service with automated model selection and backtesting."""

from datetime import date, datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.domain.models import ForecastPoint, ModelEvaluation, ForecastResult
from src.domain.exceptions import InsufficientDataError, ModelTrainingError


class FeatureEngineer:
    """Creates lag, rolling, seasonal, and trend features for time-series forecasting."""

    @staticmethod
    def create_features(
        series: pd.Series,
        lags: List[int] = [1, 2, 3, 6, 12],
        rolling_windows: List[int] = [3, 6, 12]
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
            df[f"rolling_std_{w}"] = df["target"].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
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
        self.test_split_months = test_split_months
        self.min_series_length = min_series_length
        self.confidence_level = confidence_level

    def _get_candidate_models(self) -> Dict[str, Any]:
        """Returns initialized candidate forecasting estimators."""
        return {
            "LinearTrend": LinearRegression(),
            "RidgeRegression": Ridge(alpha=1.0, random_state=42),
            "RandomForest": RandomForestRegressor(n_estimators=60, max_depth=6, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42),
        }

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> ModelEvaluation:
        """Calculates standard regression and percentage metrics."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        # Clip negative predictions for trade facts
        y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        # WAPE (Weighted Absolute Percentage Error)
        sum_true = float(np.sum(np.abs(y_true)))
        wape = float(np.sum(np.abs(y_true - y_pred)) / (sum_true + 1e-8))

        # MAPE
        epsilon = 1.0 if np.mean(y_true) > 100 else 0.01
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))))

        # R2
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 and np.var(y_true) > 0 else 0.0

        return ModelEvaluation(
            model_name=model_name,
            mae=round(mae, 3),
            rmse=round(rmse, 3),
            mape=round(mape, 4),
            wape=round(wape, 4),
            r2=round(r2, 4),
            test_samples=len(y_true),
        )

    def _forecast_baseline(
        self,
        train_series: pd.Series,
        horizon: int,
        strategy: str = "Naive"
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
        if time_series_df.empty or len(time_series_df) < self.min_series_length:
            raise InsufficientDataError(
                f"Series '{series_id}' has {len(time_series_df)} points, minimum required is {self.min_series_length}."
            )

        ts = time_series_df.copy()
        ts["date"] = pd.to_datetime(ts["date"])
        ts = ts.sort_values("date").set_index("date")
        target_series = ts["value"].astype(float)

        # Train / Test Split for Backtesting
        split_point = len(target_series) - self.test_split_months
        if split_point < 12:
            split_point = int(len(target_series) * 0.8)

        train_s = target_series.iloc[:split_point]
        test_s = target_series.iloc[split_point:]
        test_h = len(test_s)

        candidate_evals: List[ModelEvaluation] = []
        candidate_models = self._get_candidate_models()

        # 1. Evaluate Baselines
        for b_name in ["Naive", "SeasonalNaive", "MovingAverage"]:
            preds = self._forecast_baseline(train_s, test_h, strategy=b_name)
            eval_res = self._calculate_metrics(test_s.values, preds, b_name)
            candidate_evals.append(eval_res)

        # 2. Evaluate ML Regressors
        feat_df = FeatureEngineer.create_features(target_series)
        feat_cols = [c for c in feat_df.columns if c != "target"]

        # Drop warm-up rows with NaNs in feature space
        valid_mask = ~feat_df[feat_cols].isna().any(axis=1)
        valid_feat_df = feat_df[valid_mask]

        # Train split in feature space
        train_feat = valid_feat_df.loc[valid_feat_df.index.isin(train_s.index)]
        test_feat = valid_feat_df.loc[valid_feat_df.index.isin(test_s.index)]

        fitted_ml_models: Dict[str, Any] = {}

        if len(train_feat) > 10 and len(test_feat) > 0:
            X_train = train_feat[feat_cols]
            y_train = train_feat["target"]
            X_test = test_feat[feat_cols]
            y_test = test_feat["target"]

            for m_name, estimator in candidate_models.items():
                try:
                    estimator.fit(X_train, y_train)
                    y_pred = estimator.predict(X_test)
                    eval_res = self._calculate_metrics(y_test.values, y_pred, m_name)
                    candidate_evals.append(eval_res)
                    fitted_ml_models[m_name] = estimator
                except Exception:
                    continue

        # 3. Select Champion Model (Lowest WAPE, tiebreaker lowest MAE)
        candidate_evals.sort(key=lambda x: (x.wape, x.mae))
        champion_eval = candidate_evals[0]
        champion_name = champion_eval.model_name

        # 4. Generate Future Forecasts (Refit on full dataset)
        future_dates = [
            target_series.index.max().date() + relativedelta(months=i)
            for i in range(1, horizon_months + 1)
        ]

        future_preds: np.ndarray
        champion_model_obj: Any = None

        if champion_name in ["Naive", "SeasonalNaive", "MovingAverage"]:
            future_preds = self._forecast_baseline(target_series, horizon_months, strategy=champion_name)
            champion_model_obj = {"type": "baseline", "strategy": champion_name}
        else:
            # Full ML fit
            X_full = valid_feat_df[feat_cols]
            y_full = valid_feat_df["target"]
            model_class = candidate_models[champion_name]
            model_class.fit(X_full, y_full)
            champion_model_obj = model_class

            # Recursive Multi-Step Forecasting
            future_preds = []
            extended_series = target_series.copy()

            for future_d in future_dates:
                # Append next placeholder
                extended_series[pd.to_datetime(future_d)] = extended_series.iloc[-1]
                # Re-generate features for the last point
                temp_feat = FeatureEngineer.create_features(extended_series)
                last_row = temp_feat[feat_cols].iloc[-1:]
                next_pred = float(model_class.predict(last_row)[0])
                next_pred = max(0.0, next_pred)
                # Overwrite placeholder with predicted value
                extended_series.iloc[-1] = next_pred
                future_preds.append(next_pred)

            future_preds = np.array(future_preds)

        # 5. Compute Prediction Intervals (95% CI based on backtest RMSE & horizon scale)
        z = 1.96 if self.confidence_level == 0.95 else 1.645
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
        )

        return result, champion_model_obj
