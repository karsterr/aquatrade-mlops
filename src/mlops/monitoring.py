"""Data Drift Monitoring and Distribution Shift Analysis."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


@dataclass
class DriftMetric:
    feature_name: str
    psi_score: float
    status: str  # 'NO_DRIFT', 'MODERATE_DRIFT', 'SIGNIFICANT_DRIFT'
    baseline_mean: float
    target_mean: float
    mean_change_pct: float


@dataclass
class DriftReport:
    metrics: List[DriftMetric]
    overall_status: str
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "generated_at": self.generated_at,
            "metrics": [asdict(m) for m in self.metrics],
        }


class DataDriftMonitor:
    """Detects statistical shifts between historical baseline and recent inference data."""

    @staticmethod
    def calculate_psi(
        baseline: np.ndarray, target: np.ndarray, num_buckets: int = 10, epsilon: float = 1e-4
    ) -> float:
        """Calculates Population Stability Index (PSI) between baseline and target distributions."""
        baseline = np.asarray(baseline, dtype=float)
        baseline = baseline[np.isfinite(baseline)]
        target = np.asarray(target, dtype=float)
        target = target[np.isfinite(target)]

        if len(baseline) < 10 or len(target) < 10:
            raise ValueError("PSI requires at least ten finite observations in each sample.")

        # Create quantiles on baseline
        percentiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(baseline, percentiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        # Ensure unique bin edges
        bins = np.unique(bins)
        if np.ptp(baseline) == 0 or len(bins) < 3:
            center = float(baseline[0])
            bins = np.array([-np.inf, center - 1e-8, center + 1e-8, np.inf])

        baseline_counts, _ = np.histogram(baseline, bins=bins)
        target_counts, _ = np.histogram(target, bins=bins)

        baseline_pct = baseline_counts / len(baseline)
        target_pct = target_counts / len(target)

        # Smooth zero counts with epsilon
        baseline_pct = np.where(baseline_pct == 0, epsilon, baseline_pct)
        target_pct = np.where(target_pct == 0, epsilon, target_pct)

        # Normalize
        baseline_pct = baseline_pct / np.sum(baseline_pct)
        target_pct = target_pct / np.sum(target_pct)

        psi_value = np.sum((target_pct - baseline_pct) * np.log(target_pct / baseline_pct))
        return float(max(0.0, psi_value))

    def evaluate_drift(
        self,
        baseline_df: pd.DataFrame,
        target_df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> DriftReport:
        """Computes PSI and distribution shift across specified numerical features."""
        from datetime import datetime

        if features is None:
            features = [
                "import_qty",
                "import_vlu",
                "export_qty",
                "export_vlu",
                "import_unit_price",
                "export_unit_price",
            ]

        metrics: List[DriftMetric] = []
        has_significant_drift = False
        has_moderate_drift = False

        for feat in features:
            if feat in baseline_df.columns and feat in target_df.columns:
                b_series = baseline_df[feat].dropna().values
                t_series = target_df[feat].dropna().values
                b_series = b_series[np.isfinite(b_series)]
                t_series = t_series[np.isfinite(t_series)]

                if np.isfinite(b_series).sum() >= 10 and np.isfinite(t_series).sum() >= 10:
                    psi = self.calculate_psi(b_series, t_series)
                    b_mean = float(np.mean(b_series))
                    t_mean = float(np.mean(t_series))
                    mean_change_pct = (
                        float(((t_mean - b_mean) / b_mean) * 100) if b_mean != 0 else 0.0
                    )

                    if psi < 0.1:
                        status = "NO_DRIFT"
                    elif psi < 0.25:
                        status = "MODERATE_DRIFT"
                        has_moderate_drift = True
                    else:
                        status = "SIGNIFICANT_DRIFT"
                        has_significant_drift = True

                    metrics.append(
                        DriftMetric(
                            feature_name=feat,
                            psi_score=round(psi, 4),
                            status=status,
                            baseline_mean=round(b_mean, 2),
                            target_mean=round(t_mean, 2),
                            mean_change_pct=round(mean_change_pct, 2),
                        )
                    )

        if has_significant_drift:
            overall_status = "SIGNIFICANT_DRIFT"
        elif has_moderate_drift:
            overall_status = "MODERATE_DRIFT"
        elif not metrics:
            overall_status = "INSUFFICIENT_DATA"
        else:
            overall_status = "STABLE"

        return DriftReport(
            metrics=metrics,
            overall_status=overall_status,
            generated_at=datetime.now().isoformat(),
        )
