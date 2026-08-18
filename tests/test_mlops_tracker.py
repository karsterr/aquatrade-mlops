"""Unit tests for ExperimentTracker, ModelRegistry, and DataDriftMonitor."""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.mlops.tracker import ExperimentTracker
from src.mlops.registry import ModelRegistry
from src.mlops.monitoring import DataDriftMonitor


def test_experiment_tracker_lifecycle(temp_dir):
    tracker = ExperimentTracker(temp_dir / "experiments")
    run_id = tracker.start_run("test_experiment", tags={"env": "pytest"})

    tracker.log_param("learning_rate", 0.01)
    tracker.log_metric("mae", 12.34)

    completed = tracker.end_run(status="SUCCESS")
    assert completed["status"] == "SUCCESS"
    assert completed["run_id"] == run_id
    assert completed["metrics"]["mae"] == 12.34

    runs = tracker.get_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id


def test_model_registry_save_and_load(temp_dir):
    registry = ModelRegistry(temp_dir / "models")
    dummy_model = Ridge(alpha=1.0)
    X = np.array([[1, 2], [3, 4]])
    y = np.array([10, 20])
    dummy_model.fit(X, y)
    expected_pred = dummy_model.predict(np.array([[1, 2]]))[0]

    series_id = "Salmon__export_vlu__World"
    registry.register_model(
        model_object=dummy_model,
        series_id=series_id,
        commodity_desc="Salmon",
        target_metric="export_vlu",
        champion_model_name="RidgeRegression",
        metrics={"mae": 1.2, "mape": 0.05},
        version="v1.0.0",
    )

    loaded_model, meta = registry.load_model(series_id, version="v1.0.0")
    assert meta["champion_model_name"] == "RidgeRegression"
    assert loaded_model.predict(np.array([[1, 2]]))[0] == pytest.approx(expected_pred)


def test_data_drift_psi_calculation():
    np.random.seed(42)
    baseline = np.random.normal(100, 15, 1000)
    # Similar distribution -> low PSI
    target_same = np.random.normal(100, 15, 1000)
    # Shifted distribution -> high PSI
    target_shifted = np.random.normal(200, 15, 1000)

    psi_same = DataDriftMonitor.calculate_psi(baseline, target_same)
    psi_shifted = DataDriftMonitor.calculate_psi(baseline, target_shifted)

    assert psi_same < 0.1
    assert psi_shifted > 0.25
