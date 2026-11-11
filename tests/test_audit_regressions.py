"""Persistence, ETL and CLI regression checks for the engineering audit."""

import json
import sys
import numpy as np
import pytest
from src.mlops.registry import ModelRegistry
from src.mlops.tracker import ExperimentTracker
from src.mlops.monitoring import DataDriftMonitor
from src.use_cases.etl_pipeline import ETLService
from src.domain.exceptions import DataQualityError, DomainValidationError
from src.domain.models import TradeRecord
from src.use_cases.pipeline_orchestrator import AquaTradePipeline
from main import parse_args


def register(registry, **kwargs):
    return registry.register_model(
        {"type": "baseline"}, "fish", "fish", "export_vlu", "Naive", {"mae": 0}, **kwargs
    )


def test_registry_versions_and_portability(tmp_path):
    registry = ModelRegistry(tmp_path / "models")
    one, two = register(registry), register(registry)
    assert one != two
    model, meta = registry.load_model("fish")
    assert model == {"type": "baseline"}
    assert meta["artifact_path"] == two.name
    register(registry, version="explicit")
    with pytest.raises(ValueError):
        register(registry, version="explicit")
    with pytest.raises(ValueError):
        register(registry, version="../escape")


def test_corrupt_indexes_are_not_discarded(tmp_path):
    (tmp_path / "registry_index.json").write_text("{broken")
    with pytest.raises(json.JSONDecodeError):
        ModelRegistry(tmp_path).list_models()
    (tmp_path / "runs.json").write_text("{broken")
    with pytest.raises(json.JSONDecodeError):
        ExperimentTracker(tmp_path).get_runs()


def test_tracker_hashes_all_rows_and_rejects_nested_run(tmp_path):
    import pandas as pd

    tracker = ExperimentTracker(tmp_path)
    run_id = tracker.start_run("audit")
    assert (tmp_path / f"{run_id}.json").exists()
    with pytest.raises(RuntimeError):
        tracker.start_run("nested")
    frame = pd.DataFrame({"x": range(101)})
    tracker.log_dataset_metadata(frame)
    first = tracker._current_run["dataset_info"]["content_sha256"]
    frame.loc[100, "x"] = 999
    tracker.log_dataset_metadata(frame)
    assert first != tracker._current_run["dataset_info"]["content_sha256"]
    tracker.log_metric("undefined", float("nan"))
    tracker.end_run()
    assert json.loads((tmp_path / "runs.json").read_text())[0]["metrics"]["undefined"] is None


def test_missing_hs_codes_are_retained(sample_raw_df):
    frame = sample_raw_df.copy()
    frame["HS_CODE"] = np.nan
    assert len(ETLService().transform(frame)) == 32


def test_negative_measurements_rejected(sample_raw_df):
    sample_raw_df.loc[0, "AMOUNT"] = -99999
    with pytest.raises(DataQualityError):
        ETLService().transform(sample_raw_df)


def test_domain_rejects_nonfinite():
    from datetime import date

    with pytest.raises(DomainValidationError):
        TradeRecord(date.today(), 1, "World", "fish", import_qty=float("inf"))


@pytest.mark.parametrize("flag", ["--horizon", "--top-n"])
@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_cli_invalid_positive_integers(monkeypatch, flag, value):
    monkeypatch.setattr(sys, "argv", ["main.py", flag, value])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


def test_repository_preserves_missing_months_and_rejects_sql(test_repository):
    series = test_repository.get_time_series("Atlantic Salmon")
    assert series.value.isna().any()
    with pytest.raises(ValueError):
        test_repository.get_time_series("fish", metric="1; DROP TABLE fact_trade")
    with pytest.raises(ValueError):
        test_repository.client.save_dataframe(series, "unsafe;DROP")


def test_psi_constant_shift_and_short_samples():
    assert DataDriftMonitor.calculate_psi(np.ones(100), np.ones(100)) == 0
    assert DataDriftMonitor.calculate_psi(np.ones(100), np.ones(100) * 2) > 0.25
    with pytest.raises(ValueError):
        DataDriftMonitor.calculate_psi(np.ones(3), np.ones(3))


def test_pipeline_records_failure(tmp_path, monkeypatch):
    from src.config import AppConfig

    pipeline = AquaTradePipeline(AppConfig(base_dir=tmp_path))

    def fail():
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(pipeline, "run_etl", fail)
    with pytest.raises(RuntimeError):
        pipeline.run_autopilot()
    assert pipeline.tracker.get_runs()[0]["status"].startswith("FAILED")
