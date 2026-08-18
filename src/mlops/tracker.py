"""MLOps Experiment Tracker for pipeline and model training runs."""

import json
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd


class ExperimentTracker:
    """Tracks training runs, metrics, parameters, dataset signatures, and artifacts."""

    def __init__(self, experiments_dir: Path):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.runs_file = self.experiments_dir / "runs.json"
        self._current_run: Optional[Dict[str, Any]] = None

    def start_run(self, experiment_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Starts a new experiment run session."""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self._current_run = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "status": "RUNNING",
            "tags": tags or {},
            "parameters": {},
            "metrics": {},
            "dataset_info": {},
            "artifacts": [],
        }
        return run_id

    def log_param(self, key: str, value: Any) -> None:
        """Logs a single parameter to current run."""
        if self._current_run:
            self._current_run["parameters"][key] = value

    def log_params(self, params: Dict[str, Any]) -> None:
        """Logs multiple parameters."""
        if self._current_run:
            self._current_run["parameters"].update(params)

    def log_metric(self, key: str, value: float) -> None:
        """Logs a single metric value."""
        if self._current_run:
            self._current_run["metrics"][key] = float(value)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Logs multiple metrics."""
        if self._current_run:
            for k, v in metrics.items():
                self._current_run["metrics"][k] = float(v)

    def log_dataset_metadata(self, df: pd.DataFrame, name: str = "dataset") -> None:
        """Computes and logs dataset size and hash for data versioning."""
        if self._current_run:
            sample_bytes = df.head(100).to_csv(index=False).encode("utf-8")
            data_hash = hashlib.md5(sample_bytes).hexdigest()
            self._current_run["dataset_info"] = {
                "name": name,
                "shape": list(df.shape),
                "columns": df.columns.tolist(),
                "sample_hash": data_hash,
            }

    def log_artifact(self, artifact_path: str) -> None:
        """Registers a saved artifact path."""
        if self._current_run:
            self._current_run["artifacts"].append(str(artifact_path))

    def end_run(self, status: str = "COMPLETED") -> Dict[str, Any]:
        """Finalizes and persists current run record."""
        if not self._current_run:
            raise RuntimeError("No active experiment run to end.")

        self._current_run["end_time"] = datetime.now().isoformat()
        self._current_run["status"] = status

        completed_run = self._current_run
        self._save_run(completed_run)
        self._current_run = None
        return completed_run

    def _save_run(self, run_data: Dict[str, Any]) -> None:
        """Appends run data to global runs.json and individual run file."""
        # Individual run JSON
        run_file = self.experiments_dir / f"{run_data['run_id']}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2)

        # Global runs list
        all_runs = []
        if self.runs_file.exists():
            try:
                with open(self.runs_file, "r", encoding="utf-8") as f:
                    all_runs = json.load(f)
            except Exception:
                all_runs = []

        all_runs.append({
            "run_id": run_data["run_id"],
            "experiment_name": run_data["experiment_name"],
            "start_time": run_data["start_time"],
            "end_time": run_data["end_time"],
            "status": run_data["status"],
            "metrics": run_data["metrics"],
            "parameters": run_data["parameters"],
        })

        with open(self.runs_file, "w", encoding="utf-8") as f:
            json.dump(all_runs, f, indent=2)

    def get_runs(self) -> List[Dict[str, Any]]:
        """Reads all recorded experiment runs."""
        if self.runs_file.exists():
            try:
                with open(self.runs_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []
