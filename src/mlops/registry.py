"""Model Registry for versioning, saving, loading, and tracking model artifacts."""

import json
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class ModelRegistry:
    """Handles serialization, versioning, and metadata indexing for trained ML models."""

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_index_file = self.models_dir / "registry_index.json"

    def register_model(
        self,
        model_object: Any,
        series_id: str,
        commodity_desc: str,
        target_metric: str,
        champion_model_name: str,
        metrics: Dict[str, float],
        version: str = "v1.0.0",
        feature_columns: Optional[list] = None,
    ) -> Path:
        """Saves a model artifact and updates the registry index."""
        clean_series_id = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in series_id)
        artifact_filename = f"model_{clean_series_id}_{version}.joblib"
        artifact_path = self.models_dir / artifact_filename

        # Save model using joblib
        joblib.dump(model_object, artifact_path)

        # Build metadata payload
        metadata = {
            "series_id": series_id,
            "commodity_desc": commodity_desc,
            "target_metric": target_metric,
            "champion_model_name": champion_model_name,
            "version": version,
            "artifact_path": str(artifact_path),
            "metrics": metrics,
            "feature_columns": feature_columns or [],
            "registered_at": datetime.now().isoformat(),
        }

        # Save individual metadata JSON
        meta_path = self.models_dir / f"model_{clean_series_id}_{version}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Update global index
        self._update_index(series_id, metadata)
        return artifact_path

    def load_model(self, series_id: str, version: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """Loads a model artifact and its metadata by series_id."""
        index = self._load_index()
        if series_id not in index:
            raise KeyError(f"No model registered for series_id: '{series_id}'")

        series_entry = index[series_id]
        target_version = version or series_entry.get("latest_version")
        if not target_version or target_version not in series_entry.get("versions", {}):
            raise KeyError(f"Version '{target_version}' not found for series_id '{series_id}'")

        meta = series_entry["versions"][target_version]
        artifact_path = Path(meta["artifact_path"])
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact file missing: {artifact_path}")

        model_obj = joblib.load(artifact_path)
        return model_obj, meta

    def list_models(self) -> Dict[str, Any]:
        """Returns the full registry index."""
        return self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        if self.registry_index_file.exists():
            try:
                with open(self.registry_index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _update_index(self, series_id: str, metadata: Dict[str, Any]) -> None:
        index = self._load_index()
        version = metadata["version"]

        if series_id not in index:
            index[series_id] = {
                "series_id": series_id,
                "commodity_desc": metadata["commodity_desc"],
                "target_metric": metadata["target_metric"],
                "latest_version": version,
                "versions": {},
            }

        index[series_id]["latest_version"] = version
        index[series_id]["versions"][version] = metadata

        with open(self.registry_index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
