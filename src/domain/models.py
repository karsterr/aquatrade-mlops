"""Domain models and value objects for AquaTrade MLOps."""

from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Dict, List, Optional, Any
import numpy as np

from src.domain.exceptions import DomainValidationError


@dataclass(frozen=True)
class Commodity:
    """Represents a traded aquaculture commodity."""
    hs_code: float
    description: str
    category: str = "Aquaculture"

    def __post_init__(self):
        if not self.description or not self.description.strip():
            raise DomainValidationError("Commodity description cannot be empty.")


@dataclass(frozen=True)
class Geography:
    """Represents a trading country or geographic region."""
    code: int
    name: str

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise DomainValidationError("Geography name cannot be empty.")


@dataclass
class TradeRecord:
    """Core domain entity representing trade statistics for a commodity-geography pair at a monthly date."""
    date: date
    geography_code: int
    geography_desc: str
    commodity_desc: str
    hs_code: Optional[float] = None
    import_qty: Optional[float] = None
    import_vlu: Optional[float] = None
    export_qty: Optional[float] = None
    export_vlu: Optional[float] = None

    def __post_init__(self):
        if self.import_qty is not None and self.import_qty < 0:
            raise DomainValidationError(f"import_qty must be non-negative, got {self.import_qty}")
        if self.import_vlu is not None and self.import_vlu < 0:
            raise DomainValidationError(f"import_vlu must be non-negative, got {self.import_vlu}")
        if self.export_qty is not None and self.export_qty < 0:
            raise DomainValidationError(f"export_qty must be non-negative, got {self.export_qty}")
        if self.export_vlu is not None and self.export_vlu < 0:
            raise DomainValidationError(f"export_vlu must be non-negative, got {self.export_vlu}")

    def calculate_import_unit_price(self) -> Optional[float]:
        """Calculates import unit price ($ / kg or unit). Returns None if quantity is zero or missing."""
        if (
            self.import_qty is not None
            and self.import_qty > 0
            and self.import_vlu is not None
        ):
            return float(self.import_vlu / self.import_qty)
        return None

    def calculate_export_unit_price(self) -> Optional[float]:
        """Calculates export unit price ($ / kg or unit). Returns None if quantity is zero or missing."""
        if (
            self.export_qty is not None
            and self.export_qty > 0
            and self.export_vlu is not None
        ):
            return float(self.export_vlu / self.export_qty)
        return None

    def calculate_trade_balance_vlu(self) -> Optional[float]:
        """Calculates trade balance in value ($): export_vlu - import_vlu."""
        if self.export_vlu is not None and self.import_vlu is not None:
            return float(self.export_vlu - self.import_vlu)
        if self.export_vlu is not None:
            return float(self.export_vlu)
        if self.import_vlu is not None:
            return float(-self.import_vlu)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes entity to dictionary with derived attributes."""
        d = asdict(self)
        d["import_unit_price"] = self.calculate_import_unit_price()
        d["export_unit_price"] = self.calculate_export_unit_price()
        d["trade_balance_vlu"] = self.calculate_trade_balance_vlu()
        return d


@dataclass(frozen=True)
class ForecastPoint:
    """Represents a single forecasted timestamp with confidence bounds."""
    date: date
    predicted_value: float
    lower_bound: float
    upper_bound: float

    def __post_init__(self):
        # Bound predicted values to realistic bounds (non-negative if lower bound negative)
        if self.lower_bound > self.upper_bound:
            raise DomainValidationError("lower_bound cannot exceed upper_bound.")


@dataclass
class ModelEvaluation:
    """Holds backtest evaluation metrics for a forecasting model."""
    model_name: str
    mae: float
    rmse: float
    mape: float
    wape: float
    r2: float
    test_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ForecastResult:
    """Domain aggregate containing the complete forecasting result for a series."""
    series_id: str
    commodity_desc: str
    geography_desc: str
    target_metric: str
    champion_model: str
    horizon_months: int
    evaluation: ModelEvaluation
    predictions: List[ForecastPoint]
    candidate_evaluations: List[ModelEvaluation] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "series_id": self.series_id,
            "commodity_desc": self.commodity_desc,
            "geography_desc": self.geography_desc,
            "target_metric": self.target_metric,
            "champion_model": self.champion_model,
            "horizon_months": self.horizon_months,
            "evaluation": self.evaluation.to_dict(),
            "predictions": [asdict(p) for p in self.predictions],
            "candidate_evaluations": [c.to_dict() for c in self.candidate_evaluations],
            "created_at": self.created_at,
        }


@dataclass
class DataQualityReport:
    """Encapsulates data quality profile metrics."""
    total_records: int
    unique_dates: int
    min_date: date
    max_date: date
    unique_commodities: int
    unique_geographies: int
    missing_import_qty: int
    missing_import_vlu: int
    missing_export_qty: int
    missing_export_vlu: int
    negative_value_count: int
    outlier_count: int
    is_valid: bool
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["min_date"] = str(self.min_date)
        d["max_date"] = str(self.max_date)
        return d
