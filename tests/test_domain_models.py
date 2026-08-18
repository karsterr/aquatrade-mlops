"""Unit tests for domain entities, value objects, and business validation rules."""

import pytest
from datetime import date
from src.domain.models import (
    TradeRecord,
    Commodity,
    Geography,
    ForecastPoint,
    ModelEvaluation,
    ForecastResult,
)
from src.domain.exceptions import DomainValidationError
from src.adapters.gateways.trade_mapper import TradeMapper


def test_trade_record_valid_creation():
    rec = TradeRecord(
        date=date(2021, 5, 1),
        geography_code=1220,
        geography_desc="Canada",
        commodity_desc="Atlantic Salmon",
        import_qty=1000.0,
        import_vlu=5000.0,
        export_qty=2000.0,
        export_vlu=12000.0,
    )
    assert rec.calculate_import_unit_price() == pytest.approx(5.0)
    assert rec.calculate_export_unit_price() == pytest.approx(6.0)
    assert rec.calculate_trade_balance_vlu() == pytest.approx(7000.0)


def test_trade_record_negative_quantity_raises_error():
    with pytest.raises(DomainValidationError):
        TradeRecord(
            date=date(2021, 5, 1),
            geography_code=1,
            geography_desc="World",
            commodity_desc="Trout",
            import_qty=-10.0,
        )


def test_trade_record_zero_quantity_unit_price():
    rec = TradeRecord(
        date=date(2021, 5, 1),
        geography_code=1,
        geography_desc="World",
        commodity_desc="Trout",
        import_qty=0.0,
        import_vlu=500.0,
    )
    assert rec.calculate_import_unit_price() is None


def test_commodity_empty_description_raises():
    with pytest.raises(DomainValidationError):
        Commodity(hs_code=12345.0, description="  ")


def test_geography_empty_name_raises():
    with pytest.raises(DomainValidationError):
        Geography(code=10, name="")


def test_forecast_point_inverted_bounds_raises():
    with pytest.raises(DomainValidationError):
        ForecastPoint(
            date=date(2022, 1, 1),
            predicted_value=100.0,
            lower_bound=120.0,
            upper_bound=80.0,
        )


def test_forecast_result_serialization():
    evaluation = ModelEvaluation(
        model_name="RandomForest",
        mae=10.5,
        rmse=15.2,
        mape=0.08,
        wape=0.07,
        r2=0.88,
        test_samples=24,
    )
    point = ForecastPoint(
        date=date(2022, 1, 1),
        predicted_value=120.0,
        lower_bound=100.0,
        upper_bound=140.0,
    )
    res = ForecastResult(
        series_id="Salmon__export_vlu__World",
        commodity_desc="Salmon",
        geography_desc="World",
        target_metric="export_vlu",
        champion_model="RandomForest",
        horizon_months=1,
        evaluation=evaluation,
        predictions=[point],
    )
    res_dict = res.to_dict()
    assert res_dict["series_id"] == "Salmon__export_vlu__World"
    assert res_dict["champion_model"] == "RandomForest"
    assert len(res_dict["predictions"]) == 1
