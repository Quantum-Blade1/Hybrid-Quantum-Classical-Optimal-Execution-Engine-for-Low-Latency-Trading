"""Real-bar impact fill model, alone and through the shared ExecutionEngine rules."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from qexec.execution.engine import ExecutionEngine, OrderSide, ParentOrder
from qexec.execution.fill import ImpactFillModel
from qexec.execution.strategies.fixed import FixedScheduleStrategy


def bar(price=100.0, volume=1000.0, half_spread=1.0, **extra):
    return pd.Series(
        {"price": price, "spread": 0.0, "volume": volume, "half_spread_bps": half_spread, **extra}
    )


def test_fill_price_is_half_spread_plus_linear_impact():
    model = ImpactFillModel(impact_bps=20.0, participation_cap=0.25)
    fill = model.fill(bar(), 100, "buy", key=0)
    assert fill.quantity == 100
    # 1 bps half spread + 20 bps * 100/1000 participation = 3 bps
    assert fill.average_price == pytest.approx(100 * (1 + 3e-4))
    assert fill.half_spread_cost == pytest.approx(100 * 1e-4 * 100)
    assert fill.impact_cost == pytest.approx(100 * 2e-4 * 100)
    sell = model.fill(bar(), 100, "sell", key=0)
    assert sell.average_price == pytest.approx(100 * (1 - 3e-4))


def test_participation_cap_and_empty_bars():
    model = ImpactFillModel(impact_bps=20.0, participation_cap=0.25)
    assert model.fill(bar(), 1000, "buy", key=0).quantity == 250
    assert model.fill(bar(volume=0.0), 10, "buy", key=0).quantity == 0
    assert model.fill(bar(volume=3.0), 10, "buy", key=0).quantity == 0  # floor(0.75) = 0


def test_missing_half_spread_falls_back_to_spread_column():
    model = ImpactFillModel(impact_bps=0.0)
    row = pd.Series({"price": 100.0, "spread": 0.02, "volume": 1000.0, "half_spread_bps": np.nan})
    assert model.half_spread_bps(row) == pytest.approx(1.0)


def test_completion_price_has_no_cap_and_uses_expected_volume_when_empty():
    model = ImpactFillModel(impact_bps=20.0, participation_cap=0.25)
    assert model.completion_price(bar(), 1000, "buy", key=0) == pytest.approx(100 * (1 + 21e-4))
    empty = bar(volume=0.0, expected_volume=500.0)
    assert model.completion_price(empty, 100, "buy", key=0) == pytest.approx(100 * (1 + 5e-4))


def test_invalid_parameters():
    with pytest.raises(ValueError):
        ImpactFillModel(impact_bps=-1)
    with pytest.raises(ValueError):
        ImpactFillModel(impact_bps=1, participation_cap=0)


def test_engine_carries_forward_and_charges_opportunity_cost():
    start = datetime(2026, 7, 22)
    data = pd.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=k) for k in range(3)],
            "price": [100.0, 100.0, 100.0],
            "spread": [0.0] * 3,
            "volume": [400.0, 0.0, 400.0],
            "half_spread_bps": [1.0, 1.0, 1.0],
        }
    )
    engine = ExecutionEngine(fill_model=ImpactFillModel(impact_bps=10.0, participation_cap=0.25))
    order = ParentOrder("X", OrderSide.BUY, 300, 3)
    report = engine.process_order(
        order, data, FixedScheduleStrategy([300, 0, 0]), arrival_price=100.0
    )
    # minute 0: 100 filled; minute 1: no volume; minute 2: 100 filled; 100 left over.
    assert report.filled_quantity == 200
    assert report.unfilled_quantity == 100
    fill_bps = 1 + 10 * 100 / 400
    assert report.average_execution_price == pytest.approx(100 * (1 + fill_bps / 1e4))
    clean_bps = 1 + 10 * 100 / 400
    expected = (200 * fill_bps + 100 * clean_bps) / 300
    assert report.implementation_shortfall_bps == pytest.approx(expected)
