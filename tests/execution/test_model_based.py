"""AC, QUBO and adaptive hybrid schedules for the execution cost model."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from qexec.execution.cost_model import CostModel
from qexec.execution.engine import ExecutionEngine, OrderSide, ParentOrder
from qexec.execution.fill import ImpactFillModel
from qexec.execution.strategies.model_based import (
    AdaptiveQUBOStrategy,
    QUBOSettings,
    ac_schedule,
    dp_schedule,
    qubo_plan,
)

SETTINGS = QUBOSettings(num_slices=4, bits=3, units=12, sweeps=300, restarts=8)


def model(minutes: int = 20, risk_aversion: float = 0.0) -> CostModel:
    x = np.linspace(0, 1, minutes)
    return CostModel(
        expected_volume=1000 * (1 + 4 * (x - 0.5) ** 2),
        half_spread_bps=np.full(minutes, 0.5),
        sigma_bps=np.full(minutes, 4.0),
        impact_bps=1.5,
        risk_aversion=risk_aversion,
    )


def test_ac_schedule_is_integer_and_complete():
    schedule = ac_schedule(model(), 2000)
    assert schedule.sum() == 2000 and np.all(schedule >= 0)


@pytest.mark.parametrize("lam", [0.0, 0.05])
def test_qubo_plan_reaches_the_integer_optimum_and_stays_close_to_ac(lam):
    m = model(risk_aversion=lam)
    plan = qubo_plan(m, 2000, SETTINGS, seed=0)
    assert plan.feasible
    assert plan.schedule.sum() == 2000
    dp = dp_schedule(m, 2000, SETTINGS)
    assert plan.objective_bps == pytest.approx(m.objective(dp, 2000), abs=1e-9)
    assert plan.gap_to_ip_optimum_bps == pytest.approx(0.0, abs=1e-6)
    ac = m.objective(ac_schedule(m, 2000), 2000)
    assert ac <= plan.objective_bps + 1e-9
    if lam == 0:  # with strong urgency the 4-slice discretization costs more (see RESULTS)
        assert plan.objective_bps - ac < 0.05


def bars(minutes: int, volume_scale: float = 1.0) -> pd.DataFrame:
    m = model(minutes)
    start = datetime(2026, 7, 22)
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 4e-4, minutes)))
    return pd.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=k) for k in range(minutes)],
            "price": close,
            "close": close,
            "spread": 0.0,
            "volume": m.expected_volume * volume_scale,
            "expected_volume": m.expected_volume,
            "half_spread_bps": 0.5,
            "half_spread_obs": np.where(np.arange(minutes) % 2 == 0, 1.0, np.nan),
        }
    )


def test_adaptive_strategy_rescales_the_model_from_observed_bars():
    data = bars(20, volume_scale=4.0)
    strategy = AdaptiveQUBOStrategy(model(20), SETTINGS, seed=1)
    updated = strategy.updated_model(10, data)
    assert updated.num_minutes == 10
    # observed volume is 4x the expectation, clipped to 2x; spread 1.0 vs 0.5 -> 2x
    assert np.allclose(updated.expected_volume, model(20).expected_volume[10:] * 2.0)
    assert np.allclose(updated.half_spread_bps, 1.0)


def test_adaptive_strategy_replans_at_checkpoints_and_completes_the_order():
    data = bars(20)
    strategy = AdaptiveQUBOStrategy(model(20), SETTINGS, seed=1, num_checkpoints=3)
    engine = ExecutionEngine(fill_model=ImpactFillModel(impact_bps=1.5, participation_cap=0.25))
    order = ParentOrder("X", OrderSide.BUY, 2000, 20)
    report = engine.process_order(order, data, strategy, arrival_price=100.0)
    assert strategy.invocations == 4  # initial plan + 3 checkpoints
    assert strategy.infeasible_solutions == 0
    assert report.filled_quantity == 2000
    assert strategy.replan(7, 100, data, 20) is None  # not a checkpoint
