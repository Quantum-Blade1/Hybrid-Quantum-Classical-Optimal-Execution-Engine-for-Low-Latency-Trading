"""Execution QUBOs: penalty algebra, energy decomposition, decoding, and the pinned toy problem."""

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qexec.optimization.hft_qubo import HFTExecutionQUBO, HFTQUBOConfig
from qexec.optimization.qubo import ExecutionQUBO, QUBOConfig
from qexec.optimization.schedule import (
    optimize_schedule,
    repair_schedule,
    slice_level_config,
    spread_over_minutes,
)
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
from qexec.optimization.solvers.exact import BruteForceSolver
from qexec.optimization.solvers.metrics import energy_bounds
from qexec.optimization.toy import toy_execution_qubo

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
NO_CAPACITY_LIMIT = 10**9


def execution_qubo(**overrides) -> ExecutionQUBO:
    kwargs = {
        "total_shares": 1000,
        "num_time_slices": 4,
        "num_venues": 2,
        "quantity_levels": [0, 100, 250, 500],
        "max_shares_per_slice": NO_CAPACITY_LIMIT,
    }
    return ExecutionQUBO(QUBOConfig(**{**kwargs, **overrides}))


def hft_qubo(**overrides) -> HFTExecutionQUBO:
    kwargs = {
        "total_shares": 500,
        "num_tick_slices": 3,
        "num_venues": 2,
        "quantity_levels": [0, 100, 250],
        "kyle_lambda": 0.02,
        "vpin": 0.4,
        "adverse_selection_cost": 0.01,
        "max_shares_per_tick": NO_CAPACITY_LIMIT,
    }
    return HFTExecutionQUBO(HFTQUBOConfig(**{**kwargs, **overrides}))


def binary_vectors(n: int):
    return st.lists(st.integers(0, 1), min_size=n, max_size=n).map(np.array)


# --- ExecutionQUBO -------------------------------------------------------------------------


@given(x=binary_vectors(32))
def test_equality_penalty_is_p_times_squared_shortfall(x):
    # Without the capacity term, x^T C x = P (sum q x - S)^2 - P S^2, so the penalty
    # (plus the dropped constant) is zero iff the selected quantity equals the order size.
    qubo = execution_qubo()
    cfg = qubo.config
    costs = qubo.calculate_solution_cost(x)
    shortfall = costs["total_shares"] - cfg.total_shares
    penalty = costs["constraint_penalty"] + cfg.equality_penalty * cfg.total_shares**2
    assert penalty == pytest.approx(cfg.equality_penalty * shortfall**2, abs=1e-6)
    assert (abs(penalty) < 1e-6) == (shortfall == 0)


@given(x=binary_vectors(32))
def test_execution_qubo_energy_is_sum_of_weighted_terms(x):
    qubo = execution_qubo(max_shares_per_slice=600)
    costs = qubo.calculate_solution_cost(x)
    parts = ("impact_cost", "timing_cost", "transaction_cost", "constraint_penalty")
    assert costs["total_cost"] == pytest.approx(sum(costs[p] for p in parts), rel=1e-9)


@given(x=binary_vectors(32))
def test_slice_quantities_are_non_negative_and_sum_to_selected_shares(x):
    qubo = execution_qubo()
    quantities = qubo.slice_quantities(x)
    assert quantities.shape == (qubo.config.num_time_slices,)
    assert np.all(quantities >= 0)
    assert quantities.sum() == qubo.calculate_solution_cost(x)["total_shares"]
    trades = qubo.interpret_solution(x)
    assert (trades["quantity"].sum() if len(trades) else 0) == quantities.sum()


def test_variable_index_is_a_bijection():
    cfg = execution_qubo().config
    indices = [
        cfg.variable_index(t, v, k)
        for t in range(cfg.num_time_slices)
        for v in range(cfg.num_venues)
        for k in range(cfg.num_quantity_levels)
    ]
    assert sorted(indices) == list(range(cfg.num_variables))
    assert all(cfg.variable_index(*cfg.decode_index(i)) == i for i in indices)


def test_capacity_violation_raises_energy():
    qubo = execution_qubo(quantity_levels=[0, 250, 500], max_shares_per_slice=500)
    cfg = qubo.config
    Q = qubo.build_qubo_matrix()
    within = np.zeros(cfg.num_variables)
    for t in range(4):
        within[cfg.variable_index(t, 0, 1)] = 1  # 4 x 250 = 1000 shares, 250 per slice
    over = np.zeros(cfg.num_variables)
    over[cfg.variable_index(0, 0, 2)] = 1
    over[cfg.variable_index(0, 1, 2)] = 1  # 500 on each venue in slice 0: 1000 > cap
    assert qubo.slice_quantities(over).sum() == qubo.slice_quantities(within).sum() == 1000
    assert float(over @ Q @ over) > float(within @ Q @ within)


# --- Schedules from solved QUBOs ------------------------------------------------------------


@pytest.mark.parametrize(("total_shares", "num_slices"), [(900, 3), (1200, 4), (1000, 5)])
def test_exact_slice_schedule_is_feasible(total_shares, num_slices):
    # Levels {0, N/2T, N/T}: when they divide N the optimum meets the order size exactly.
    qubo = ExecutionQUBO(slice_level_config(total_shares, num_slices))
    schedule, _ = optimize_schedule(qubo, BruteForceSolver())
    assert np.all(schedule >= 0)
    assert schedule.sum() == total_shares


def test_annealed_slice_schedule_reaches_the_exact_optimum():
    qubo = ExecutionQUBO(slice_level_config(1000, 5))
    Q = qubo.build_qubo_matrix()
    schedule, result = optimize_schedule(qubo, SimulatedAnnealingSolver(num_sweeps=300, seed=0), Q)
    assert result.energy == pytest.approx(energy_bounds(Q).min_energy)
    assert result.energy == pytest.approx(float(result.solution @ Q @ result.solution))
    assert schedule.sum() == 1000


@given(
    quantities=st.lists(st.integers(0, 10_000), min_size=1, max_size=12),
    extra_minutes=st.integers(0, 50),
)
def test_spread_over_minutes_conserves_shares(quantities, extra_minutes):
    num_minutes = len(quantities) + extra_minutes
    schedule = spread_over_minutes(np.array(quantities, dtype=float), num_minutes)
    assert len(schedule) == num_minutes
    assert np.all(schedule >= 0)
    assert schedule.sum() == sum(quantities)


# --- HFT QUBO ---------------------------------------------------------------------------------


@given(x=binary_vectors(18))
def test_hft_qubo_energy_is_sum_of_weighted_terms(x):
    qubo = hft_qubo(max_shares_per_tick=300)
    costs = qubo.calculate_cost_breakdown(x)
    parts = [k for k in costs if k != "total_cost"]
    assert len(parts) == 7
    assert costs["total_cost"] == pytest.approx(sum(costs[p] for p in parts), rel=1e-9, abs=1e-9)


@given(x=binary_vectors(18))
def test_hft_equality_penalty_is_zero_iff_order_size_selected(x):
    qubo = hft_qubo()
    cfg = qubo.config
    costs = qubo.calculate_cost_breakdown(x)
    shortfall = qubo.slice_quantities(x).sum() - cfg.total_shares
    penalty = costs["constraint_penalty"] + cfg.equality_penalty * cfg.total_shares**2
    assert penalty == pytest.approx(cfg.equality_penalty * shortfall**2, abs=1e-6)


def test_hft_qubo_minimum_selects_exactly_the_order_size():
    qubo = hft_qubo()
    result = BruteForceSolver().solve(qubo.build_qubo_matrix())
    decoded = qubo.interpret_solution(result.solution)
    assert decoded["total_shares"] == qubo.config.total_shares
    assert decoded["fill_rate"] == 1.0


def test_hft_qubo_vpin_raises_cost_of_the_same_schedule():
    x = np.zeros(18)
    x[[1, 8, 14]] = 1
    calm = hft_qubo(vpin=0.0).calculate_cost_breakdown(x)["adverse_selection_cost"]
    toxic = hft_qubo(vpin=0.9).calculate_cost_breakdown(x)["adverse_selection_cost"]
    assert toxic == pytest.approx(calm * (1 + 2 * 0.9) / (1 + 2 * 0.0))
    assert toxic > calm > 0


# --- Toy hardware-benchmark QUBO --------------------------------------------------------------


def test_toy_execution_qubo_n4_is_pinned():
    # Changing this matrix changes the problem that results/bench_hw_*.json were measured on.
    expected = np.array(
        [
            [-29.85, 20.0, 10.0, 20.0],
            [20.0, -39.5, 20.0, 40.0],
            [10.0, 20.0, -29.8, 20.0],
            [20.0, 40.0, 20.0, -39.4],
        ]
    )
    np.testing.assert_allclose(toy_execution_qubo(4), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("n", [4, 6, 8, 10])
def test_toy_execution_qubo_optimum_matches_committed_benchmarks(n):
    for name in (f"bench_hw_n{n}.json", f"bench_n{n}.json"):
        records = json.loads((RESULTS_DIR / name).read_text())
        recorded = {round(r["optimal_energy"], 9) for r in records}
        assert recorded == {round(energy_bounds(toy_execution_qubo(n)).min_energy, 9)}


# --- Schedule repair ----------------------------------------------------------------------


@given(
    quantities=st.lists(st.floats(0, 1e5, allow_nan=False), min_size=1, max_size=30),
    total=st.integers(0, 10**7),
)
def test_repair_schedule_sums_exactly_to_the_order(quantities, total):
    repaired = repair_schedule(np.array(quantities), total)
    assert repaired.dtype.kind == "i"
    assert repaired.sum() == total
    assert np.all(repaired >= 0)
    q = np.array(quantities)
    if q.sum() > 0:
        # Proportional up to rounding: each slice is within one share of its exact share.
        assert np.all(np.abs(repaired - q * total / q.sum()) < 1 + 1e-9)
        assert np.all(repaired[q == 0] == 0)


def test_repair_schedule_fixes_the_sa_4997_of_5000_case():
    # Levels N/(2T), N/T with integer division cannot reach 5000 exactly for T = 3.
    qubo = ExecutionQUBO(slice_level_config(5000, 3))
    slice_qty, _ = optimize_schedule(qubo, BruteForceSolver())
    assert slice_qty.sum() != 5000
    repaired = repair_schedule(slice_qty, 5000)
    assert repaired.sum() == 5000


def test_repair_schedule_of_an_empty_plan_is_uniform():
    assert repair_schedule(np.zeros(4), 10).tolist() == [3, 3, 2, 2]
    with pytest.raises(ValueError):
        repair_schedule(np.array([]), 5)
