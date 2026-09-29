"""Cost model, discretized Almgren-Chriss optimum, and the exact binary-encoded QUBO."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qexec.execution.cost_model import CostModel
from qexec.optimization.slice_program import SliceProgram
from qexec.optimization.solvers.metrics import enumerate_energies


def model(minutes: int, seed: int, risk_aversion: float = 0.0, impact: float = 30.0) -> CostModel:
    rng = np.random.default_rng(seed)
    return CostModel(
        expected_volume=rng.uniform(200, 2000, minutes),
        half_spread_bps=rng.uniform(0.0, 2.0, minutes),
        sigma_bps=rng.uniform(1.0, 8.0, minutes),
        impact_bps=impact,
        risk_aversion=risk_aversion,
    )


@given(seed=st.integers(0, 10_000), lam=st.sampled_from([0.0, 1e-3, 1e-2]))
@settings(max_examples=30, deadline=None)
def test_quadratic_form_matches_direct_objective(seed, lam):
    m = model(12, seed, lam)
    q = np.random.default_rng(seed).uniform(0, 100, 12)
    total = 500.0
    A, b, c = m.quadratic_form(total)
    f = q / total
    assert f @ A @ f + b @ f + c == pytest.approx(m.objective(q, total), rel=1e-10)


def test_variance_of_immediate_execution_is_the_first_minute_only():
    m = model(5, 0)
    assert m.variance_bps2([100, 0, 0, 0, 0], 100) == pytest.approx(m.sigma_bps[0] ** 2)


@pytest.mark.parametrize("lam", [0.0, 1e-3, 1e-1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_optimal_fractions_satisfy_kkt_and_beat_perturbations(lam, seed):
    m = model(30, seed, lam)
    total = 20_000.0
    f = m.optimal_fractions(total)
    assert f.sum() == pytest.approx(1.0) and np.all(f >= 0)
    assert m.kkt_residual(f, total) < 1e-5
    best = m.objective(f * total, total)
    rng = np.random.default_rng(seed)
    for _ in range(50):
        g = np.clip(f + rng.normal(0, 0.01, f.size), 0, None)
        g /= g.sum()
        assert m.objective(g * total, total) >= best - 1e-9


def test_risk_neutral_optimum_with_equal_spreads_is_proportional_to_volume():
    volume = np.array([100.0, 300.0, 600.0])
    m = CostModel(volume, np.ones(3), np.ones(3), impact_bps=10.0)
    assert m.optimal_fractions(1000.0) == pytest.approx(volume / volume.sum())


def test_cost_model_validation():
    with pytest.raises(ValueError):
        CostModel(np.ones(3), np.ones(2), np.ones(3), 1.0)
    with pytest.raises(ValueError):
        CostModel(np.zeros(3), np.ones(3), np.ones(3), 1.0)
    with pytest.raises(ValueError):
        CostModel(np.ones(3), np.ones(3), np.ones(3), 0.0)
    m = model(10, 0)
    assert m.window(2, 5).num_minutes == 3
    assert m.with_impact(5.0).impact_bps == 5.0
    assert m.with_risk_aversion(0.1).risk_aversion == 0.1


def program(slices=3, *, bits=2, units=4, minutes=9, seed=0, lam=1e-2) -> SliceProgram:
    return SliceProgram(
        model(minutes, seed, lam), total=5000, num_slices=slices, units=units, bits=bits
    )


@pytest.mark.parametrize(
    ("slices", "bits", "units", "seed", "lam"),
    [
        (2, 3, 6, 0, 0.0),
        (3, 2, 4, 1, 1e-2),
        (4, 2, 6, 2, 1e-1),
        (3, 3, 8, 3, 1e-3),
        (5, 2, 7, 4, 1e-2),
    ],
)
def test_qubo_optimum_decodes_to_the_integer_program_optimum(slices, bits, units, seed, lam):
    p = program(slices, bits=bits, units=units, minutes=slices * 3, seed=seed, lam=lam)
    problem = p.qubo()
    energies = enumerate_energies(problem.Q) + problem.offset
    best = int(np.argmin(energies))
    z = np.array([(best >> i) & 1 for i in range(p.num_variables)])
    counts = p.decode(z)
    _, ip_value = p.solve_enumeration()
    assert p.is_feasible(counts)
    assert energies[best] == pytest.approx(ip_value, rel=1e-9, abs=1e-12)
    assert p.objective(counts) == pytest.approx(ip_value, rel=1e-9, abs=1e-12)
    # Every infeasible assignment is strictly worse than the optimum.
    for index in np.argsort(energies)[:20]:
        c = p.decode(np.array([(int(index) >> i) & 1 for i in range(p.num_variables)]))
        if not p.is_feasible(c):
            assert energies[index] > ip_value


@given(seed=st.integers(0, 1000), lam=st.sampled_from([0.0, 1e-3, 5e-2]))
@settings(max_examples=25, deadline=None)
def test_qubo_energy_equals_objective_plus_penalty(seed, lam):
    p = program(3, bits=3, units=9, minutes=10, seed=seed, lam=lam)
    problem = p.qubo()
    z = np.random.default_rng(seed).integers(0, 2, p.num_variables)
    c = p.decode(z)
    expected = p.objective(c) + problem.penalty * (c.sum() - p.units) ** 2
    assert problem.energy(z) == pytest.approx(expected, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("lam", [0.0, 1e-2, 1.0])
def test_dynamic_programming_is_exact(seed, lam):
    p = program(4, bits=3, units=10, minutes=13, seed=seed, lam=lam)
    dp_counts, dp_value = p.solve_dp()
    _, enum_value = p.solve_enumeration()
    assert dp_value == pytest.approx(enum_value, rel=1e-9)
    assert p.objective(dp_counts) == pytest.approx(dp_value, rel=1e-9)
    assert p.is_feasible(dp_counts)


def test_encode_decode_and_schedule():
    p = program(3, bits=3, units=9, minutes=10)
    c = np.array([2, 7, 0])
    assert p.decode(p.encode(c)).tolist() == c.tolist()
    schedule = p.schedule(np.array([3, 3, 3]))
    assert schedule.sum() == p.total and schedule.size == 10
    assert p.slice_bounds == ((0, 4), (4, 7), (7, 10))
    assert np.allclose(p.weights.sum(axis=0), 1.0)
    with pytest.raises(ValueError):
        p.encode(np.array([8, 0, 1]))
    with pytest.raises(ValueError):
        SliceProgram(model(4, 0), total=10, num_slices=2, units=10, bits=2)  # capacity 6
    with pytest.raises(ValueError):
        SliceProgram(model(4, 0), total=10, num_slices=2, units=4, bits=2, penalty_margin=1.0)
