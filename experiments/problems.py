"""QUBO instance families shared by the solver and QAOA benchmarks.

toy        `toy_execution_qubo(n)`: the problem of the IBM hardware runs (claims audit F3)
execution  `ExecutionQUBO(slice_level_config(100 T, T))`: T slices x levels {0, 50, 100}
random     symmetric Gaussian Q, (A + A^T)/2 with A_ij ~ N(0, 1); one instance per seed
slice      Phase 7 exact binary encoding (`SliceProgram`) of the real-data cost model on a
           synthetic 30-minute window: T slices x B bits, n = T B (`SLICE_SHAPES`)
"""

import numpy as np
from numpy.typing import NDArray

from qexec.execution.cost_model import CostModel
from qexec.optimization.qubo import ExecutionQUBO, QUBOConfig
from qexec.optimization.schedule import slice_level_config
from qexec.optimization.slice_program import SliceProgram
from qexec.optimization.toy import toy_execution_qubo

FAMILIES = ("toy", "execution", "random", "slice")

# n -> (slices T, bits B); units U = T (2^B - 1) // 2 (half the capacity).
SLICE_SHAPES = {
    4: (2, 2),
    6: (2, 3),
    8: (4, 2),
    9: (3, 3),
    10: (5, 2),
    12: (4, 3),
    15: (5, 3),
    16: (4, 4),
    18: (6, 3),
    20: (5, 4),
}
_SLICE_MINUTES = 30


def slice_program(n: int) -> SliceProgram:
    """Synthetic 30-minute window: U-shaped volume, spread rising at the edges, 4 bps/min
    volatility, impact 1.5 bps per unit participation (the order of the dev-day BTCUSDT
    fit), an order of 10% of window volume, and lambda set so that lambda Var = E for
    TWAP (the protocol's secondary risk-aversion rule)."""
    if n not in SLICE_SHAPES:
        raise ValueError(f"slice family sizes are {sorted(SLICE_SHAPES)}")
    slices, bits = SLICE_SHAPES[n]
    x = np.linspace(0.0, 1.0, _SLICE_MINUTES)
    volume = 1000.0 * (1.0 + 4.0 * (x - 0.5) ** 2)
    model = CostModel(
        expected_volume=volume,
        half_spread_bps=0.5 + 0.5 * np.abs(x - 0.5),
        sigma_bps=np.full(_SLICE_MINUTES, 4.0),
        impact_bps=1.5,
    )
    total = int(0.1 * volume.sum())
    twap = np.full(_SLICE_MINUTES, total / _SLICE_MINUTES)
    lam = model.expected_cost_bps(twap, total) / model.variance_bps2(twap, total)
    units = slices * (2**bits - 1) // 2
    return SliceProgram(model.with_risk_aversion(lam), total, slices, units, bits)


def execution_qubo(n: int) -> ExecutionQUBO:
    if n % 3:
        raise ValueError("execution family sizes are multiples of 3 (T slices x 3 levels)")
    slices = n // 3
    return ExecutionQUBO(slice_level_config(100 * slices, slices))


def fig07_qubo() -> ExecutionQUBO:
    """The 12-variable instance of the old qaoa_vs_sa script: 400 shares, 4 slices x
    levels {0, 100, 200}, equality penalty 100, capacity penalty 50."""
    return ExecutionQUBO(
        QUBOConfig(
            total_shares=400,
            num_time_slices=4,
            num_venues=1,
            quantity_levels=[0, 100, 200],
            equality_penalty=100.0,
            capacity_penalty=50.0,
        )
    )


def instance(family: str, n: int, seed: int) -> NDArray[np.float64]:
    """QUBO matrix; only the random family depends on `seed`."""
    if family == "toy":
        return toy_execution_qubo(n)
    if family == "execution":
        return execution_qubo(n).build_qubo_matrix()
    if family == "fig07":
        return fig07_qubo().build_qubo_matrix()
    if family == "slice":
        return slice_program(n).qubo().Q
    if family == "random":
        A = np.random.default_rng([seed, n]).standard_normal((n, n))
        return np.asarray((A + A.T) / 2)
    raise ValueError(f"unknown family {family!r}")
