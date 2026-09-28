"""Execution QUBO: split an order over time slices, venues and discrete quantity levels.

x_{t,v,k} = 1 means "trade q_k shares in slice t on venue v". The objective is
w_I C_impact + w_T C_timing + w_C C_transaction + P (sum q_k x_{t,v,k} - S)^2
plus a pairwise per-slice capacity penalty; see `ExecutionQUBO.build_qubo_matrix`.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qexec.optimization.solvers.result import BinaryVector

_VENUE_FEE_MULTIPLIERS = (1.0, 0.8)
_VENUE_NAMES = ("Primary", "Dark Pool", "ECN", "Venue 3")
_CROSS_VENUE_IMPACT = 0.3
_ADJACENT_SLICE_SMOOTHING = -0.01
_COEFF_TOL = 1e-10


@dataclass
class QUBOConfig:
    """Problem size, cost weights, market parameters and penalty weights."""

    total_shares: int = 10_000
    num_time_slices: int = 10
    num_venues: int = 2
    quantity_levels: list[int] = field(default_factory=lambda: [0, 500, 1000, 1500, 2000])
    impact_weight: float = 0.4
    timing_weight: float = 0.3
    transaction_weight: float = 0.3
    volatility: float = 0.02
    avg_spread_bps: float = 5.0
    impact_coefficient: float = 0.1
    equality_penalty: float = 1000.0
    capacity_penalty: float = 500.0
    max_shares_per_slice: int = 3000

    @property
    def num_quantity_levels(self) -> int:
        return len(self.quantity_levels)

    @property
    def num_variables(self) -> int:
        """T * V * K."""
        return self.num_time_slices * self.num_venues * self.num_quantity_levels

    def variable_index(self, time: int, venue: int, qty_level: int) -> int:
        """Flat index t*V*K + v*K + k."""
        K = self.num_quantity_levels
        V = self.num_venues
        return time * V * K + venue * K + qty_level

    def decode_index(self, idx: int) -> tuple[int, int, int]:
        """Inverse of `variable_index`: (time slice, venue, quantity level)."""
        K = self.num_quantity_levels
        V = self.num_venues
        time, remainder = divmod(idx, V * K)
        venue, qty_level = divmod(remainder, K)
        return time, venue, qty_level


class ExecutionQUBO:
    """Builds the symmetric QUBO matrix Q (minimise x^T Q x) for a `QUBOConfig`."""

    def __init__(self, config: QUBOConfig) -> None:
        self.config = config
        n = config.num_variables
        self.Q: NDArray[np.float64] | None = None
        self._impact_matrix = np.zeros((n, n))
        self._timing_matrix = np.zeros((n, n))
        self._transaction_matrix = np.zeros((n, n))
        self._constraint_matrix = np.zeros((n, n))

    def build_qubo_matrix(self) -> NDArray[np.float64]:
        """Weighted sum of the cost terms plus unweighted penalties, symmetrised."""
        n = self.config.num_variables
        self._impact_matrix = np.zeros((n, n))
        self._timing_matrix = np.zeros((n, n))
        self._transaction_matrix = np.zeros((n, n))
        self._constraint_matrix = np.zeros((n, n))

        self._add_market_impact_cost()
        self._add_timing_risk_cost()
        self._add_transaction_cost()
        self._add_equality_constraint()
        self._add_capacity_constraint()

        cfg = self.config
        Q = (
            cfg.impact_weight * self._impact_matrix
            + cfg.timing_weight * self._timing_matrix
            + cfg.transaction_weight * self._transaction_matrix
            + self._constraint_matrix
        )
        self.Q = (Q + Q.T) / 2
        return self.Q

    def _add_market_impact_cost(self) -> None:
        """Linear impact eta sigma q on the diagonal, plus same-slice cross-venue impact
        0.3 eta sigma sqrt(q_i q_j) (a linearised Almgren-Chriss temporary impact)."""
        cfg = self.config
        impact_per_share = cfg.impact_coefficient * cfg.volatility
        for t in range(cfg.num_time_slices):
            for v in range(cfg.num_venues):
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q_i = cfg.quantity_levels[k]
                    self._impact_matrix[i, i] += impact_per_share * q_i
                    for v2 in range(cfg.num_venues):
                        if v2 == v:
                            continue
                        for k2 in range(cfg.num_quantity_levels):
                            j = cfg.variable_index(t, v2, k2)
                            q_j = cfg.quantity_levels[k2]
                            self._impact_matrix[i, j] += (
                                _CROSS_VENUE_IMPACT * impact_per_share * (q_i * q_j) ** 0.5
                            )

    def _add_timing_risk_cost(self) -> None:
        """sigma sqrt((t+1)/T) q^2 / S on the diagonal (later, larger slices cost more), and a
        small negative coupling -0.01 sigma min(q_i, q_j) between adjacent slices."""
        cfg = self.config
        for t in range(cfg.num_time_slices):
            time_risk_factor = cfg.volatility * np.sqrt((t + 1) / cfg.num_time_slices)
            for v in range(cfg.num_venues):
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q_i = cfg.quantity_levels[k]
                    self._timing_matrix[i, i] += time_risk_factor * (q_i**2) / cfg.total_shares

        for t in range(cfg.num_time_slices - 1):
            for v in range(cfg.num_venues):
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q_i = cfg.quantity_levels[k]
                    for v2 in range(cfg.num_venues):
                        for k2 in range(cfg.num_quantity_levels):
                            j = cfg.variable_index(t + 1, v2, k2)
                            q_j = cfg.quantity_levels[k2]
                            self._timing_matrix[i, j] += (
                                _ADJACENT_SLICE_SMOOTHING * cfg.volatility * min(q_i, q_j)
                            )

    def _add_transaction_cost(self) -> None:
        """Half-spread per share times a venue fee multiplier (venue 1 is 20% cheaper)."""
        cfg = self.config
        half_spread = (cfg.avg_spread_bps / 10000) / 2
        for t in range(cfg.num_time_slices):
            for v in range(cfg.num_venues):
                multiplier = _VENUE_FEE_MULTIPLIERS[v] if v < len(_VENUE_FEE_MULTIPLIERS) else 1.0
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q = cfg.quantity_levels[k]
                    self._transaction_matrix[i, i] += half_spread * q * multiplier

    def _add_equality_constraint(self) -> None:
        """P (sum_i q_i x_i - S)^2 without the constant P S^2 (x_i^2 = x_i on the diagonal)."""
        cfg = self.config
        P = cfg.equality_penalty
        S = cfg.total_shares
        for i in range(cfg.num_variables):
            q_i = cfg.quantity_levels[cfg.decode_index(i)[2]]
            self._constraint_matrix[i, i] += P * q_i * q_i - 2 * P * S * q_i
            for j in range(i + 1, cfg.num_variables):
                q_j = cfg.quantity_levels[cfg.decode_index(j)[2]]
                self._constraint_matrix[i, j] += 2 * P * q_i * q_j

    def _add_capacity_constraint(self) -> None:
        """Soft cap: P_cap (q_i + q_j - M) for every same-slice pair with q_i + q_j > M."""
        cfg = self.config
        P = cfg.capacity_penalty
        M = cfg.max_shares_per_slice
        for t in range(cfg.num_time_slices):
            vars_at_t = [
                (cfg.variable_index(t, v, k), cfg.quantity_levels[k])
                for v in range(cfg.num_venues)
                for k in range(cfg.num_quantity_levels)
            ]
            for idx1, (i, q_i) in enumerate(vars_at_t):
                for j, q_j in vars_at_t[idx1 + 1 :]:
                    if q_i + q_j > M:
                        self._constraint_matrix[i, j] += P * (q_i + q_j - M)

    def _selected_quantity(self, x: BinaryVector) -> int:
        cfg = self.config
        return sum(
            cfg.quantity_levels[cfg.decode_index(i)[2]] for i, val in enumerate(x) if val > 0.5
        )

    def slice_quantities(self, x: BinaryVector) -> NDArray[np.float64]:
        """Total quantity selected in each time slice (summed over venues and levels)."""
        cfg = self.config
        quantities = np.zeros(cfg.num_time_slices)
        for i in np.flatnonzero(np.asarray(x) > 0.5):
            t, _, k = cfg.decode_index(int(i))
            quantities[t] += cfg.quantity_levels[k]
        return quantities

    def interpret_solution(self, x: BinaryVector) -> pd.DataFrame:
        """Non-zero trades as rows (time_slice, venue, venue_name, quantity), sorted by slice."""
        cfg = self.config
        schedule = []
        for i, val in enumerate(x):
            if val <= 0.5:
                continue
            t, v, k = cfg.decode_index(i)
            q = cfg.quantity_levels[k]
            if q > 0:
                name = _VENUE_NAMES[v] if v < len(_VENUE_NAMES) else f"Venue {v}"
                schedule.append({"time_slice": t, "venue": v, "venue_name": name, "quantity": q})
        df = pd.DataFrame(schedule)
        if len(df) > 0:
            df = df.sort_values("time_slice").reset_index(drop=True)
        return df

    def _built_matrix(self) -> NDArray[np.float64]:
        return self.Q if self.Q is not None else self.build_qubo_matrix()

    def calculate_solution_cost(self, x: BinaryVector) -> dict[str, float]:
        """Total QUBO energy, each weighted cost term, the penalty, and shares selected."""
        Q = self._built_matrix()
        cfg = self.config
        total_shares = self._selected_quantity(x)
        return {
            "total_cost": float(x @ Q @ x),
            "impact_cost": float(x @ self._impact_matrix @ x) * cfg.impact_weight,
            "timing_cost": float(x @ self._timing_matrix @ x) * cfg.timing_weight,
            "transaction_cost": float(x @ self._transaction_matrix @ x) * cfg.transaction_weight,
            "constraint_penalty": float(x @ self._constraint_matrix @ x),
            "total_shares": total_shares,
            "target_shares": cfg.total_shares,
            "shares_difference": total_shares - cfg.total_shares,
        }

    def validate_solution(self, x: BinaryVector) -> dict[str, bool]:
        """Total within 1% of the target, per-slice capacity, and binary entries."""
        cfg = self.config
        shares_ok = abs(self._selected_quantity(x) - cfg.total_shares) < 0.01 * cfg.total_shares
        capacity_ok = bool(np.all(self.slice_quantities(x) <= cfg.max_shares_per_slice))
        binary_ok = all(val in (0, 1) or abs(val - round(val)) < 0.1 for val in x)
        return {
            "total_shares_satisfied": shares_ok,
            "capacity_satisfied": capacity_ok,
            "binary_satisfied": binary_ok,
            "all_satisfied": shares_ok and capacity_ok and binary_ok,
        }

    def get_qubo_dict(self) -> dict[tuple[int, int], float]:
        """Upper-triangular non-zero entries {(i, j): Q_ij}."""
        Q = self._built_matrix()
        n = self.config.num_variables
        return {
            (i, j): float(Q[i, j])
            for i in range(n)
            for j in range(i, n)
            if abs(Q[i, j]) > _COEFF_TOL
        }

    def get_linear_and_quadratic(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(diag(Q), Q with zeroed diagonal)."""
        Q = self._built_matrix()
        quadratic = Q.copy()
        np.fill_diagonal(quadratic, 0)
        return np.diag(Q).copy(), quadratic


def create_random_binary_solution(config: QUBOConfig, seed: int | None = None) -> BinaryVector:
    """One random (venue, level) choice per time slice."""
    rng = np.random.default_rng(seed)
    x = np.zeros(config.num_variables)
    for t in range(config.num_time_slices):
        v = int(rng.integers(0, config.num_venues))
        k = int(rng.integers(0, config.num_quantity_levels))
        x[config.variable_index(t, v, k)] = 1
    return x


def create_uniform_solution(config: QUBOConfig) -> BinaryVector:
    """TWAP-like baseline: on venue 0, the level closest to S/T in every slice."""
    x = np.zeros(config.num_variables)
    shares_per_slice = config.total_shares // config.num_time_slices
    best_k = int(np.argmin([abs(q - shares_per_slice) for q in config.quantity_levels]))
    for t in range(config.num_time_slices):
        x[config.variable_index(t, 0, best_k)] = 1
    return x
