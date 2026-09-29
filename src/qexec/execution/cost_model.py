"""Expected cost and risk of a per-minute schedule, and its exact continuous optimum.

This is the model the Phase 7 optimisers minimise (AC, the binary-encoded QUBO, the hybrid)
and the expected value of what `ImpactFillModel` charges on real bars. For an order of N
units split as q_k over minutes k = 0..K-1, with fractions f_k = q_k / N and remaining
fraction r_k = 1 - sum_{j<k} f_j before minute k (docs/MATHEMATICAL_MODEL.md):

    E[IS]   = sum_k f_k (h_k + beta N f_k / V_k)                    bps of arrival notional
    Var[IS] = sum_k sigma_k^2 r_k^2                                 bps^2
    J(f)    = E[IS] + lambda Var[IS]

h_k is the half spread (bps), V_k the expected volume (units), sigma_k the per-minute
return volatility (bps) and beta the impact in bps per unit participation. J is a convex
quadratic f^T A f + b^T f + c; `optimal_fractions` returns its exact minimiser over the
simplex (discretized Almgren-Chriss with time-varying inputs).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

_BISECTION_STEPS = 200
_KKT_TOL = 1e-9


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return arr


@dataclass(frozen=True)
class CostModel:
    """Per-minute market inputs of the execution cost model."""

    expected_volume: NDArray[np.float64]
    half_spread_bps: NDArray[np.float64]
    sigma_bps: NDArray[np.float64]
    impact_bps: float
    risk_aversion: float = 0.0

    def __post_init__(self) -> None:
        volume = _vector(self.expected_volume, "expected_volume")
        spread = _vector(self.half_spread_bps, "half_spread_bps")
        sigma = _vector(self.sigma_bps, "sigma_bps")
        if not volume.size == spread.size == sigma.size:
            raise ValueError("per-minute inputs must have the same length")
        if np.any(volume <= 0):
            raise ValueError("expected_volume must be positive")
        if np.any(spread < 0) or np.any(sigma < 0):
            raise ValueError("half spread and sigma must be non-negative")
        if self.impact_bps <= 0:
            raise ValueError("impact_bps must be positive")
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative")
        object.__setattr__(self, "expected_volume", volume)
        object.__setattr__(self, "half_spread_bps", spread)
        object.__setattr__(self, "sigma_bps", sigma)

    @property
    def num_minutes(self) -> int:
        return int(self.expected_volume.size)

    def window(self, start: int, stop: int | None = None) -> CostModel:
        """The model restricted to minutes start..stop-1."""
        sl = slice(start, stop)
        return replace(
            self,
            expected_volume=self.expected_volume[sl],
            half_spread_bps=self.half_spread_bps[sl],
            sigma_bps=self.sigma_bps[sl],
        )

    def with_risk_aversion(self, risk_aversion: float) -> CostModel:
        return replace(self, risk_aversion=risk_aversion)

    def with_impact(self, impact_bps: float) -> CostModel:
        return replace(self, impact_bps=impact_bps)

    # -- cost of a schedule ---------------------------------------------------------------

    def _fractions(self, schedule: ArrayLike, total: float) -> NDArray[np.float64]:
        q = np.asarray(schedule, dtype=np.float64)
        if q.shape != (self.num_minutes,):
            raise ValueError(f"schedule must have {self.num_minutes} entries")
        if total <= 0:
            raise ValueError("total must be positive")
        return q / total

    def expected_cost_bps(self, schedule: ArrayLike, total: float) -> float:
        """Expected spread + impact cost of `schedule` (units per minute) for an order of
        `total` units, in bps of arrival notional."""
        f = self._fractions(schedule, total)
        impact = self.impact_bps * total * f**2 / self.expected_volume
        return float(f @ self.half_spread_bps + impact.sum())

    def variance_bps2(self, schedule: ArrayLike, total: float) -> float:
        """Variance of the timing cost, bps^2."""
        f = self._fractions(schedule, total)
        remaining = 1.0 - np.concatenate(([0.0], np.cumsum(f)[:-1]))
        return float(np.sum(self.sigma_bps**2 * remaining**2))

    def objective(self, schedule: ArrayLike, total: float) -> float:
        return self.expected_cost_bps(schedule, total) + self.risk_aversion * self.variance_bps2(
            schedule, total
        )

    def quadratic_form(
        self, total: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """(A, b, c) with J(f) = f^T A f + b^T f + c for fractions f (A symmetric)."""
        K = self.num_minutes
        lower = np.tril(np.ones((K, K)), k=-1)  # r = 1 - lower @ f
        s2 = self.sigma_bps**2
        lam = self.risk_aversion
        A = np.diag(self.impact_bps * total / self.expected_volume) + lam * (lower.T * s2) @ lower
        b = self.half_spread_bps - 2 * lam * lower.T @ s2
        return (A + A.T) / 2, b, float(lam * s2.sum())

    # -- exact optimum ----------------------------------------------------------------

    def _water_filling(self, total: float) -> NDArray[np.float64]:
        """Exact minimiser for lambda = 0: f_k = max(0, (nu - h_k) V_k / (2 beta N))."""
        h = self.half_spread_bps
        scale = self.expected_volume / (2 * self.impact_bps * total)

        def mass(nu: float) -> float:
            return float(np.sum(np.maximum(0.0, nu - h) * scale))

        lo, hi = float(h.min()), float(h.max()) + 1.0 / float(scale.min())
        while mass(hi) < 1:
            hi *= 2
        for _ in range(_BISECTION_STEPS):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mass(mid) < 1 else (lo, mid)
        f = np.maximum(0.0, hi - h) * scale
        return np.asarray(f / f.sum(), dtype=np.float64)

    def optimal_fractions(self, total: float) -> NDArray[np.float64]:
        """Minimiser of J over {f >= 0, sum f = 1}: water-filling when lambda = 0, else a
        convex QP solved by SLSQP from the water-filling start."""
        start = self._water_filling(total)
        if self.risk_aversion == 0:
            return start
        A, b, _ = self.quadratic_form(total)
        result = minimize(
            lambda f: float(f @ A @ f + b @ f),
            start,
            jac=lambda f: 2 * A @ f + b,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * self.num_minutes,
            constraints=[{"type": "eq", "fun": lambda f: f.sum() - 1.0, "jac": np.ones_like}],
            options={"maxiter": 1000, "ftol": 1e-14},
        )
        f = np.clip(result.x, 0.0, None)
        return np.asarray(f / f.sum(), dtype=np.float64)

    def kkt_residual(self, fractions: ArrayLike, total: float) -> float:
        """Largest violation of the KKT conditions of min J on the simplex (0 = optimal).

        At the optimum the gradient equals a common multiplier nu on the support and is
        >= nu off it.
        """
        f = np.asarray(fractions, dtype=np.float64)
        A, b, _ = self.quadratic_form(total)
        grad = 2 * A @ f + b
        support = f > _KKT_TOL
        nu = float(np.mean(grad[support]))
        on = np.abs(grad[support] - nu)
        off = np.maximum(0.0, nu - grad[~support])
        return float(max(on.max(initial=0.0), off.max(initial=0.0), abs(f.sum() - 1)))
