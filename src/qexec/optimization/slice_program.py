"""Exact binary-encoded execution QUBO (Phase 7) for the `CostModel` objective.

The order of N units is cut into U equal *units*; the horizon into T contiguous *slices*.
Slice t trades c_t units, c_t in {0, ..., 2^B - 1}, encoded in base 2 by B bits:

    c_t = sum_b 2^b z_{t,b},     z in {0,1}^{T B}.

Within a slice the units are spread over its minutes in proportion to expected volume
(weights w_k, summing to 1 per slice), so the minute fractions are f = W c / U and the
objective is J(W c / U), a quadratic in c and hence in z. The integer program is

    (IP)   min_c J(W c / U)   s.t.  sum_t c_t = U,  0 <= c_t <= 2^B - 1,  c integer,

and the QUBO is E(z) = J(W E z / U) + P (sum_t c_t - U)^2 (constant terms in `offset`).

Penalty rule (exactness). J >= 0 for every c >= 0 (half spreads, impact and variance are
non-negative), and every infeasible z has (sum c - U)^2 >= 1. With P > J(c_ref) for one
feasible c_ref (the balanced allocation), every infeasible z has E(z) >= P > J(c_ref) >=
min IP, so the QUBO minimisers are exactly the encodings of the IP minimisers. We use
P = penalty_margin * J(c_ref) with penalty_margin = 1.5 (> 1 as the argument requires).

Unlike the Phase 6 level encoding (per-slice indicator variables for a few quantity
levels, which allowed summing two levels in one slice and could not represent most
schedules), every integer allocation with c_t < 2^B is representable, exactly once.

The IP also has a chain structure: slice t's cost depends only on c_t and the units still
unexecuted before it, so `solve_dp` finds the exact IP optimum by dynamic programming in
O(T U 2^B). It is the ground truth for the solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qexec.execution.cost_model import CostModel
from qexec.optimization.schedule import repair_schedule

DEFAULT_PENALTY_MARGIN = 1.5


@dataclass(frozen=True)
class QUBOProblem:
    """E(z) = z^T Q z + offset; `penalty` is the equality-penalty weight used."""

    Q: NDArray[np.float64]
    offset: float
    penalty: float

    def energy(self, z: ArrayLike) -> float:
        x = np.asarray(z, dtype=np.float64)
        return float(x @ self.Q @ x) + self.offset


@dataclass(frozen=True)
class SliceProgram:
    """Integer program over per-slice unit counts and its exact binary QUBO."""

    model: CostModel
    total: int
    num_slices: int
    units: int
    bits: int
    penalty_margin: float = DEFAULT_PENALTY_MARGIN
    _weights: NDArray[np.float64] = field(init=False, repr=False)
    _bounds: tuple[tuple[int, int], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        K = self.model.num_minutes
        if not 1 <= self.num_slices <= K:
            raise ValueError("need 1 <= num_slices <= number of minutes")
        if self.bits < 1 or self.units < 1 or self.total < 1:
            raise ValueError("bits, units and total must be positive")
        if self.units > self.num_slices * self.capacity:
            raise ValueError("units exceed num_slices * (2^bits - 1): infeasible")
        if self.penalty_margin <= 1:
            raise ValueError("penalty_margin must exceed 1 for the encoding to be exact")
        bounds = tuple(
            (int(idx[0]), int(idx[-1]) + 1) for idx in np.array_split(np.arange(K), self.num_slices)
        )
        W = np.zeros((K, self.num_slices))
        for t, (a, b) in enumerate(bounds):
            v = self.model.expected_volume[a:b]
            W[a:b, t] = v / v.sum()
        object.__setattr__(self, "_bounds", bounds)
        object.__setattr__(self, "_weights", W)

    # -- structure ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Largest representable count per slice, 2^B - 1."""
        return int(2**self.bits - 1)

    @property
    def num_variables(self) -> int:
        return self.num_slices * self.bits

    @property
    def slice_bounds(self) -> tuple[tuple[int, int], ...]:
        """(first minute, one past the last minute) of each slice."""
        return self._bounds

    @property
    def weights(self) -> NDArray[np.float64]:
        """Minutes x slices matrix W: f = W c / U."""
        return self._weights

    def encoding_matrix(self) -> NDArray[np.float64]:
        """E (slices x variables) with c = E z; bit b of slice t is variable t B + b."""
        E = np.zeros((self.num_slices, self.num_variables))
        for t in range(self.num_slices):
            E[t, t * self.bits : (t + 1) * self.bits] = 2.0 ** np.arange(self.bits)
        return E

    def decode(self, z: ArrayLike) -> NDArray[np.int_]:
        x = np.asarray(z)
        if x.shape != (self.num_variables,):
            raise ValueError(f"expected {self.num_variables} bits")
        bits = (x > 0.5).reshape(self.num_slices, self.bits).astype(np.int_)
        return np.asarray(bits @ (2 ** np.arange(self.bits)), dtype=np.int_)

    def encode(self, counts: ArrayLike) -> NDArray[np.int8]:
        c = np.asarray(counts, dtype=np.int_)
        if c.shape != (self.num_slices,) or np.any(c < 0) or np.any(c > self.capacity):
            raise ValueError("counts out of the encodable range")
        return np.asarray(((c[:, None] >> np.arange(self.bits)) & 1).ravel(), dtype=np.int8)

    def is_feasible(self, counts: ArrayLike) -> bool:
        c = np.asarray(counts)
        return bool(c.sum() == self.units and np.all(c >= 0) and np.all(c <= self.capacity))

    # -- objective ------------------------------------------------------------------

    def fractions(self, counts: ArrayLike) -> NDArray[np.float64]:
        """Per-minute fractions of the order, W c / U."""
        return np.asarray(self._weights @ np.asarray(counts, dtype=np.float64) / self.units)

    def objective(self, counts: ArrayLike) -> float:
        """J (bps) of the allocation; defined for infeasible counts too."""
        A, b, c0 = self.model.quadratic_form(self.total)
        f = self.fractions(counts)
        return float(f @ A @ f + b @ f + c0)

    def reference_counts(self) -> NDArray[np.int_]:
        """Balanced feasible allocation (largest remainder of U / T)."""
        return np.asarray(repair_schedule(np.ones(self.num_slices), self.units), dtype=np.int_)

    def penalty_weight(self) -> float:
        return self.penalty_margin * self.objective(self.reference_counts())

    def qubo(self) -> QUBOProblem:
        """Exact QUBO; its minimisers encode the IP minimisers (see module docstring)."""
        A, b, c0 = self.model.quadratic_form(self.total)
        M = self._weights @ self.encoding_matrix() / self.units  # f = M z
        P = self.penalty_weight()
        ones = self.encoding_matrix().sum(axis=0)  # sum_t c_t = ones @ z
        Q = M.T @ A @ M + P * np.outer(ones, ones)
        linear = M.T @ b - 2 * P * self.units * ones
        Q[np.diag_indices_from(Q)] += linear  # z_i^2 = z_i
        return QUBOProblem(Q=(Q + Q.T) / 2, offset=c0 + P * self.units**2, penalty=P)

    # -- schedules ------------------------------------------------------------------

    def schedule(self, counts: ArrayLike) -> NDArray[np.int_]:
        """Integer units per minute summing to `total` (off-constraint counts rescaled)."""
        return repair_schedule(self.fractions(counts) * self.total, self.total)

    # -- exact solutions ------------------------------------------------------------

    def _slice_cost(self, t: int, remaining: int, count: int) -> float:
        """Cost of slice t trading `count` units with `remaining` units unexecuted."""
        a, b = self._bounds[t]
        m = self.model
        w = self._weights[a:b, t]
        f = count * w / self.units
        before = np.concatenate(([0.0], np.cumsum(w)[:-1]))
        r = (remaining - count * before) / self.units
        spread_impact = f @ m.half_spread_bps[a:b] + np.sum(
            m.impact_bps * self.total * f**2 / m.expected_volume[a:b]
        )
        risk = m.risk_aversion * np.sum(m.sigma_bps[a:b] ** 2 * r**2)
        return float(spread_impact + risk)

    def solve_dp(self) -> tuple[NDArray[np.int_], float]:
        """Exact IP optimum (counts, J) by dynamic programming over remaining units."""
        T, U, C = self.num_slices, self.units, self.capacity
        value = np.full((T + 1, U + 1), np.inf)
        choice = np.zeros((T, U + 1), dtype=np.int_)
        value[T, 0] = 0.0
        for t in range(T - 1, -1, -1):
            for remaining in range(U + 1):
                best, arg = np.inf, 0
                for count in range(min(remaining, C) + 1):
                    tail = value[t + 1, remaining - count]
                    if not np.isfinite(tail):
                        continue
                    cost = self._slice_cost(t, remaining, count) + tail
                    if cost < best:
                        best, arg = cost, count
                value[t, remaining], choice[t, remaining] = best, arg
        counts = np.zeros(T, dtype=np.int_)
        remaining = U
        for t in range(T):
            counts[t] = choice[t, remaining]
            remaining -= counts[t]
        return counts, float(value[0, U])

    def solve_enumeration(self) -> tuple[NDArray[np.int_], float]:
        """Exact IP optimum by enumerating every feasible allocation (small problems)."""
        best_c, best = np.zeros(self.num_slices, dtype=np.int_), np.inf
        for head in product(range(self.capacity + 1), repeat=self.num_slices - 1):
            last = self.units - sum(head)
            if not 0 <= last <= self.capacity:
                continue
            c = np.array((*head, last), dtype=np.int_)
            value = self.objective(c)
            if value < best:
                best_c, best = c, value
        return best_c, best
