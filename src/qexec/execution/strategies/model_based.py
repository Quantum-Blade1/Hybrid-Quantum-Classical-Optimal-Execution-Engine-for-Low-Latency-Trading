"""Schedules that minimise the execution `CostModel`: discretized Almgren-Chriss, the
binary-encoded QUBO solved by simulated annealing, and an adaptive (hybrid) QUBO strategy
that re-plans the remainder of the order from what it has observed.

Settings (`QUBOSettings`) were tuned on development data only (docs/PROTOCOL.md): 5
slices x 3 bits (n = 15 variables), 20 units, SA with 1,000 sweeps x 16 restarts, which
reached the exact integer optimum (dynamic programming) on every development window.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qexec.execution.cost_model import CostModel
from qexec.execution.strategies.base import BaseStrategy
from qexec.optimization.schedule import repair_schedule
from qexec.optimization.slice_program import SliceProgram
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver

_MIN_RETURNS_FOR_SIGMA = 5


@dataclass(frozen=True)
class QUBOSettings:
    num_slices: int = 5
    bits: int = 3
    units: int = 20
    sweeps: int = 1000
    restarts: int = 16


@dataclass(frozen=True)
class QUBOPlan:
    """A decoded QUBO solution. `objective_bps` is J of the decoded integer schedule and
    `optimal_objective_bps` J of the exact integer-program optimum (DP), rounded the same way."""

    schedule: NDArray[np.int_]
    counts: NDArray[np.int_]
    feasible: bool
    objective_bps: float
    optimal_objective_bps: float
    solve_time_s: float

    @property
    def gap_to_ip_optimum_bps(self) -> float:
        return self.objective_bps - self.optimal_objective_bps


def ac_schedule(model: CostModel, total: int) -> NDArray[np.int_]:
    """Integer schedule of the exact continuous optimum (discretized Almgren-Chriss)."""
    return repair_schedule(model.optimal_fractions(total) * total, total)


def proportional_schedule(weights: NDArray[np.float64], total: int) -> NDArray[np.int_]:
    return repair_schedule(weights, total)


def slice_program(model: CostModel, total: int, settings: QUBOSettings) -> SliceProgram:
    num_slices = min(settings.num_slices, model.num_minutes)
    units = min(settings.units, num_slices * (2**settings.bits - 1))
    return SliceProgram(model, total, num_slices, units, settings.bits)


def qubo_plan(model: CostModel, total: int, settings: QUBOSettings, seed: int) -> QUBOPlan:
    """Solve the binary QUBO of `model` with SA and decode it to a per-minute schedule."""
    program = slice_program(model, total, settings)
    start = perf_counter()
    result = SimulatedAnnealingSolver(
        num_sweeps=settings.sweeps, num_restarts=settings.restarts, seed=seed
    ).solve(program.qubo().Q)
    elapsed = perf_counter() - start
    counts = program.decode(result.solution)
    schedule = program.schedule(counts)
    dp_counts, _ = program.solve_dp()
    optimum = model.objective(program.schedule(dp_counts), total)
    return QUBOPlan(
        schedule=schedule,
        counts=counts,
        feasible=program.is_feasible(counts),
        objective_bps=model.objective(schedule, total),
        optimal_objective_bps=optimum,
        solve_time_s=elapsed,
    )


def dp_schedule(model: CostModel, total: int, settings: QUBOSettings) -> NDArray[np.int_]:
    """Exact integer-program optimum of the same slice program (diagnostic)."""
    program = slice_program(model, total, settings)
    counts, _ = program.solve_dp()
    return program.schedule(counts)


def _clipped_ratio(num: float, den: float, clip: tuple[float, float]) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return 1.0
    return float(np.clip(num / den, *clip))


class AdaptiveQUBOStrategy(BaseStrategy):
    """Hybrid: start from the QUBO plan; at `num_checkpoints` evenly spaced minutes re-solve
    the remaining units over the remaining minutes with the QUBO/SA, after rescaling the
    development profiles by what this order has observed so far.

    Observed bars are those strictly before the current minute (the current bar's volume
    and spread are not known before trading in it). Required columns: `volume`,
    `expected_volume`, `half_spread_obs` (NaN when not estimated), `close`. Ratios
    observed / expected of volume, half spread and volatility are clipped to `clip`.
    """

    strategy_name = "Hybrid"

    def __init__(
        self,
        model: CostModel,
        settings: QUBOSettings,
        *,
        seed: int,
        num_checkpoints: int = 3,
        clip: tuple[float, float] = (0.5, 2.0),
    ) -> None:
        super().__init__(seed=seed)
        self.model = model
        self.settings = settings
        self.seed = seed
        self.num_checkpoints = num_checkpoints
        self.clip = clip
        self.invocations = 0
        self.optimization_time = 0.0
        self.infeasible_solutions = 0
        self._checkpoints: set[int] = set()

    def calculate_schedule(self, total_shares: int, market_data: pd.DataFrame) -> NDArray[np.int_]:
        n = len(market_data)
        step = n / (self.num_checkpoints + 1)
        self._checkpoints = {round(k * step) for k in range(1, self.num_checkpoints + 1)}
        plan = self._solve(self.model, total_shares, self.seed)
        return plan.schedule

    def _solve(self, model: CostModel, total: int, seed: int) -> QUBOPlan:
        plan = qubo_plan(model, total, self.settings, seed)
        self.invocations += 1
        self.optimization_time += plan.solve_time_s
        self.infeasible_solutions += int(not plan.feasible)
        return plan

    def updated_model(self, minute: int, observed: pd.DataFrame) -> CostModel:
        """Development model of minutes `minute..` rescaled by the observed ratios."""
        past = observed.iloc[:minute]
        volume_ratio = _clipped_ratio(
            float(past["volume"].sum()), float(past["expected_volume"].sum()), self.clip
        )
        seen = past["half_spread_obs"].notna().to_numpy()
        spread_ratio = _clipped_ratio(
            float(past["half_spread_obs"].to_numpy()[seen].mean()) if seen.any() else np.nan,
            float(self.model.half_spread_bps[:minute][seen].mean()) if seen.any() else np.nan,
            self.clip,
        )
        returns = np.diff(np.log(past["close"].to_numpy(dtype=np.float64))) * 1e4
        sigma_ratio = 1.0
        if returns.size >= _MIN_RETURNS_FOR_SIGMA:
            realized = float(np.sqrt(np.mean(returns**2)))
            expected = float(np.sqrt(np.mean(self.model.sigma_bps[1:minute] ** 2)))
            sigma_ratio = _clipped_ratio(realized, expected, self.clip)
        rest = self.model.window(minute)
        return CostModel(
            expected_volume=rest.expected_volume * volume_ratio,
            half_spread_bps=rest.half_spread_bps * spread_ratio,
            sigma_bps=rest.sigma_bps * sigma_ratio,
            impact_bps=rest.impact_bps,
            risk_aversion=rest.risk_aversion,
        )

    def replan(
        self, minute: int, remaining_shares: int, observed: pd.DataFrame, num_minutes: int
    ) -> NDArray[np.float64] | None:
        if minute not in self._checkpoints or remaining_shares <= 0:
            return None
        model = self.updated_model(minute, observed)
        plan = self._solve(model, remaining_shares, self.seed + minute)
        return plan.schedule.astype(np.float64)
