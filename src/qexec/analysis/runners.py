"""Single-order runners on a minute-bar DataFrame, all through the same `ExecutionEngine`.

    TWAP      uniform over the minutes
    VWAP      proportional to the *expected* intraday volume profile (no look-ahead)
    SA-QUBO   one SA-solved `slice_level_config` QUBO schedule, repaired to the order size
    Hybrid    starts uniform; at five checkpoints the decision layer may re-solve the
              remaining shares with SA-QUBO over the remaining minutes

Every runner pays the same fill model (a synthetic book whose depth scales with bar
volume, nothing fills in a zero-volume bar, unfilled shares carry forward) and, with the
same `seed`, faces the same book at every minute. The shortfall charges unfilled shares
at the final bar's far touch. This replaces the earlier runners, in which the SA and
hybrid modes filled at mid + spread/2 with no impact and even during outages
(docs/CLAIMS_AUDIT.md F6).
"""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qexec.execution.engine import ExecutionEngine, ExecutionReport, OrderSide, ParentOrder
from qexec.execution.strategies.base import BaseStrategy
from qexec.execution.strategies.fixed import FixedScheduleStrategy
from qexec.execution.strategies.vwap import VWAPStrategy
from qexec.market.simulator import VolumeProfileGenerator
from qexec.optimization.qubo import ExecutionQUBO
from qexec.optimization.schedule import (
    optimize_schedule,
    repair_schedule,
    slice_level_config,
    spread_over_minutes,
)
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
from qexec.runtime.decision import DecisionConfig, MarketState, OptimizationDecisionEngine

NUM_CHECKPOINTS = 5
DEFAULT_DAILY_VOLUME = 50_000_000
_MIN_SHARES_TO_REOPTIMIZE = 1000
_RECENT_BARS = 5
_SA_SLICES = 20
_SA_SWEEPS = 500
_HYBRID_SWEEPS = 300

STRATEGIES = ("TWAP", "VWAP", "SA-QUBO", "Hybrid")


@dataclass
class ExecutionResult:
    """Fills and shortfall of one order; costs positive = worse, in currency or bps."""

    mode: str
    total_shares: int
    executed_shares: int
    avg_price: float
    arrival_price: float
    shortfall: float
    execution_cost: float
    opportunity_cost: float
    shortfall_bps: float
    execution_cost_bps: float
    spread_cost: float
    impact_cost: float
    timing_cost: float
    slippage_vs_vwap_bps: float
    execution_log: list[dict[str, Any]]
    optimization_invocations: int = 0
    optimization_time: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def fill_rate(self) -> float:
        return self.executed_shares / self.total_shares if self.total_shares else 0.0

    def metrics(self) -> dict[str, float | int | str]:
        """Flat row for a results table."""
        return {
            "strategy": self.mode,
            "total_shares": self.total_shares,
            "filled_shares": self.executed_shares,
            "fill_rate": self.fill_rate,
            "shortfall_bps": self.shortfall_bps,
            "execution_cost_bps": self.execution_cost_bps,
            "opportunity_cost_bps": self._bps(self.opportunity_cost),
            "spread_cost_bps": self._bps(self.spread_cost),
            "impact_cost_bps": self._bps(self.impact_cost),
            "timing_cost_bps": self._bps(self.timing_cost),
            "slippage_vs_vwap_bps": self.slippage_vs_vwap_bps,
            "optimization_invocations": self.optimization_invocations,
            "optimization_time_s": self.optimization_time,
            **self.extra,
        }

    def _bps(self, value: float) -> float:
        notional = self.total_shares * self.arrival_price
        return value / notional * 10_000 if notional > 0 else 0.0


def seed_improvement_prior(engine: OptimizationDecisionEngine) -> None:
    """Five synthetic 5% improvements (baseline 100 -> 95): an assumption, not data."""
    for _ in range(5):
        engine.record_outcome(
            baseline_cost=100, optimized_cost=95, order_size=10_000, volatility=0.01
        )


def expected_volume_profile(num_minutes: int, daily_volume: int = DEFAULT_DAILY_VOLUME) -> Any:
    """The simulator's noise-free U-shaped volume curve, in shares per minute."""
    return VolumeProfileGenerator._volume_profile_weights(num_minutes) * daily_volume


class HybridStrategy(BaseStrategy):
    """Uniform start; at `NUM_CHECKPOINTS` evenly spaced minutes the decision layer decides
    whether to re-solve the remaining shares with SA-QUBO over the remaining minutes.

    The decision layer's improvement tracker is seeded with five synthetic 5% improvements
    (`seed_improvement_prior`), so its invocation decisions rest on an assumed prior
    (docs/CLAIMS_AUDIT.md F5). Each re-solve uses max(4, remaining minutes / 3) slices.
    """

    strategy_name = "Hybrid"

    def __init__(self, lambda_tradeoff: float = 0.5, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.seed = seed
        self.decision_engine = OptimizationDecisionEngine(
            DecisionConfig(lambda_tradeoff=lambda_tradeoff, min_order_size=500, max_latency_ms=2000)
        )
        seed_improvement_prior(self.decision_engine)
        self.invocations = 0
        self.optimization_time = 0.0
        self._checkpoints: set[int] = set()

    def calculate_schedule(self, total_shares: int, market_data: pd.DataFrame) -> NDArray[np.int_]:
        n = len(market_data)
        step = max(1, n // NUM_CHECKPOINTS)
        self._checkpoints = {k * step for k in range(NUM_CHECKPOINTS) if k * step < n}
        return repair_schedule(np.ones(n), total_shares)

    def replan(
        self, minute: int, remaining_shares: int, observed: pd.DataFrame, num_minutes: int
    ) -> NDArray[np.float64] | None:
        if minute not in self._checkpoints:
            return None
        recent = observed.iloc[max(0, minute - _RECENT_BARS) :]
        decision = self.decision_engine.decide(
            order_size=remaining_shares,
            market_state=MarketState.from_market_data(recent),
            optimization_latency_ms=500,
        )
        if not decision.invoke_optimization or remaining_shares <= _MIN_SHARES_TO_REOPTIMIZE:
            return None
        start = perf_counter()
        remaining_minutes = num_minutes - minute
        num_slices = min(remaining_minutes, max(4, remaining_minutes // 3))
        qubo = ExecutionQUBO(slice_level_config(remaining_shares, num_slices))
        solver = SimulatedAnnealingSolver(num_sweeps=_HYBRID_SWEEPS, seed=self.seed + minute)
        slice_qty, _ = optimize_schedule(qubo, solver)
        self.optimization_time += perf_counter() - start
        self.invocations += 1
        repaired = repair_schedule(slice_qty, remaining_shares).astype(float)
        return spread_over_minutes(repaired, remaining_minutes)


def sa_qubo_schedule(
    total_shares: int, num_minutes: int, *, num_slices: int = _SA_SLICES, seed: int = 42
) -> tuple[NDArray[np.float64], float, int]:
    """(per-minute schedule, solve time, pre-repair QUBO shares) of one SA-QUBO solve."""
    start = perf_counter()
    num_slices = min(num_slices, num_minutes)
    qubo = ExecutionQUBO(slice_level_config(total_shares, num_slices))
    slice_qty, _ = optimize_schedule(
        qubo, SimulatedAnnealingSolver(num_sweeps=_SA_SWEEPS, seed=seed)
    )
    elapsed = perf_counter() - start
    repaired = repair_schedule(slice_qty, total_shares).astype(float)
    return spread_over_minutes(repaired, num_minutes), elapsed, int(slice_qty.sum())


def make_strategy(
    name: str,
    market_data: pd.DataFrame,
    total_shares: int,
    *,
    seed: int,
    daily_volume: int = DEFAULT_DAILY_VOLUME,
) -> tuple[BaseStrategy, dict[str, float]]:
    """Strategy object for `name` (one of `STRATEGIES`) and solver diagnostics."""
    n = len(market_data)
    if name == "TWAP":
        return FixedScheduleStrategy(np.ones(n)), {}
    if name == "VWAP":
        profile = expected_volume_profile(n, daily_volume)
        return VWAPStrategy(historical_profile=profile, seed=seed), {}
    if name == "SA-QUBO":
        schedule, elapsed, selected = sa_qubo_schedule(total_shares, n, seed=seed)
        return FixedScheduleStrategy(schedule), {
            "optimization_time_s": elapsed,
            "qubo_selected_shares": selected,
        }
    if name == "Hybrid":
        return HybridStrategy(seed=seed), {}
    raise ValueError(f"Unknown strategy {name!r}; expected one of {STRATEGIES}")


def execute_report(
    strategy: BaseStrategy,
    market_data: pd.DataFrame,
    total_shares: int,
    *,
    seed: int,
    side: OrderSide = OrderSide.BUY,
) -> tuple[ExecutionReport, ExecutionEngine]:
    """Run `strategy` through a fresh engine whose book is keyed by `seed`."""
    engine = ExecutionEngine(seed=seed)
    order = ParentOrder(
        symbol=str(market_data["symbol"].iloc[0]) if "symbol" in market_data else "SIM",
        side=side,
        total_quantity=total_shares,
        time_horizon_minutes=len(market_data),
    )
    report = engine.process_order(order, market_data, strategy)
    return report, engine


def run_strategy(
    name: str,
    market_data: pd.DataFrame,
    total_shares: int,
    *,
    seed: int = 42,
    daily_volume: int = DEFAULT_DAILY_VOLUME,
) -> ExecutionResult:
    """Execute a buy of `total_shares` with strategy `name` and return its shortfall."""
    strategy, extra = make_strategy(
        name, market_data, total_shares, seed=seed, daily_volume=daily_volume
    )
    report, engine = execute_report(strategy, market_data, total_shares, seed=seed)
    assert engine.state is not None
    execution_log = [
        {"minute": c.minute_index, "shares": c.filled_quantity, "price": c.execution_price}
        for c in engine.state.child_orders
        if c.filled_quantity > 0
    ]
    # Shortfall decomposition of the fills (buy): crossing = sum n (p - m) splits into the
    # half spread (touch - mid) and impact (walking past the touch); timing = sum n (m - P0).
    filled = [c for c in engine.state.child_orders if c.filled_quantity > 0]
    crossing = sum(
        c.filled_quantity * (c.execution_price - c.market_price_at_execution) for c in filled
    )
    timing = sum(
        c.filled_quantity * (c.market_price_at_execution - report.arrival_price) for c in filled
    )
    invocations, opt_time = 0, float(extra.pop("optimization_time_s", 0.0))
    if isinstance(strategy, HybridStrategy):
        invocations, opt_time = strategy.invocations, strategy.optimization_time
    elif name == "SA-QUBO":
        invocations = 1
    return ExecutionResult(
        mode=name,
        total_shares=total_shares,
        executed_shares=report.filled_quantity,
        avg_price=report.average_execution_price,
        arrival_price=report.arrival_price,
        shortfall=report.implementation_shortfall,
        execution_cost=report.execution_shortfall,
        opportunity_cost=report.opportunity_cost,
        shortfall_bps=report.implementation_shortfall_bps,
        execution_cost_bps=report.slippage_vs_arrival_bps,
        spread_cost=report.spread_cost,
        impact_cost=crossing - report.spread_cost,
        timing_cost=timing,
        slippage_vs_vwap_bps=report.slippage_vs_vwap_bps,
        execution_log=execution_log,
        optimization_invocations=invocations,
        optimization_time=opt_time,
        extra={k: float(v) for k, v in extra.items()},
    )


def run_vwap_execution(
    market_data: pd.DataFrame, total_shares: int, seed: int = 42
) -> ExecutionResult:
    return run_strategy("VWAP", market_data, total_shares, seed=seed)


def run_twap_execution(
    market_data: pd.DataFrame, total_shares: int, seed: int = 42
) -> ExecutionResult:
    return run_strategy("TWAP", market_data, total_shares, seed=seed)


def run_sa_execution(
    market_data: pd.DataFrame, total_shares: int, seed: int = 42
) -> ExecutionResult:
    return run_strategy("SA-QUBO", market_data, total_shares, seed=seed)


def run_hybrid_execution(
    market_data: pd.DataFrame, total_shares: int, seed: int = 42
) -> ExecutionResult:
    return run_strategy("Hybrid", market_data, total_shares, seed=seed)
