"""Single-order runners on a minute-bar DataFrame, each returning an `ExecutionResult`.

    run_vwap_execution    VWAPStrategy through ExecutionEngine (order book with impact)
    run_sa_execution      one SA-QUBO schedule filled at mid + spread/2
    run_hybrid_execution  decision layer re-optimises the remaining order with SA-QUBO at
                          five checkpoints; fills at mid + spread/2

The SA and hybrid runners have no impact model, so their costs are not comparable
with the VWAP runner's (docs/CLAIMS_AUDIT.md F6). Costs are measured against the
market VWAP on the executed shares.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from qexec.execution.engine import ExecutionEngine, OrderSide, ParentOrder
from qexec.execution.strategies.vwap import VWAPStrategy
from qexec.market.simulator import calculate_vwap
from qexec.optimization.qubo import ExecutionQUBO
from qexec.optimization.schedule import optimize_schedule, slice_level_config, spread_over_minutes
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
from qexec.runtime.decision import DecisionConfig, MarketState, OptimizationDecisionEngine

NUM_CHECKPOINTS = 5
_MIN_SHARES_TO_REOPTIMIZE = 1000
_RECENT_BARS = 5


@dataclass
class ExecutionResult:
    mode: str
    total_shares: int
    executed_shares: int
    avg_price: float
    total_cost: float
    slippage_bps: float
    execution_log: list[dict[str, Any]]
    optimization_invocations: int = 0
    optimization_time: float = 0.0


def _half_spread_fill_result(
    *,
    mode: str,
    market_data: pd.DataFrame,
    total_shares: int,
    execution_log: list[dict[str, Any]],
    invocations: int,
    opt_time: float,
) -> ExecutionResult:
    executed = sum(e["shares"] for e in execution_log)
    value = sum(e["shares"] * e["price"] for e in execution_log)
    avg_price = value / executed if executed > 0 else 0.0
    benchmark = calculate_vwap(market_data)
    slippage_bps = (avg_price - benchmark) / benchmark * 10_000 if benchmark > 0 else 0.0
    return ExecutionResult(
        mode=mode,
        total_shares=total_shares,
        executed_shares=executed,
        avg_price=avg_price,
        total_cost=value - executed * benchmark,
        slippage_bps=slippage_bps,
        execution_log=execution_log,
        optimization_invocations=invocations,
        optimization_time=opt_time,
    )


def _fill(market_data: pd.DataFrame, minute: int, shares: int) -> dict[str, Any]:
    row = market_data.iloc[minute]
    return {"minute": minute, "shares": shares, "price": row["price"] + row["spread"] / 2}


def run_vwap_execution(
    market_data: pd.DataFrame, total_shares: int, seed: int = 42
) -> ExecutionResult:
    """VWAP through the engine; the log holds the engine's actual child-order fills."""
    engine = ExecutionEngine(seed=seed)
    order = ParentOrder(
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=total_shares,
        time_horizon_minutes=len(market_data),
    )
    report = engine.process_order(
        order, market_data, VWAPStrategy(participation_rate=0.1, seed=seed)
    )
    assert engine.state is not None
    execution_log = [
        {"minute": c.minute_index, "shares": c.filled_quantity, "price": c.execution_price}
        for c in engine.state.child_orders
        if c.filled_quantity > 0
    ]
    return ExecutionResult(
        mode="VWAP",
        total_shares=total_shares,
        executed_shares=report.filled_quantity,
        avg_price=report.average_execution_price,
        total_cost=report.total_cost,
        slippage_bps=report.slippage_vs_vwap_bps,
        execution_log=execution_log,
    )


def run_sa_execution(
    market_data: pd.DataFrame, total_shares: int, num_slices: int = 20, seed: int = 42
) -> ExecutionResult:
    """One SA-QUBO solve, spread over the minutes and filled at the ask."""
    start = perf_counter()
    qubo = ExecutionQUBO(slice_level_config(total_shares, num_slices))
    slice_qty, _ = optimize_schedule(qubo, SimulatedAnnealingSolver(num_sweeps=500, seed=seed))
    opt_time = perf_counter() - start

    schedule = spread_over_minutes(slice_qty, len(market_data))
    execution_log = [
        _fill(market_data, minute, int(schedule[minute]))
        for minute in range(len(market_data))
        if int(schedule[minute]) > 0
    ]
    return _half_spread_fill_result(
        mode="SA-Optimized",
        market_data=market_data,
        total_shares=total_shares,
        execution_log=execution_log,
        invocations=1,
        opt_time=opt_time,
    )


def run_hybrid_execution(
    market_data: pd.DataFrame,
    total_shares: int,
    lambda_tradeoff: float = 0.5,
    seed: int = 42,
) -> ExecutionResult:
    """Start uniform; at each of five checkpoints the decision layer may re-solve the rest.

    The improvement tracker is seeded with five synthetic 5% improvements, so the
    invocation decisions rest on an assumed prior (docs/CLAIMS_AUDIT.md F5). Each re-solve
    uses max(4, remaining minutes / 3) slices.
    """
    decision_engine = OptimizationDecisionEngine(
        DecisionConfig(lambda_tradeoff=lambda_tradeoff, min_order_size=500, max_latency_ms=2000)
    )
    for _ in range(5):
        decision_engine.record_outcome(
            baseline_cost=100, optimized_cost=95, order_size=10_000, volatility=0.01
        )

    num_minutes = len(market_data)
    remaining_shares = total_shares
    minutes_per_check = num_minutes // NUM_CHECKPOINTS
    current_schedule = np.full(num_minutes, total_shares // num_minutes)
    execution_log: list[dict[str, Any]] = []
    invocations = 0
    total_opt_time = 0.0

    for check_point in range(NUM_CHECKPOINTS):
        start_minute = check_point * minutes_per_check
        end_minute = min((check_point + 1) * minutes_per_check, num_minutes)
        recent = market_data.iloc[max(0, start_minute - _RECENT_BARS) : start_minute + 1]
        decision = decision_engine.decide(
            order_size=remaining_shares,
            market_state=MarketState.from_market_data(recent),
            optimization_latency_ms=500,
        )

        if decision.invoke_optimization and remaining_shares > _MIN_SHARES_TO_REOPTIMIZE:
            opt_start = perf_counter()
            remaining_minutes = num_minutes - start_minute
            qubo = ExecutionQUBO(
                slice_level_config(remaining_shares, max(4, remaining_minutes // 3))
            )
            solver = SimulatedAnnealingSolver(num_sweeps=300, seed=seed + check_point)
            slice_qty, _ = optimize_schedule(qubo, solver)
            total_opt_time += perf_counter() - opt_start
            invocations += 1
            new_schedule = spread_over_minutes(slice_qty, remaining_minutes)
            current_schedule[start_minute:] = 0
            current_schedule[start_minute : start_minute + len(new_schedule)] = new_schedule

        for minute in range(start_minute, end_minute):
            shares = min(int(current_schedule[minute]), remaining_shares)
            if shares > 0:
                execution_log.append(_fill(market_data, minute, shares))
                remaining_shares -= shares

    return _half_spread_fill_result(
        mode="Hybrid-Quantum",
        market_data=market_data,
        total_shares=total_shares,
        execution_log=execution_log,
        invocations=invocations,
        opt_time=total_opt_time,
    )
