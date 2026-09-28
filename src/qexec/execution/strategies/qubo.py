"""QUBO-scheduled execution strategy and a VWAP/TWAP/QUBO comparison through the engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qexec.execution.engine import ExecutionEngine, ExecutionReport, ParentOrder
from qexec.execution.strategies.base import BaseStrategy
from qexec.execution.strategies.twap import TWAPStrategy
from qexec.execution.strategies.vwap import VWAPStrategy
from qexec.market.order_book import OrderBook
from qexec.optimization.qubo import ExecutionQUBO, QUBOConfig
from qexec.optimization.schedule import optimize_schedule, spread_over_minutes
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
from qexec.optimization.solvers.result import QUBOResult

logger = logging.getLogger(__name__)


def default_quantity_levels(total_shares: int, num_time_slices: int) -> list[int]:
    """Levels {0, 1, 2, 3, 4} x N/(2T)."""
    base_qty = total_shares // (num_time_slices * 2)
    return [0, base_qty, base_qty * 2, base_qty * 3, base_qty * 4]


class QUBOStrategy(BaseStrategy):
    """Schedule from an `ExecutionQUBO` solved by simulated annealing, spread over minutes."""

    strategy_name = "QUBO"

    def __init__(
        self,
        num_time_slices: int = 20,
        *,
        num_venues: int = 1,
        quantity_levels: list[int] | None = None,
        sa_sweeps: int = 1000,
        order_book: OrderBook | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(order_book=order_book, seed=seed)
        self.num_time_slices = num_time_slices
        self.num_venues = num_venues
        self.quantity_levels = quantity_levels
        self.sa_sweeps = sa_sweeps
        self.seed = seed

        self.qubo: ExecutionQUBO | None = None
        self.qubo_result: QUBOResult | None = None
        self.optimization_time: float = 0.0

    def calculate_schedule(self, total_shares: int, market_data: pd.DataFrame) -> NDArray[np.int_]:
        start = perf_counter()
        levels = self.quantity_levels or default_quantity_levels(total_shares, self.num_time_slices)
        config = QUBOConfig(
            total_shares=total_shares,
            num_time_slices=self.num_time_slices,
            num_venues=self.num_venues,
            quantity_levels=levels,
            impact_weight=0.4,
            timing_weight=0.3,
            transaction_weight=0.3,
            equality_penalty=1000.0,
            capacity_penalty=500.0,
            max_shares_per_slice=total_shares // 3,
        )
        self.qubo = ExecutionQUBO(config)
        solver = SimulatedAnnealingSolver(
            num_sweeps=self.sa_sweeps,
            initial_temp=10.0,
            final_temp=0.01,
            cooling_rate=0.95,
            seed=self.seed,
        )
        slice_qty, self.qubo_result = optimize_schedule(self.qubo, solver)
        self.optimization_time = perf_counter() - start
        return spread_over_minutes(slice_qty, len(market_data)).astype(int)

    def get_optimization_stats(self) -> dict[str, Any]:
        """Energy, solver effort, constraint check and cost breakdown of the last solve."""
        if self.qubo is None or self.qubo_result is None:
            return {}
        solution = self.qubo_result.solution
        return {
            "qubo_energy": self.qubo_result.energy,
            "sa_evaluations": self.qubo_result.num_evaluations,
            "sa_iterations": self.qubo_result.iterations,
            "optimization_time_s": self.optimization_time,
            "constraints_satisfied": self.qubo.validate_solution(solution)["all_satisfied"],
            **self.qubo.calculate_solution_cost(solution),
        }


@dataclass
class StrategyComparison:
    """Engine reports of VWAP, TWAP and QUBO on the same order and market data."""

    vwap_report: ExecutionReport
    twap_report: ExecutionReport
    qubo_report: ExecutionReport
    qubo_optimization_time: float
    qubo_energy: float
    qubo_constraints_satisfied: bool
    best_strategy: str = field(init=False)
    cost_savings: float = field(init=False)

    def __post_init__(self) -> None:
        costs = {
            "VWAP": self.vwap_report.total_cost,
            "TWAP": self.twap_report.total_cost,
            "QUBO": self.qubo_report.total_cost,
        }
        self.best_strategy = min(costs, key=lambda name: costs[name])
        self.cost_savings = (costs["VWAP"] + costs["TWAP"]) / 2 - costs["QUBO"]

    @property
    def reports(self) -> dict[str, ExecutionReport]:
        return {"VWAP": self.vwap_report, "TWAP": self.twap_report, "QUBO": self.qubo_report}

    def to_dataframe(self) -> pd.DataFrame:
        metrics = [
            ("Avg Exec Price ($)", "average_execution_price", "{:.4f}"),
            ("Benchmark Price ($)", "benchmark_vwap", "{:.4f}"),
            ("Slippage (bps)", "slippage_vs_vwap_bps", "{:+.2f}"),
            ("Total Cost ($)", "total_cost", "{:.2f}"),
            ("Spread Cost ($)", "spread_cost", "{:.2f}"),
            ("Impact Cost ($)", "impact_cost", "{:.2f}"),
            ("Timing Risk ($)", "timing_risk", "{:.4f}"),
            ("Fill Rate (%)", "fill_rate", "{:.1%}"),
            ("Child Orders", "num_child_orders", "{:d}"),
            ("Exec Time (min)", "execution_time_minutes", "{:d}"),
        ]
        rows = []
        for name, attr, fmt in metrics:
            row = {"Metric": name}
            for strategy, report in self.reports.items():
                value = getattr(report, attr)
                row[strategy] = fmt.format(int(value) if fmt == "{:d}" else value)
            rows.append(row)
        return pd.DataFrame(rows)


def run_integrated_comparison(
    parent_order: ParentOrder,
    market_data: pd.DataFrame,
    qubo_time_slices: int = 20,
    qubo_sa_sweeps: int = 1000,
    seed: int = 42,
) -> StrategyComparison:
    """Execute copies of `parent_order` with VWAP, TWAP and QUBO, each in a fresh engine."""
    qubo_strategy = QUBOStrategy(
        num_time_slices=qubo_time_slices, sa_sweeps=qubo_sa_sweeps, seed=seed
    )
    strategies: dict[str, BaseStrategy] = {
        "VWAP": VWAPStrategy(participation_rate=0.1, seed=seed),
        "TWAP": TWAPStrategy(interval_minutes=1, seed=seed),
        "QUBO": qubo_strategy,
    }
    reports: dict[str, ExecutionReport] = {}
    for name, strategy in strategies.items():
        order = ParentOrder(
            symbol=parent_order.symbol,
            side=parent_order.side,
            total_quantity=parent_order.total_quantity,
            time_horizon_minutes=len(market_data),
        )
        reports[name] = ExecutionEngine(seed=seed).process_order(order, market_data, strategy)
        logger.info(
            "%s: cost $%.2f, slippage %+.2f bps",
            name,
            reports[name].total_cost,
            reports[name].slippage_vs_vwap_bps,
        )

    stats = qubo_strategy.get_optimization_stats()
    return StrategyComparison(
        vwap_report=reports["VWAP"],
        twap_report=reports["TWAP"],
        qubo_report=reports["QUBO"],
        qubo_optimization_time=stats.get("optimization_time_s", 0.0),
        qubo_energy=stats.get("qubo_energy", 0.0),
        qubo_constraints_satisfied=stats.get("constraints_satisfied", False),
    )
