"""TWAP: equal slices at fixed intervals, capped at a fraction of the order per slice."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qexec.execution.strategies.base import BaseStrategy, ExecutionMetrics
from qexec.market.order_book import OrderBook
from qexec.market.simulator import calculate_vwap


class TWAPStrategy(BaseStrategy):
    """Time-weighted execution benchmarked against the TWAP of the execution points."""

    strategy_name = "TWAP"
    benchmark_name = "TWAP"

    def __init__(
        self,
        interval_minutes: int = 1,
        max_slice_pct: float = 0.05,
        order_book: OrderBook | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(order_book=order_book, seed=seed)
        self.interval_minutes = interval_minutes
        self.max_slice_pct = max_slice_pct

    def calculate_schedule(self, total_shares: int, market_data: pd.DataFrame) -> NDArray[np.int_]:
        """Equal slices every `interval_minutes`; shares cut by the cap are topped up afterwards."""
        num_minutes = len(market_data)
        schedule = np.zeros(num_minutes, dtype=int)
        execution_points = list(range(0, num_minutes, self.interval_minutes))
        num_slices = len(execution_points)
        if num_slices == 0:
            return schedule

        remainder = total_shares % num_slices
        max_slice_shares = int(total_shares * self.max_slice_pct)
        base_slice_size = min(total_shares // num_slices, max_slice_shares)

        shares_allocated = 0
        for i, point in enumerate(execution_points):
            if shares_allocated >= total_shares:
                break
            slice_size = base_slice_size + (1 if i < remainder else 0)
            slice_size = min(slice_size, total_shares - shares_allocated)
            schedule[point] = slice_size
            shares_allocated += slice_size

        remaining = total_shares - shares_allocated
        for point in execution_points:
            if remaining <= 0:
                break
            additional = min(remaining, max_slice_shares - schedule[point])
            if additional > 0:
                schedule[point] += additional
                remaining -= additional

        return schedule

    def calculate_benchmark(self, market_data: pd.DataFrame) -> float:
        """Mean price at the execution points."""
        execution_points = list(range(0, len(market_data), self.interval_minutes))
        return float(market_data.iloc[execution_points]["price"].mean())


def compare_strategies(
    vwap_strategy: BaseStrategy,
    twap_strategy: BaseStrategy,
    total_shares: int,
    side: str,
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """Execute both strategies on the same data and tabulate their metrics."""
    vwap_metrics = vwap_strategy.execute(total_shares, side, market_data)
    twap_metrics = twap_strategy.execute(total_shares, side, market_data)
    vwap_price = calculate_vwap(market_data)

    def column(m: ExecutionMetrics, benchmark: str) -> list[str]:
        return [
            f"${m.average_execution_price:.4f}",
            benchmark,
            f"{m.slippage_bps:+.2f}",
            f"${m.spread_cost:.2f}",
            f"${m.impact_cost:.2f}",
            f"${m.total_cost:.2f}",
            f"${m.timing_risk:.4f}",
            f"{m.fill_rate * 100:.1f}%",
            str(m.num_slices),
            str(m.execution_time_minutes),
        ]

    return pd.DataFrame(
        {
            "Metric": [
                "Average Execution Price",
                "Benchmark Price",
                "Slippage (bps)",
                "Spread Cost ($)",
                "Impact Cost ($)",
                "Total Cost ($)",
                "Timing Risk ($)",
                "Fill Rate (%)",
                "Num Slices",
                "Execution Time (min)",
            ],
            "VWAP": column(vwap_metrics, f"${vwap_price:.4f} (VWAP)"),
            "TWAP": column(twap_metrics, f"${twap_metrics.benchmark_price:.4f} (TWAP)"),
        }
    )
