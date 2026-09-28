"""Abstract execution strategy and the fill simulation shared by all strategies."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qexec.market.order_book import OrderBook, OrderBookSnapshot
from qexec.market.simulator import calculate_vwap


@dataclass
class ExecutionMetrics:
    """Execution quality of one order; slippage is positive when worse than the benchmark."""

    strategy_name: str
    average_execution_price: float
    benchmark_price: float
    benchmark_name: str
    slippage_bps: float
    total_cost: float
    spread_cost: float
    impact_cost: float
    timing_risk: float
    fill_rate: float
    num_slices: int
    total_shares: int
    filled_shares: int
    execution_time_minutes: int

    def __repr__(self) -> str:
        return (
            f"ExecutionMetrics({self.strategy_name})\n"
            f"  avg_price={self.average_execution_price:.4f}\n"
            f"  {self.benchmark_name}={self.benchmark_price:.4f}\n"
            f"  slippage={self.slippage_bps:+.2f} bps\n"
            f"  total_cost=${self.total_cost:.2f}\n"
            f"  timing_risk=${self.timing_risk:.4f}\n"
            f"  fill_rate={self.fill_rate * 100:.1f}%\n"
            f"  filled={self.filled_shares:,}/{self.total_shares:,} shares"
        )


@dataclass
class ExecutionSlice:
    """Record of one executed slice."""

    minute: int
    timestamp: pd.Timestamp
    target_quantity: int
    filled_quantity: int
    execution_price: float
    market_mid_price: float
    spread: float
    market_impact: float


def share_weighted_std(prices: Sequence[float], quantities: Sequence[int]) -> float:
    """Standard deviation of fill prices, one observation per share filled."""
    q = np.asarray(quantities, dtype=np.float64)
    if q.sum() <= 1:
        return 0.0
    p = np.asarray(prices, dtype=np.float64)
    mean = np.average(p, weights=q)
    return float(np.sqrt(np.average((p - mean) ** 2, weights=q)))


def half_spread_cost(
    snapshot: OrderBookSnapshot, mid_price: float, side: str, filled: int
) -> float:
    """Cost of crossing from mid to the touch for `filled` shares."""
    if side.lower() == "buy":
        return (float(snapshot.best_ask or mid_price) - mid_price) * filled
    return (mid_price - float(snapshot.best_bid or mid_price)) * filled


def slippage_bps(avg_price: float, benchmark: float, side: str) -> float:
    """Signed slippage in bps; positive means worse than benchmark for this side."""
    if benchmark <= 0:
        return 0.0
    diff = avg_price - benchmark if side.lower() == "buy" else benchmark - avg_price
    return diff / benchmark * 10_000


class BaseStrategy(ABC):
    """Strategy interface: subclasses provide `calculate_schedule` (shares per minute)."""

    strategy_name: str = "Base"
    benchmark_name: str = "VWAP"

    def __init__(self, order_book: OrderBook | None = None, seed: int | None = None) -> None:
        self.order_book = order_book or OrderBook(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.slices: list[ExecutionSlice] = []

    @abstractmethod
    def calculate_schedule(self, total_shares: int, market_data: pd.DataFrame) -> NDArray[np.int_]:
        """Target shares for each row of `market_data`."""

    def calculate_benchmark(self, market_data: pd.DataFrame) -> float:
        """Benchmark price; VWAP unless overridden."""
        return calculate_vwap(market_data)

    def execute(
        self,
        total_shares: int,
        side: str,
        market_data: pd.DataFrame,
        start_minute: int = 0,
        end_minute: int | None = None,
    ) -> ExecutionMetrics:
        """Execute the schedule against simulated order books built from `market_data`."""
        self.slices = []
        if end_minute is None:
            end_minute = len(market_data)
        execution_data = market_data.iloc[start_minute:end_minute].reset_index(drop=True)
        schedule = self.calculate_schedule(total_shares, execution_data)

        total_filled = 0
        total_value = 0.0
        total_spread_cost = 0.0
        total_impact_cost = 0.0

        prices = execution_data["price"].to_numpy(dtype=np.float64)
        spreads = execution_data["spread"].to_numpy(dtype=np.float64)
        volumes = execution_data["volume"].to_numpy(dtype=np.int64)
        timestamps = execution_data["timestamp"].to_list()

        for minute_idx in range(min(len(execution_data), len(schedule))):
            target_qty = int(schedule[minute_idx])
            if target_qty == 0:
                continue
            mid = float(prices[minute_idx])
            spread = float(spreads[minute_idx])

            snapshot = self.order_book.generate_snapshot(
                mid_price=mid, spread=spread, minute_volume=int(volumes[minute_idx])
            )
            avg_price, filled, impact = self.order_book.simulate_execution(
                snapshot=snapshot, order_size=target_qty, side=side
            )
            if filled == 0:
                continue

            total_filled += filled
            total_value += avg_price * filled
            total_spread_cost += half_spread_cost(snapshot, mid, side, filled)
            total_impact_cost += abs(impact) * filled
            self.slices.append(
                ExecutionSlice(
                    minute=minute_idx,
                    timestamp=timestamps[minute_idx],
                    target_quantity=target_qty,
                    filled_quantity=filled,
                    execution_price=avg_price,
                    market_mid_price=mid,
                    spread=spread,
                    market_impact=impact,
                )
            )

        avg_execution_price = total_value / total_filled if total_filled > 0 else 0.0
        timing_risk = share_weighted_std(
            [s.execution_price for s in self.slices], [s.filled_quantity for s in self.slices]
        )
        benchmark_price = self.calculate_benchmark(execution_data)
        execution_time = self.slices[-1].minute - self.slices[0].minute + 1 if self.slices else 0

        return ExecutionMetrics(
            strategy_name=self.strategy_name,
            average_execution_price=avg_execution_price,
            benchmark_price=benchmark_price,
            benchmark_name=self.benchmark_name,
            slippage_bps=slippage_bps(avg_execution_price, benchmark_price, side),
            total_cost=total_spread_cost + total_impact_cost,
            spread_cost=total_spread_cost,
            impact_cost=total_impact_cost,
            timing_risk=timing_risk,
            fill_rate=total_filled / total_shares if total_shares > 0 else 0.0,
            num_slices=len(self.slices),
            total_shares=total_shares,
            filled_shares=total_filled,
            execution_time_minutes=execution_time,
        )

    def get_execution_summary(self) -> pd.DataFrame:
        """One row per executed slice of the last `execute` call."""
        if not self.slices:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "minute": s.minute,
                    "timestamp": s.timestamp,
                    "target_qty": s.target_quantity,
                    "filled_qty": s.filled_quantity,
                    "fill_rate": s.filled_quantity / s.target_quantity
                    if s.target_quantity > 0
                    else 0,
                    "exec_price": s.execution_price,
                    "mid_price": s.market_mid_price,
                    "slippage": s.execution_price - s.market_mid_price,
                    "spread": s.spread,
                    "impact": s.market_impact,
                }
                for s in self.slices
            ]
        )
