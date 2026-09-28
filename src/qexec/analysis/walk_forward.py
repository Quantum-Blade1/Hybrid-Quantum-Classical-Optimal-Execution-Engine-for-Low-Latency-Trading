"""Seeded walk-forward backtest of static VWAP, adaptive VWAP and SA-QUBO schedules.

Each window trains on `train_days` simulated days and tests on the next `test_days`:
    static   - VWAP on the first training day's volume profile (stale forecast)
    adaptive - VWAP on the mean training-window volume profile
    hybrid   - SA-solved `slice_level_config` QUBO with the training-window volatility
Shortfall per test day is slippage vs arrival x arrival price x order size (positive = cost).
All three strategies on a given day share one order-book seed (common random numbers).
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qexec.execution.engine import ExecutionEngine, ExecutionReport, OrderSide, ParentOrder
from qexec.execution.strategies.base import BaseStrategy
from qexec.execution.strategies.fixed import FixedScheduleStrategy
from qexec.execution.strategies.vwap import VWAPStrategy
from qexec.market.order_book import OrderBook
from qexec.market.simulator import TRADING_DAYS_PER_YEAR, MarketDataSimulator, MarketParams
from qexec.optimization.qubo import ExecutionQUBO
from qexec.optimization.schedule import optimize_schedule, slice_level_config
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver

logger = logging.getLogger(__name__)

MINUTES_PER_DAY = 390
_MAX_QUBO_SLICES = 20
_SA_SWEEPS = 300
_MIN_DAILY_VOLATILITY = 0.001


@dataclass(frozen=True)
class WindowResult:
    """Summed implementation shortfall ($) over one window's test days."""

    window_id: int
    shortfall_static: float
    shortfall_adaptive: float
    shortfall_hybrid: float
    train_volatility: float

    @property
    def winner(self) -> str:
        scores = {
            "Static": self.shortfall_static,
            "Adaptive": self.shortfall_adaptive,
            "Hybrid": self.shortfall_hybrid,
        }
        return min(scores, key=lambda name: scores[name])


class WalkForwardAnalyzer:
    """Rolling train/test windows over `total_days` chained simulated days."""

    def __init__(
        self,
        total_days: int = 10,
        train_days: int = 3,
        test_days: int = 1,
        daily_shares: int = 50_000,
        seed: int = 42,
    ) -> None:
        self.total_days = total_days
        self.train_days = train_days
        self.test_days = test_days
        self.daily_shares = daily_shares
        self.seed = seed
        self.market_params = MarketParams(initial_price=100.0, annual_volatility=0.40)

    def generate_full_dataset(self) -> list[pd.DataFrame]:
        """`total_days` days; each opens at the previous day's last price."""
        sim = MarketDataSimulator(self.market_params, seed=self.seed)
        days = []
        price = self.market_params.initial_price
        for day_id in range(self.total_days):
            day = sim.generate(num_minutes=MINUTES_PER_DAY, initial_price=price)
            day["day"] = day_id
            days.append(day)
            price = float(day.iloc[-1]["price"])
        return days

    def run(self) -> list[WindowResult]:
        all_days = self.generate_full_dataset()
        book_seeds = np.random.SeedSequence(self.seed).spawn(self.total_days)
        num_windows = (self.total_days - self.train_days) // self.test_days
        results = []
        for w in range(num_windows):
            start = w * self.test_days
            train_end = start + self.train_days
            test_end = train_end + self.test_days
            if test_end > len(all_days):
                break
            seeds = [int(s.generate_state(1)[0]) for s in book_seeds[train_end:test_end]]
            result = self._process_window(
                w, all_days[start:train_end], all_days[train_end:test_end], seeds
            )
            logger.info(
                "window %d: vol %.1f%% static $%.0f adaptive $%.0f hybrid $%.0f (%s)",
                w,
                100 * result.train_volatility,
                result.shortfall_static,
                result.shortfall_adaptive,
                result.shortfall_hybrid,
                result.winner,
            )
            results.append(result)
        return results

    def _hybrid_schedule(self, num_minutes: int, train_vol: float) -> NDArray[np.float64]:
        """SA-QUBO slice quantities at each slice's first minute, scaled to the order size."""
        num_slices = min(_MAX_QUBO_SLICES, num_minutes)
        config = slice_level_config(
            self.daily_shares,
            num_slices,
            volatility=max(_MIN_DAILY_VOLATILITY, train_vol / np.sqrt(TRADING_DAYS_PER_YEAR)),
            impact_coefficient=0.1,
        )
        slice_qty, _ = optimize_schedule(
            ExecutionQUBO(config), SimulatedAnnealingSolver(num_sweeps=_SA_SWEEPS, seed=self.seed)
        )
        schedule = np.zeros(num_minutes)
        minutes_per_slice = max(1, num_minutes // num_slices)
        for t, quantity in enumerate(slice_qty):
            if quantity > 0:
                schedule[min(t * minutes_per_slice, num_minutes - 1)] += quantity
        total = schedule.sum()
        if total > 0:
            return np.asarray(schedule * (self.daily_shares / total))
        return np.full(num_minutes, self.daily_shares / num_minutes)

    def _shortfall(
        self, day: pd.DataFrame, strategy: BaseStrategy, engine: ExecutionEngine
    ) -> float:
        report: ExecutionReport = engine.process_order(self._create_order(), day, strategy)
        return report.slippage_vs_arrival_bps / 10_000 * report.arrival_price * self.daily_shares

    def _process_window(
        self,
        window_id: int,
        train_data: list[pd.DataFrame],
        test_data: list[pd.DataFrame],
        book_seeds: list[int],
    ) -> WindowResult:
        avg_volume_profile = np.mean([d["volume"].to_numpy() for d in train_data], axis=0)
        static_profile = train_data[0]["volume"].to_numpy(dtype=np.float64)
        returns = pd.concat(train_data)["price"].pct_change().dropna()
        train_vol = float(returns.std() * np.sqrt(MINUTES_PER_DAY * TRADING_DAYS_PER_YEAR))

        total_static = total_adaptive = total_hybrid = 0.0
        for day, book_seed in zip(test_data, book_seeds, strict=True):
            eng_s = ExecutionEngine(OrderBook(seed=book_seed))
            total_static += self._shortfall(
                day,
                VWAPStrategy(
                    historical_profile=static_profile, order_book=eng_s.order_book, seed=book_seed
                ),
                eng_s,
            )
            eng_a = ExecutionEngine(OrderBook(seed=book_seed))
            total_adaptive += self._shortfall(
                day,
                VWAPStrategy(
                    historical_profile=avg_volume_profile,
                    order_book=eng_a.order_book,
                    seed=book_seed,
                ),
                eng_a,
            )
            eng_h = ExecutionEngine(OrderBook(seed=book_seed))
            total_hybrid += self._shortfall(
                day, FixedScheduleStrategy(self._hybrid_schedule(len(day), train_vol)), eng_h
            )

        return WindowResult(
            window_id=window_id,
            shortfall_static=total_static,
            shortfall_adaptive=total_adaptive,
            shortfall_hybrid=total_hybrid,
            train_volatility=train_vol,
        )

    def _create_order(self) -> ParentOrder:
        return ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=self.daily_shares,
            time_horizon_minutes=MINUTES_PER_DAY,
        )
