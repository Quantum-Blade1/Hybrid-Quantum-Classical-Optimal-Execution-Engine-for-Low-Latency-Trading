"""Parent/child order execution engine with fill simulation and post-trade metrics."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qexec.execution.strategies.base import (
    BaseStrategy,
    half_spread_cost,
    share_weighted_std,
    slippage_bps,
)
from qexec.market.order_book import OrderBook
from qexec.market.simulator import calculate_vwap
from qexec.optimization.schedule import repair_schedule


def _short_id() -> str:
    return str(uuid.uuid4())[:8]


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class ParentOrder:
    """Full trading intention, split into child orders by a strategy."""

    symbol: str
    side: OrderSide
    total_quantity: int
    time_horizon_minutes: int
    start_time: datetime | None = None
    strategy_name: str = "VWAP"
    order_id: str = field(default_factory=_short_id)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: float = 0.0

    @property
    def remaining_quantity(self) -> int:
        return self.total_quantity - self.filled_quantity

    @property
    def fill_rate(self) -> float:
        if self.total_quantity == 0:
            return 0.0
        return self.filled_quantity / self.total_quantity

    @property
    def is_complete(self) -> bool:
        return self.filled_quantity >= self.total_quantity

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "total_quantity": self.total_quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_price": self.average_price,
            "fill_rate": f"{self.fill_rate * 100:.1f}%",
            "status": self.status.value,
            "strategy": self.strategy_name,
            "time_horizon": f"{self.time_horizon_minutes} min",
        }


@dataclass
class ChildOrder:
    """One scheduled slice of a parent order, executed as a marketable order."""

    parent_id: str
    target_quantity: int
    target_time: datetime
    sequence: int = 0
    minute_index: int = 0
    child_id: str = field(default_factory=_short_id)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    execution_price: float = 0.0
    executed_at: datetime | None = None
    market_price_at_execution: float = 0.0
    spread_at_execution: float = 0.0
    market_impact: float = 0.0

    @property
    def slippage(self) -> float:
        """Execution price minus mid at execution."""
        if self.execution_price > 0 and self.market_price_at_execution > 0:
            return self.execution_price - self.market_price_at_execution
        return 0.0

    @property
    def slippage_bps(self) -> float:
        if self.market_price_at_execution > 0:
            return self.slippage / self.market_price_at_execution * 10_000
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "child_id": self.child_id,
            "parent_id": self.parent_id,
            "sequence": self.sequence,
            "target_qty": self.target_quantity,
            "filled_qty": self.filled_quantity,
            "exec_price": f"${self.execution_price:.4f}" if self.execution_price else "-",
            "market_price": f"${self.market_price_at_execution:.4f}"
            if self.market_price_at_execution
            else "-",
            "slippage_bps": f"{self.slippage_bps:+.2f}",
            "status": self.status.value,
        }


@dataclass
class ExecutionState:
    """Running totals while a parent order executes."""

    parent_order: ParentOrder
    child_orders: list[ChildOrder] = field(default_factory=list)
    current_minute: int = 0
    total_value_executed: float = 0.0
    total_spread_cost: float = 0.0
    total_impact_cost: float = 0.0
    fill_prices: list[float] = field(default_factory=list)
    fill_quantities: list[int] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def elapsed_minutes(self) -> int:
        return self.current_minute

    @property
    def average_execution_price(self) -> float:
        if self.parent_order.filled_quantity > 0:
            return self.total_value_executed / self.parent_order.filled_quantity
        return 0.0

    @property
    def total_cost(self) -> float:
        return self.total_spread_cost + self.total_impact_cost

    @property
    def timing_risk(self) -> float:
        """Share-weighted standard deviation of fill prices."""
        return share_weighted_std(self.fill_prices, self.fill_quantities)


def implementation_shortfall(
    side: str,
    *,
    arrival_price: float,
    filled_quantity: int,
    average_price: float,
    unfilled_quantity: int,
    completion_price: float,
) -> tuple[float, float]:
    """(execution cost, opportunity cost) in currency, positive = cost, vs the arrival mid.

    Execution cost is sum n_i (p_i - P_0) for a buy (sign flipped for a sell); opportunity
    cost is U (P_c - P_0) for the U unfilled shares at `completion_price` P_c, the price of
    completing them at the end of the horizon (`ExecutionEngine` uses a clean-up market
    order against the final bar's book).
    """
    sign = 1.0 if side.lower() == "buy" else -1.0
    execution = sign * (average_price - arrival_price) * filled_quantity
    opportunity = sign * (completion_price - arrival_price) * unfilled_quantity
    return execution, opportunity


@dataclass
class ExecutionReport:
    """Post-trade analytics of one parent order; slippage is positive when adverse.

    `implementation_shortfall` (currency) = execution cost of the fills vs the arrival mid
    + opportunity cost of unfilled shares (see `implementation_shortfall()`);
    `implementation_shortfall_bps` divides it by the arrival notional of the whole order.
    """

    order_id: str
    symbol: str
    side: str
    total_quantity: int
    filled_quantity: int
    average_execution_price: float
    benchmark_vwap: float
    benchmark_twap: float
    arrival_price: float
    slippage_vs_vwap_bps: float
    slippage_vs_twap_bps: float
    slippage_vs_arrival_bps: float
    total_cost: float
    spread_cost: float
    impact_cost: float
    cost_per_share: float
    timing_risk: float
    fill_rate: float
    num_child_orders: int
    execution_time_minutes: int
    strategy_used: str
    started_at: datetime | None
    completed_at: datetime | None
    completion_price: float = 0.0

    @property
    def unfilled_quantity(self) -> int:
        return self.total_quantity - self.filled_quantity

    @property
    def execution_shortfall(self) -> float:
        return self._shortfall()[0]

    @property
    def opportunity_cost(self) -> float:
        return self._shortfall()[1]

    @property
    def implementation_shortfall(self) -> float:
        return sum(self._shortfall())

    @property
    def implementation_shortfall_bps(self) -> float:
        notional = self.total_quantity * self.arrival_price
        return self.implementation_shortfall / notional * 10_000 if notional > 0 else 0.0

    def _shortfall(self) -> tuple[float, float]:
        return implementation_shortfall(
            self.side,
            arrival_price=self.arrival_price,
            filled_quantity=self.filled_quantity,
            average_price=self.average_execution_price,
            unfilled_quantity=self.unfilled_quantity,
            completion_price=self.completion_price,
        )

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            ("Order ID", self.order_id),
            ("Symbol", self.symbol),
            ("Side", self.side),
            ("Total Quantity", f"{self.total_quantity:,}"),
            ("Filled Quantity", f"{self.filled_quantity:,}"),
            ("Fill Rate", f"{self.fill_rate * 100:.1f}%"),
            ("Avg Execution Price", f"${self.average_execution_price:.4f}"),
            ("Benchmark VWAP", f"${self.benchmark_vwap:.4f}"),
            ("Arrival Price", f"${self.arrival_price:.4f}"),
            ("Slippage vs VWAP", f"{self.slippage_vs_vwap_bps:+.2f} bps"),
            ("Slippage vs Arrival", f"{self.slippage_vs_arrival_bps:+.2f} bps"),
            ("Total Cost", f"${self.total_cost:.2f}"),
            ("Spread Cost", f"${self.spread_cost:.2f}"),
            ("Impact Cost", f"${self.impact_cost:.2f}"),
            ("Cost/Share", f"${self.cost_per_share:.4f}"),
            ("Timing Risk", f"${self.timing_risk:.4f}"),
            ("Child Orders", str(self.num_child_orders)),
            ("Execution Time", f"{self.execution_time_minutes} min"),
            ("Strategy", self.strategy_used),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"])

    def __repr__(self) -> str:
        return (
            f"ExecutionReport(order={self.order_id})\n"
            f"  {self.side.upper()} {self.filled_quantity:,}/{self.total_quantity:,} "
            f"{self.symbol}\n"
            f"  Avg Price: ${self.average_execution_price:.4f} | VWAP: ${self.benchmark_vwap:.4f}\n"
            f"  Slippage: {self.slippage_vs_vwap_bps:+.2f} bps | Cost: ${self.total_cost:.2f}\n"
            f"  Strategy: {self.strategy_used} | Time: {self.execution_time_minutes} min"
        )


class ExecutionEngine:
    """Turns a strategy schedule into child orders, fills them against a simulated book,
    and reports cost and slippage against VWAP, TWAP and arrival price.

    Fill model (the same for every strategy, docs/MATHEMATICAL_MODEL.md, "Execution
    simulation"): each minute's child order walks a fresh synthetic book whose depth
    scales with that bar's volume; a bar with zero volume fills nothing. With
    `carry_forward` (default), shares a child could not fill are added to the next
    minute's target, so a liquidity gap delays shares instead of dropping them; shares
    still unfilled after the last bar count as opportunity cost in the report. The
    book's level sizes are keyed by (seed, minute), so strategies run with the same
    `seed` face identical books at every minute.
    """

    def __init__(
        self,
        order_book: OrderBook | None = None,
        seed: int | None = None,
        carry_forward: bool = True,
    ) -> None:
        self.order_book = order_book or OrderBook(seed=seed)
        self.carry_forward = carry_forward
        self._minute_offset = 0
        self.state: ExecutionState | None = None
        self.execution_history: list[ExecutionReport] = []

    def process_order(
        self,
        parent_order: ParentOrder,
        market_data: pd.DataFrame,
        strategy: BaseStrategy,
        start_minute: int = 0,
        end_minute: int | None = None,
    ) -> ExecutionReport:
        """Execute `parent_order` over `market_data[start_minute:end_minute]`."""
        if parent_order.total_quantity <= 0:
            raise ValueError("Order quantity must be positive")
        if end_minute is None:
            end_minute = min(start_minute + parent_order.time_horizon_minutes, len(market_data))

        parent_order.status = OrderStatus.ACTIVE
        parent_order.start_time = market_data.iloc[start_minute]["timestamp"]
        self.state = ExecutionState(parent_order=parent_order, started_at=parent_order.start_time)

        execution_data = market_data.iloc[start_minute:end_minute].reset_index(drop=True)
        self._minute_offset = start_minute
        arrival_price = float(execution_data.iloc[0]["price"])
        num_minutes = len(execution_data)
        plan = self._cap_schedule(
            strategy.calculate_schedule(parent_order.total_quantity, execution_data),
            parent_order.total_quantity,
            num_minutes,
        )

        carry = 0
        for minute_idx in range(num_minutes):
            remaining = parent_order.remaining_quantity
            if remaining <= 0:
                break
            new_plan = strategy.replan(
                minute_idx, remaining, execution_data.iloc[: minute_idx + 1], num_minutes
            )
            if new_plan is not None:
                plan[minute_idx:] = repair_schedule(
                    np.asarray(new_plan, dtype=float)[: num_minutes - minute_idx], remaining
                )
                carry = 0
            target = min(int(plan[minute_idx]) + carry, remaining)
            if target <= 0:
                continue
            child = ChildOrder(
                parent_id=parent_order.order_id,
                target_quantity=target,
                target_time=execution_data.iloc[minute_idx]["timestamp"],
                sequence=len(self.state.child_orders) + 1,
                minute_index=minute_idx,
            )
            self.state.child_orders.append(child)
            self._execute_slice(child, parent_order.side.value, execution_data, start_minute)
            carry = target - child.filled_quantity if self.carry_forward else 0

        parent_order.status = (
            OrderStatus.FILLED if parent_order.is_complete else OrderStatus.PARTIALLY_FILLED
        )
        self.state.completed_at = datetime.now()

        report = self._calculate_metrics(execution_data, arrival_price, strategy.strategy_name)
        self.execution_history.append(report)
        return report

    @staticmethod
    def _cap_schedule(schedule: NDArray[Any], total: int, num_minutes: int) -> NDArray[np.int_]:
        """Integer per-minute plan truncated so cumulative targets never exceed the order."""
        plan = np.zeros(num_minutes, dtype=np.int_)
        unassigned = total
        for minute_idx, scheduled in enumerate(np.asarray(schedule)[:num_minutes]):
            qty = min(max(int(scheduled), 0), unassigned)
            plan[minute_idx] = qty
            unassigned -= qty
        return plan

    def _execute_slice(
        self, child: ChildOrder, side: str, market_data: pd.DataFrame, minute_offset: int = 0
    ) -> None:
        """Fill one child order at its scheduled minute and update parent and state totals."""
        if self.state is None:
            raise RuntimeError("No active execution state")
        parent = self.state.parent_order

        minute_idx = child.minute_index
        if minute_idx >= len(market_data):
            child.status = OrderStatus.REJECTED
            return
        row = market_data.iloc[minute_idx]
        mid = float(row["price"])

        child.status = OrderStatus.ACTIVE
        child.market_price_at_execution = mid
        child.spread_at_execution = float(row["spread"])
        child.executed_at = row["timestamp"]
        self.state.current_minute = minute_idx + 1

        snapshot = self.order_book.generate_snapshot(
            mid_price=mid,
            spread=row["spread"],
            minute_volume=int(row["volume"]),
            key=minute_offset + minute_idx,
        )
        avg_price, filled, impact = self.order_book.simulate_execution(
            snapshot=snapshot, order_size=child.target_quantity, side=side
        )

        child.filled_quantity = filled
        child.execution_price = avg_price
        child.market_impact = impact
        if filled == 0:
            child.status = OrderStatus.REJECTED
            return
        child.status = (
            OrderStatus.FILLED if filled >= child.target_quantity else OrderStatus.PARTIALLY_FILLED
        )

        parent.filled_quantity += filled
        previous_value = parent.average_price * (parent.filled_quantity - filled)
        parent.average_price = (previous_value + avg_price * filled) / parent.filled_quantity
        self.state.total_value_executed += avg_price * filled
        self.state.fill_prices.append(avg_price)
        self.state.fill_quantities.append(filled)
        self.state.total_spread_cost += half_spread_cost(snapshot, mid, side, filled)
        self.state.total_impact_cost += abs(impact) * filled

    def _calculate_metrics(
        self, market_data: pd.DataFrame, arrival_price: float, strategy_name: str
    ) -> ExecutionReport:
        if self.state is None:
            raise RuntimeError("No active execution state")
        parent = self.state.parent_order
        side = parent.side.value

        benchmark_vwap = calculate_vwap(market_data)
        benchmark_twap = float(market_data["price"].mean())
        avg_price = self.state.average_execution_price
        num_children = sum(1 for c in self.state.child_orders if c.filled_quantity > 0)
        cost_per_share = (
            self.state.total_cost / parent.filled_quantity if parent.filled_quantity > 0 else 0.0
        )

        return ExecutionReport(
            order_id=parent.order_id,
            symbol=parent.symbol,
            side=side,
            total_quantity=parent.total_quantity,
            filled_quantity=parent.filled_quantity,
            average_execution_price=avg_price,
            benchmark_vwap=benchmark_vwap,
            benchmark_twap=benchmark_twap,
            arrival_price=arrival_price,
            slippage_vs_vwap_bps=slippage_bps(avg_price, benchmark_vwap, side),
            slippage_vs_twap_bps=slippage_bps(avg_price, benchmark_twap, side),
            slippage_vs_arrival_bps=slippage_bps(avg_price, arrival_price, side),
            total_cost=self.state.total_cost,
            spread_cost=self.state.total_spread_cost,
            impact_cost=self.state.total_impact_cost,
            cost_per_share=cost_per_share,
            timing_risk=self.state.timing_risk,
            fill_rate=parent.fill_rate,
            num_child_orders=num_children,
            execution_time_minutes=self.state.elapsed_minutes,
            strategy_used=strategy_name,
            started_at=self.state.started_at,
            completed_at=self.state.completed_at,
            completion_price=self._completion_price(market_data, parent.remaining_quantity, side),
        )

    def _completion_price(self, market_data: pd.DataFrame, unfilled: int, side: str) -> float:
        """Average price of a clean-up market order for `unfilled` shares at the last bar.

        It walks the final bar's (keyed) book; shares beyond the book's depth are priced at
        its deepest level. With no liquidity in the final bar, the far touch P_T +- s_T/2.
        A buy's remainder is therefore charged at least the half spread and the impact of
        trading it at once, so underfilling cannot make a strategy look cheaper.
        """
        row = market_data.iloc[-1]
        mid, spread = float(row["price"]), float(row["spread"])
        far_touch = mid + spread / 2 if side == "buy" else mid - spread / 2
        if unfilled <= 0:
            return far_touch
        snapshot = self.order_book.generate_snapshot(
            mid_price=mid,
            spread=spread,
            minute_volume=int(row["volume"]),
            key=len(market_data) - 1 + self._minute_offset,
        )
        levels = snapshot.asks if side == "buy" else snapshot.bids
        if not levels:
            return far_touch
        avg_price, filled, _ = self.order_book.simulate_execution(snapshot, unfilled, side)
        beyond = unfilled - filled
        return (avg_price * filled + levels[-1].price * beyond) / unfilled

    def get_execution_report(self) -> ExecutionReport | None:
        """Most recent report, if any."""
        return self.execution_history[-1] if self.execution_history else None

    def get_child_orders_df(self) -> pd.DataFrame:
        if self.state is None or not self.state.child_orders:
            return pd.DataFrame()
        return pd.DataFrame([c.to_dict() for c in self.state.child_orders])

    def get_execution_history(self) -> pd.DataFrame:
        """One summary row per processed order."""
        return pd.DataFrame(
            [
                {
                    "order_id": r.order_id,
                    "symbol": r.symbol,
                    "side": r.side,
                    "quantity": r.filled_quantity,
                    "avg_price": f"${r.average_execution_price:.2f}",
                    "slippage_bps": f"{r.slippage_vs_vwap_bps:+.2f}",
                    "cost": f"${r.total_cost:.2f}",
                    "strategy": r.strategy_used,
                }
                for r in self.execution_history
            ]
        )
