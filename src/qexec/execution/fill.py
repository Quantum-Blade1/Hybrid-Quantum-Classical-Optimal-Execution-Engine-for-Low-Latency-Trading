from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from qexec.execution.strategies.base import half_spread_cost
from qexec.market.order_book import OrderBook


@dataclass(frozen=True)
class Fill:
    """Outcome of one child order; costs are in currency and positive when adverse."""

    quantity: int
    average_price: float
    half_spread_cost: float
    impact_cost: float
    market_impact: float  # per share; book model: last fill price minus the touch


NO_FILL = Fill(0, 0.0, 0.0, 0.0, 0.0)


def _side_sign(side: str) -> float:
    return 1.0 if side.lower() == "buy" else -1.0


class FillModel(Protocol):
    def fill(self, bar: pd.Series, quantity: int, side: str, key: int) -> Fill: ...

    def completion_price(self, bar: pd.Series, quantity: int, side: str, key: int) -> float: ...


class BookFillModel:
    """Walk a fresh synthetic book per bar; level sizes keyed by (seed, minute)."""

    def __init__(self, order_book: OrderBook) -> None:
        self.order_book = order_book

    def _snapshot(self, bar: pd.Series, key: int):  # type: ignore[no-untyped-def]
        return self.order_book.generate_snapshot(
            mid_price=float(bar["price"]),
            spread=float(bar["spread"]),
            minute_volume=int(bar["volume"]),
            key=key,
        )

    def fill(self, bar: pd.Series, quantity: int, side: str, key: int) -> Fill:
        snapshot = self._snapshot(bar, key)
        avg_price, filled, impact = self.order_book.simulate_execution(
            snapshot=snapshot, order_size=quantity, side=side
        )
        if filled == 0:
            return NO_FILL
        return Fill(
            quantity=filled,
            average_price=avg_price,
            half_spread_cost=half_spread_cost(snapshot, float(bar["price"]), side, filled),
            impact_cost=abs(impact) * filled,
            market_impact=impact,
        )

    def completion_price(self, bar: pd.Series, quantity: int, side: str, key: int) -> float:
        """Shares past the book's depth pay its deepest level; with no liquidity, the far touch."""
        mid, spread = float(bar["price"]), float(bar["spread"])
        far_touch = mid + spread / 2 if side == "buy" else mid - spread / 2
        if quantity <= 0:
            return far_touch
        snapshot = self._snapshot(bar, key)
        levels = snapshot.asks if side == "buy" else snapshot.bids
        if not levels:
            return far_touch
        avg_price, filled, _ = self.order_book.simulate_execution(snapshot, quantity, side)
        beyond = quantity - filled
        return (avg_price * filled + levels[-1].price * beyond) / quantity


class ImpactFillModel:
    """Half spread + linear impact in participation, capped (docs/MATHEMATICAL_MODEL.md)."""

    def __init__(self, impact_bps: float, participation_cap: float = 0.25) -> None:
        if impact_bps < 0:
            raise ValueError("impact_bps must be non-negative")
        if not 0 < participation_cap <= 1:
            raise ValueError("participation_cap must be in (0, 1]")
        # Price move in bps per unit participation (filled / V).
        self.impact_bps = impact_bps
        self.participation_cap = participation_cap

    @staticmethod
    def half_spread_bps(bar: pd.Series) -> float:
        if "half_spread_bps" in bar and not pd.isna(bar["half_spread_bps"]):
            return float(bar["half_spread_bps"])
        return float(bar["spread"]) / 2 / float(bar["price"]) * 1e4

    def _price(self, bar: pd.Series, quantity: int, volume: float, side: str) -> Fill:
        mid = float(bar["price"])
        h = self.half_spread_bps(bar)
        impact = self.impact_bps * quantity / volume
        sign = _side_sign(side)
        return Fill(
            quantity=quantity,
            average_price=mid * (1 + sign * (h + impact) / 1e4),
            half_spread_cost=mid * h / 1e4 * quantity,
            impact_cost=mid * impact / 1e4 * quantity,
            market_impact=mid * impact / 1e4,
        )

    def fill(self, bar: pd.Series, quantity: int, side: str, key: int) -> Fill:
        volume = float(bar["volume"])
        if volume <= 0 or quantity <= 0:
            return NO_FILL
        filled = min(quantity, math.floor(self.participation_cap * volume))
        if filled <= 0:
            return NO_FILL
        return self._price(bar, filled, volume, side)

    def completion_price(self, bar: pd.Series, quantity: int, side: str, key: int) -> float:
        # The clean-up order is uncapped; an empty final bar falls back to expected volume.
        volume = float(bar["volume"])
        if volume <= 0:
            volume = float(bar["expected_volume"]) if "expected_volume" in bar else 1.0
        return self._price(bar, max(quantity, 0), max(volume, 1.0), side).average_price
