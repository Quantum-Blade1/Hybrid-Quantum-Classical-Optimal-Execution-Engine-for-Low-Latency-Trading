from dataclasses import dataclass, field

import numpy as np

# Depth is scaled relative to this per-minute volume.
_TYPICAL_MINUTE_VOLUME = 100_000
_MIN_LEVEL_BASE_VOLUME = 100
_MIN_LEVEL_QUANTITY = 10
_SHARES_PER_ORDER = 100


@dataclass
class PriceLevel:
    price: float
    quantity: int
    num_orders: int = 1


@dataclass
class OrderBookSnapshot:
    """Book state; bids sorted descending, asks ascending."""

    bids: list[PriceLevel] = field(default_factory=list)
    asks: list[PriceLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> float | None:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> float | None:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    def total_bid_volume(self, levels: int | None = None) -> int:
        bids = self.bids[:levels] if levels else self.bids
        return sum(level.quantity for level in bids)

    def total_ask_volume(self, levels: int | None = None) -> int:
        asks = self.asks[:levels] if levels else self.asks
        return sum(level.quantity for level in asks)


class OrderBook:
    """Synthetic limit order book with exponentially decaying depth."""

    def __init__(
        self,
        num_levels: int = 10,
        tick_size: float = 0.01,
        base_level_volume: int = 1000,
        volume_decay: float = 0.7,
        seed: int | None = None,
    ) -> None:
        self.num_levels = num_levels
        self.tick_size = tick_size
        self.base_level_volume = base_level_volume
        self.volume_decay = volume_decay
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _side_levels(
        self, best_price: float, direction: int, base_volume: int, rng: np.random.Generator
    ) -> list[PriceLevel]:
        levels = []
        for i in range(self.num_levels):
            price = best_price + direction * i * self.tick_size
            base_qty = base_volume * (self.volume_decay**i)
            qty = max(int(base_qty * rng.lognormal(0, 0.5)), _MIN_LEVEL_QUANTITY)
            num_orders = max(1, int(qty / _SHARES_PER_ORDER))
            levels.append(PriceLevel(price=price, quantity=qty, num_orders=num_orders))
        return levels

    def generate_snapshot(
        self, mid_price: float, spread: float, minute_volume: int, key: int | None = None
    ) -> OrderBookSnapshot:
        """Lognormal sizes decaying geometrically from the touch; empty in a bar with no volume."""
        if minute_volume <= 0:
            return OrderBookSnapshot()
        # Seeded by (seed, key) so strategies trading at the same minute see the same book.
        if key is not None and self.seed is not None:
            rng = np.random.default_rng([self.seed, key])
        else:
            rng = self.rng
        volume_scale = minute_volume / _TYPICAL_MINUTE_VOLUME
        base_volume = max(int(self.base_level_volume * volume_scale), _MIN_LEVEL_BASE_VOLUME)

        half_spread = spread / 2
        best_bid = float(np.floor((mid_price - half_spread) / self.tick_size) * self.tick_size)
        best_ask = float(np.ceil((mid_price + half_spread) / self.tick_size) * self.tick_size)

        bids = self._side_levels(best_bid, -1, base_volume, rng)
        asks = self._side_levels(best_ask, +1, base_volume, rng)
        return OrderBookSnapshot(bids=bids, asks=asks)

    def simulate_execution(
        self, snapshot: OrderBookSnapshot, order_size: int, side: str
    ) -> tuple[float, int, float]:
        """Walk the book; returns (avg price, filled, impact = last fill price minus the touch)."""
        is_buy = side.lower() == "buy"
        levels = snapshot.asks if is_buy else snapshot.bids
        if not levels:
            return 0.0, 0, 0.0

        remaining = order_size
        total_value = 0.0
        filled = 0
        initial_price = levels[0].price
        last_fill_price = initial_price

        for level in levels:
            if remaining <= 0:
                break
            fill_at_level = min(remaining, level.quantity)
            total_value += fill_at_level * level.price
            filled += fill_at_level
            remaining -= fill_at_level
            last_fill_price = level.price

        if filled == 0:
            return 0.0, 0, 0.0

        average_price = total_value / filled
        if is_buy:
            market_impact = last_fill_price - initial_price
        else:
            market_impact = initial_price - last_fill_price
        return average_price, filled, market_impact
