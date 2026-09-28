"""
Order Book Simulation

Simulates market depth with multiple price levels for realistic
execution simulation including market impact.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class PriceLevel:
    """A single price level in the order book."""
    price: float
    quantity: int
    num_orders: int = 1


@dataclass
class OrderBookSnapshot:
    """Snapshot of the order book at a point in time."""
    bids: List[PriceLevel] = field(default_factory=list)  # Sorted descending by price
    asks: List[PriceLevel] = field(default_factory=list)  # Sorted ascending by price
    
    @property
    def best_bid(self) -> Optional[float]:
        """Best (highest) bid price."""
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Best (lowest) ask price."""
        return self.asks[0].price if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Mid price between best bid and ask."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    def total_bid_volume(self, levels: int = None) -> int:
        """Total volume on bid side."""
        bids = self.bids[:levels] if levels else self.bids
        return sum(level.quantity for level in bids)
    
    def total_ask_volume(self, levels: int = None) -> int:
        """Total volume on ask side."""
        asks = self.asks[:levels] if levels else self.asks
        return sum(level.quantity for level in asks)


class OrderBook:
    """
    Simulates a limit order book with realistic market depth.
    
    Generates order book snapshots based on current market conditions
    and simulates execution including market impact.
    """
    
    def __init__(
        self,
        num_levels: int = 10,
        tick_size: float = 0.01,
        base_level_volume: int = 1000,
        volume_decay: float = 0.7,
        seed: Optional[int] = None
    ):
        """
        Initialize the order book simulator.
        
        Args:
            num_levels: Number of price levels on each side
            tick_size: Minimum price increment
            base_level_volume: Average volume at best bid/ask
            volume_decay: Decay factor for volume at deeper levels
            seed: Random seed for reproducibility
        """
        self.num_levels = num_levels
        self.tick_size = tick_size
        self.base_level_volume = base_level_volume
        self.volume_decay = volume_decay
        self.rng = np.random.default_rng(seed)
    
    def generate_snapshot(
        self,
        mid_price: float,
        spread: float,
        minute_volume: int
    ) -> OrderBookSnapshot:
        """
        Generate an order book snapshot given current market conditions.
        
        Args:
            mid_price: Current mid price
            spread: Current bid-ask spread
            minute_volume: Expected volume this minute (affects depth)
            
        Returns:
            OrderBookSnapshot with bid and ask levels
        """
        # Scale base volume by minute volume (normalized by typical volume)
        typical_minute_volume = 100000
        volume_scale = minute_volume / typical_minute_volume
        scaled_base_volume = int(self.base_level_volume * volume_scale)
        scaled_base_volume = max(scaled_base_volume, 100)  # Minimum volume
        
        # Calculate best bid/ask from mid and spread
        half_spread = spread / 2
        best_bid = mid_price - half_spread
        best_ask = mid_price + half_spread
        
        # Round to tick size
        best_bid = np.floor(best_bid / self.tick_size) * self.tick_size
        best_ask = np.ceil(best_ask / self.tick_size) * self.tick_size
        
        # Generate bid levels (descending prices)
        bids = []
        for i in range(self.num_levels):
            price = best_bid - i * self.tick_size
            
            # Volume decays exponentially with distance from best
            base_qty = scaled_base_volume * (self.volume_decay ** i)
            
            # Add randomness (log-normal)
            qty = int(base_qty * self.rng.lognormal(0, 0.5))
            qty = max(qty, 10)  # Minimum quantity
            
            num_orders = max(1, int(qty / 100))  # Rough estimate of order count
            
            bids.append(PriceLevel(price=price, quantity=qty, num_orders=num_orders))
        
        # Generate ask levels (ascending prices)
        asks = []
        for i in range(self.num_levels):
            price = best_ask + i * self.tick_size
            
            base_qty = scaled_base_volume * (self.volume_decay ** i)
            qty = int(base_qty * self.rng.lognormal(0, 0.5))
            qty = max(qty, 10)
            
            num_orders = max(1, int(qty / 100))
            
            asks.append(PriceLevel(price=price, quantity=qty, num_orders=num_orders))
        
        return OrderBookSnapshot(bids=bids, asks=asks)
    
    def simulate_execution(
        self,
        snapshot: OrderBookSnapshot,
        order_size: int,
        side: str,
        aggressive: bool = True
    ) -> Tuple[float, int, float]:
        """
        Simulate execution of an order against the order book.
        
        Args:
            snapshot: Current order book snapshot
            order_size: Number of shares to execute
            side: 'buy' or 'sell'
            aggressive: If True, takes liquidity (market order behavior)
            
        Returns:
            Tuple of:
                - average_price: Volume-weighted average execution price
                - filled_quantity: Number of shares actually filled
                - market_impact: Price impact from order (in dollars)
        """
        if side.lower() == "buy":
            levels = snapshot.asks  # Buy orders hit the ask side
        else:
            levels = snapshot.bids  # Sell orders hit the bid side
        
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
            
            # How much can we fill at this level?
            fill_at_level = min(remaining, level.quantity)
            
            total_value += fill_at_level * level.price
            filled += fill_at_level
            remaining -= fill_at_level
            last_fill_price = level.price
        
        if filled == 0:
            return 0.0, 0, 0.0
        
        average_price = total_value / filled
        
        # Market impact is the difference between last fill and initial price
        if side.lower() == "buy":
            market_impact = last_fill_price - initial_price
        else:
            market_impact = initial_price - last_fill_price
        
        return average_price, filled, market_impact
    
    def get_depth_summary(self, snapshot: OrderBookSnapshot) -> dict:
        """
        Get a summary of order book depth.
        
        Args:
            snapshot: Order book snapshot
            
        Returns:
            Dictionary with depth statistics
        """
        return {
            "best_bid": snapshot.best_bid,
            "best_ask": snapshot.best_ask,
            "mid_price": snapshot.mid_price,
            "spread": snapshot.spread,
            "spread_bps": (snapshot.spread / snapshot.mid_price * 10000) if snapshot.mid_price else None,
            "bid_depth_5": snapshot.total_bid_volume(5),
            "ask_depth_5": snapshot.total_ask_volume(5),
            "bid_depth_total": snapshot.total_bid_volume(),
            "ask_depth_total": snapshot.total_ask_volume(),
            "imbalance": (
                (snapshot.total_bid_volume(5) - snapshot.total_ask_volume(5)) /
                (snapshot.total_bid_volume(5) + snapshot.total_ask_volume(5))
                if (snapshot.total_bid_volume(5) + snapshot.total_ask_volume(5)) > 0 else 0
            )
        }
