"""
HFT Market Microstructure Models

Implements market microstructure metrics critical to high-frequency trading:
1. Kyle's Lambda (price impact coefficient)
2. VPIN (Volume-Synchronized Probability of Informed Trading)
3. Adverse Selection Cost estimation
4. Queue Position Model for limit order placement
5. Order Flow Toxicity detection

These feed directly into the QUBO cost function, making the quantum
optimizer microstructure-aware -- a combination no existing paper provides.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque


@dataclass
class MicrostructureState:
    """Real-time microstructure snapshot for QUBO parameterization."""
    kyle_lambda: float = 0.0
    vpin: float = 0.0
    adverse_selection_cost: float = 0.0
    order_imbalance: float = 0.0
    toxicity_flag: bool = False
    spread_regime: str = "normal"
    effective_spread_bps: float = 0.0
    realized_spread_bps: float = 0.0
    queue_density: float = 0.0
    timestamp_ns: int = 0


class KyleLambdaEstimator:
    """
    Estimates Kyle's lambda (price impact coefficient) from trade data.

    Kyle (1985) model: Delta_p = lambda * (signed_volume) + noise

    lambda represents the permanent price impact per unit of signed order flow.
    Higher lambda = less liquid market = higher execution cost.

    In the QUBO context, lambda scales the market impact quadratic term,
    making the optimizer naturally route less aggressively when impact is high.
    """

    def __init__(self, window_size: int = 100):
        self._price_changes: deque = deque(maxlen=window_size)
        self._signed_volumes: deque = deque(maxlen=window_size)
        self._lambda: float = 0.0
        self._window = window_size

    def update(self, price_change: float, signed_volume: float) -> float:
        self._price_changes.append(price_change)
        self._signed_volumes.append(signed_volume)

        if len(self._price_changes) < 20:
            return self._lambda

        dp = np.array(self._price_changes)
        sv = np.array(self._signed_volumes)

        sv_var = np.var(sv)
        if sv_var < 1e-12:
            return self._lambda

        self._lambda = np.cov(dp, sv)[0, 1] / sv_var
        return self._lambda

    @property
    def lambda_value(self) -> float:
        return max(0.0, self._lambda)

    def estimate_impact(self, order_size: float) -> float:
        return self._lambda * order_size


class VPINEstimator:
    """
    Volume-Synchronized Probability of Informed Trading.

    VPIN (Easley, Lopez de Prado, O'Hara 2012) estimates the probability
    that a trade is information-driven rather than noise. High VPIN signals
    that the market is toxic for market makers and liquidity providers.

    For HFT execution, high VPIN means:
    - Wider effective spreads (adverse selection)
    - Higher market impact (information leakage)
    - The QUBO should shift to more passive, spread-crossing execution

    The quantum optimizer uses VPIN to dynamically adjust:
    - impact_coefficient in QUBOConfig
    - Venue routing weights (avoid lit venues when toxic)
    - Urgency parameter (slow down in toxic flow)
    """

    def __init__(self, bucket_size: int = 1000, num_buckets: int = 50):
        self._bucket_size = bucket_size
        self._num_buckets = num_buckets
        self._current_bucket_volume = 0
        self._current_buy_volume = 0
        self._buckets: deque = deque(maxlen=num_buckets)
        self._vpin: float = 0.0

    def update(self, price: float, volume: int, prev_price: float) -> float:
        if price > prev_price:
            buy_volume = volume
        elif price < prev_price:
            buy_volume = 0
        else:
            buy_volume = volume // 2

        self._current_buy_volume += buy_volume
        self._current_bucket_volume += volume

        if self._current_bucket_volume >= self._bucket_size:
            sell_volume = self._current_bucket_volume - self._current_buy_volume
            imbalance = abs(self._current_buy_volume - sell_volume)
            self._buckets.append((imbalance, self._current_bucket_volume))

            self._current_bucket_volume = 0
            self._current_buy_volume = 0

            if len(self._buckets) >= 10:
                total_imbalance = sum(b[0] for b in self._buckets)
                total_volume = sum(b[1] for b in self._buckets)
                self._vpin = total_imbalance / total_volume if total_volume > 0 else 0

        return self._vpin

    @property
    def vpin(self) -> float:
        return self._vpin

    @property
    def is_toxic(self) -> bool:
        return self._vpin > 0.7


class AdverseSelectionModel:
    """
    Estimates adverse selection cost from trade and quote data.

    Adverse selection = cost of trading against informed counterparties.
    Measured as the difference between the effective spread (what you paid)
    and the realized spread (what you kept after price moved).

    AS_cost = (effective_spread - realized_spread) / 2

    When AS_cost is high, the QUBO should:
    - Increase dark pool routing weight (avoid lit adverse selection)
    - Widen the timing window (patient execution)
    - Reduce per-slice quantity (smaller footprint)
    """

    def __init__(self, lookback_ticks: int = 100, realized_horizon: int = 10):
        self._lookback = lookback_ticks
        self._horizon = realized_horizon
        self._trades: deque = deque(maxlen=lookback_ticks + realized_horizon)
        self._midpoints: deque = deque(maxlen=lookback_ticks + realized_horizon)

    def update(self, trade_price: float, midpoint: float, side: str) -> None:
        self._trades.append((trade_price, midpoint, side))
        self._midpoints.append(midpoint)

    def estimate(self) -> Tuple[float, float, float]:
        if len(self._trades) < self._lookback + self._horizon:
            return 0.0, 0.0, 0.0

        effective_spreads = []
        realized_spreads = []

        for i in range(len(self._trades) - self._horizon):
            price, mid_at_trade, side = self._trades[i]
            mid_after = self._midpoints[i + self._horizon]

            if side == "buy":
                eff = 2 * (price - mid_at_trade)
                real = 2 * (price - mid_after)
            else:
                eff = 2 * (mid_at_trade - price)
                real = 2 * (mid_after - price)

            effective_spreads.append(eff)
            realized_spreads.append(real)

        avg_effective = np.mean(effective_spreads) if effective_spreads else 0
        avg_realized = np.mean(realized_spreads) if realized_spreads else 0
        adverse_selection = max(0, avg_effective - avg_realized)

        return avg_effective, avg_realized, adverse_selection


class QueuePositionModel:
    """
    Models queue position dynamics for limit order placement.

    In HFT, queue position determines fill probability. Orders at the
    front of the queue fill first; orders at the back may never fill.

    The QUBO formulation incorporates queue position as:
    - Expected fill probability per price level
    - Opportunity cost of waiting (price may move away)
    - Adverse selection risk of being picked off

    This allows the quantum optimizer to decide between:
    - Aggressive (market order): guaranteed fill, pays spread
    - Passive (limit order at best): uncertain fill, earns spread
    - Passive (limit order deeper): very uncertain fill, earns more spread
    """

    def __init__(self, tick_size: float = 0.01):
        self._tick_size = tick_size
        self._fill_rates: dict = {}
        self._cancel_rates: dict = {}

    def estimate_fill_probability(
        self,
        queue_position: int,
        total_queue: int,
        time_horizon_ms: float,
        arrival_rate: float
    ) -> float:
        if total_queue <= 0:
            return 1.0

        rate = arrival_rate * (time_horizon_ms / 1000.0)
        prob = 1.0 - np.exp(-rate * (1.0 - queue_position / total_queue))
        return np.clip(prob, 0.0, 1.0)

    def optimal_placement_level(
        self,
        book_levels: List[Tuple[float, int]],
        mid_price: float,
        urgency: float,
        time_horizon_ms: float,
        arrival_rate: float = 10.0
    ) -> Tuple[int, float]:
        best_level = 0
        best_score = -np.inf

        for level_idx, (price, queue_size) in enumerate(book_levels):
            distance_ticks = abs(price - mid_price) / self._tick_size
            fill_prob = self.estimate_fill_probability(
                queue_size, queue_size, time_horizon_ms, arrival_rate
            )
            spread_earned = abs(mid_price - price)
            adverse_risk = 0.001 * distance_ticks
            score = (
                fill_prob * (spread_earned - adverse_risk)
                - (1 - fill_prob) * urgency * abs(mid_price - price)
            )

            if score > best_score:
                best_score = score
                best_level = level_idx

        return best_level, best_score


class MicrostructureAnalyzer:
    """
    Unified microstructure analysis engine that combines all models
    and produces a MicrostructureState for QUBO parameterization.

    This is the bridge between raw market data and the quantum optimizer:
    Market Ticks -> MicrostructureAnalyzer -> MicrostructureState -> QUBO params
    """

    def __init__(
        self,
        kyle_window: int = 100,
        vpin_bucket_size: int = 1000,
        vpin_buckets: int = 50,
        tick_size: float = 0.01
    ):
        self.kyle = KyleLambdaEstimator(kyle_window)
        self.vpin_estimator = VPINEstimator(vpin_bucket_size, vpin_buckets)
        self.adverse_selection = AdverseSelectionModel()
        self.queue_model = QueuePositionModel(tick_size)

        self._prev_price: Optional[float] = None
        self._prev_mid: Optional[float] = None
        self._state = MicrostructureState()

    def process_tick(
        self,
        price: float,
        volume: int,
        bid: float,
        ask: float,
        side: str = "buy",
        timestamp_ns: int = 0
    ) -> MicrostructureState:
        mid = (bid + ask) / 2
        spread = ask - bid

        if self._prev_price is not None:
            price_change = price - self._prev_price
            signed_vol = volume if price > self._prev_price else -volume
            self.kyle.update(price_change, signed_vol)
            self.vpin_estimator.update(price, volume, self._prev_price)

        if self._prev_mid is not None:
            self.adverse_selection.update(price, mid, side)

        eff_spread, real_spread, as_cost = self.adverse_selection.estimate()

        eff_spread_bps = (eff_spread / mid * 10000) if mid > 0 else 0
        real_spread_bps = (real_spread / mid * 10000) if mid > 0 else 0

        if self.vpin_estimator.vpin > 0.7:
            regime = "toxic"
        elif spread / mid * 10000 > 10:
            regime = "wide"
        else:
            regime = "normal"

        self._state = MicrostructureState(
            kyle_lambda=self.kyle.lambda_value,
            vpin=self.vpin_estimator.vpin,
            adverse_selection_cost=as_cost,
            order_imbalance=0.0,
            toxicity_flag=self.vpin_estimator.is_toxic,
            spread_regime=regime,
            effective_spread_bps=eff_spread_bps,
            realized_spread_bps=real_spread_bps,
            queue_density=0.0,
            timestamp_ns=timestamp_ns
        )

        self._prev_price = price
        self._prev_mid = mid

        return self._state

    def get_qubo_adjustments(self) -> dict:
        """
        Convert microstructure state to QUBO parameter adjustments.

        Returns multipliers/offsets that the HFT QUBO applies on top
        of the base Almgren-Chriss parameters.
        """
        s = self._state

        impact_multiplier = 1.0 + 2.0 * s.kyle_lambda
        if s.toxicity_flag:
            impact_multiplier *= 1.5

        venue_dark_pool_bonus = 0.0
        if s.adverse_selection_cost > 0:
            venue_dark_pool_bonus = min(0.5, s.adverse_selection_cost * 100)

        urgency_adjustment = 1.0
        if s.vpin > 0.6:
            urgency_adjustment = 0.5

        risk_aversion_multiplier = 1.0
        if s.spread_regime == "toxic":
            risk_aversion_multiplier = 2.0
        elif s.spread_regime == "wide":
            risk_aversion_multiplier = 1.5

        return {
            "impact_multiplier": impact_multiplier,
            "venue_dark_pool_bonus": venue_dark_pool_bonus,
            "urgency_adjustment": urgency_adjustment,
            "risk_aversion_multiplier": risk_aversion_multiplier,
            "spread_regime": s.spread_regime,
            "vpin": s.vpin,
            "kyle_lambda": s.kyle_lambda
        }


def run_microstructure_demo():
    """Demonstrate microstructure analysis on synthetic tick data."""
    np.random.seed(42)

    analyzer = MicrostructureAnalyzer(
        kyle_window=50,
        vpin_bucket_size=500,
        vpin_buckets=20
    )

    price = 100.0
    n_ticks = 2000

    print("\n" + "=" * 70)
    print(" HFT Microstructure Analysis Demo")
    print("=" * 70)

    states = []
    for i in range(n_ticks):
        ret = np.random.normal(0, 0.0005)
        if i > 1500:
            ret += 0.0003

        price *= (1 + ret)
        volume = int(np.random.exponential(200))
        spread = max(0.01, np.random.exponential(0.02))
        bid = price - spread / 2
        ask = price + spread / 2
        side = "buy" if ret > 0 else "sell"

        state = analyzer.process_tick(price, volume, bid, ask, side, i * 1_000_000)
        states.append(state)

    print(f"\n  Final State (tick {n_ticks}):")
    print(f"    Kyle's Lambda:        {states[-1].kyle_lambda:.6f}")
    print(f"    VPIN:                 {states[-1].vpin:.4f}")
    print(f"    Adverse Selection:    ${states[-1].adverse_selection_cost:.6f}")
    print(f"    Spread Regime:        {states[-1].spread_regime}")
    print(f"    Toxicity Flag:        {states[-1].toxicity_flag}")

    adjustments = analyzer.get_qubo_adjustments()
    print(f"\n  QUBO Adjustments:")
    for k, v in adjustments.items():
        print(f"    {k}: {v}")

    print("=" * 70)
    return states


if __name__ == "__main__":
    run_microstructure_demo()
