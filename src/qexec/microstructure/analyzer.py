"""
Microstructure analyzer: combines Kyle's lambda, VPIN, adverse selection and
queue models into a per-tick MicrostructureState that parameterises the HFT QUBO.
"""

from dataclasses import dataclass
from typing import Optional

from qexec.microstructure.kyle import KyleLambdaEstimator
from qexec.microstructure.vpin import VPINEstimator
from qexec.microstructure.adverse_selection import AdverseSelectionModel
from qexec.microstructure.queue import QueuePositionModel


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
