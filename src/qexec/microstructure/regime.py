"""
Adaptive Risk Aversion for Real-Time QUBO Re-parameterization

Implements dynamic risk aversion (lambda) that adapts in real-time
based on market regime detection:

1. Volatility regime detection (GARCH-inspired)
2. Spread regime classification
3. Volume regime analysis
4. Dynamic lambda computation
5. QUBO parameter hot-reload

In the Almgren-Chriss model, lambda controls the trade-off between
execution cost (speed) and timing risk (patience):
    - High lambda -> front-load execution (risk-averse)
    - Low lambda -> spread evenly (risk-neutral, approaches TWAP)

No existing quantum finance paper adapts lambda in real-time.
Rosenberg et al. (2016) and all subsequent papers use fixed parameters.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import deque
from enum import Enum


class VolatilityRegime(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class SpreadRegime(Enum):
    TIGHT = "tight"
    NORMAL = "normal"
    WIDE = "wide"
    GAPPED = "gapped"


@dataclass
class RegimeState:
    vol_regime: VolatilityRegime = VolatilityRegime.NORMAL
    spread_regime: SpreadRegime = SpreadRegime.NORMAL
    lambda_value: float = 1.0
    realized_vol: float = 0.0
    ewma_vol: float = 0.0
    vol_of_vol: float = 0.0
    avg_spread_bps: float = 0.0
    volume_ratio: float = 1.0
    regime_confidence: float = 0.0
    regime_duration_ticks: int = 0
    timestamp: int = 0


class VolatilityEstimator:
    """
    Real-time volatility estimation using EWMA with regime detection.

    Uses a dual-timescale EWMA (fast + slow) to detect regime changes:
    - Fast EWMA (alpha=0.06) responds quickly to volatility spikes
    - Slow EWMA (alpha=0.01) tracks long-term volatility level
    - Ratio of fast/slow detects regime transitions
    """

    def __init__(
        self,
        fast_alpha: float = 0.06,
        slow_alpha: float = 0.01,
        vol_window: int = 100
    ):
        self._fast_alpha = fast_alpha
        self._slow_alpha = slow_alpha
        self._fast_var: float = 0.0
        self._slow_var: float = 0.0
        self._initialized = False
        self._returns: deque = deque(maxlen=vol_window)
        self._prev_price: Optional[float] = None
        self._vol_history: deque = deque(maxlen=500)

    def update(self, price: float) -> Tuple[float, float]:
        if self._prev_price is None:
            self._prev_price = price
            return 0.0, 0.0

        ret = np.log(price / self._prev_price)
        self._prev_price = price
        self._returns.append(ret)

        r2 = ret * ret

        if not self._initialized:
            if len(self._returns) >= 20:
                init_var = np.var(list(self._returns))
                self._fast_var = init_var
                self._slow_var = init_var
                self._initialized = True
            return 0.0, 0.0

        self._fast_var = (1 - self._fast_alpha) * self._fast_var + self._fast_alpha * r2
        self._slow_var = (1 - self._slow_alpha) * self._slow_var + self._slow_alpha * r2

        fast_vol = np.sqrt(self._fast_var)
        slow_vol = np.sqrt(self._slow_var)

        self._vol_history.append(fast_vol)

        return fast_vol, slow_vol

    @property
    def fast_vol(self) -> float:
        return np.sqrt(self._fast_var) if self._initialized else 0.0

    @property
    def slow_vol(self) -> float:
        return np.sqrt(self._slow_var) if self._initialized else 0.0

    @property
    def vol_ratio(self) -> float:
        if self._slow_var > 1e-12:
            return np.sqrt(self._fast_var / self._slow_var)
        return 1.0

    @property
    def vol_of_vol(self) -> float:
        if len(self._vol_history) < 20:
            return 0.0
        return float(np.std(list(self._vol_history)))

    def detect_regime(self) -> VolatilityRegime:
        ratio = self.vol_ratio

        if ratio < 0.7:
            return VolatilityRegime.LOW
        elif ratio < 1.3:
            return VolatilityRegime.NORMAL
        elif ratio < 2.0:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME


class SpreadEstimator:
    """Real-time spread regime estimator."""

    def __init__(self, window: int = 100):
        self._spreads_bps: deque = deque(maxlen=window)
        self._baseline_spread: Optional[float] = None

    def update(self, bid: float, ask: float) -> SpreadRegime:
        mid = (bid + ask) / 2
        if mid <= 0:
            return SpreadRegime.NORMAL

        spread_bps = (ask - bid) / mid * 10000
        self._spreads_bps.append(spread_bps)

        if len(self._spreads_bps) < 20:
            return SpreadRegime.NORMAL

        if self._baseline_spread is None:
            self._baseline_spread = float(np.median(list(self._spreads_bps)))

        current = float(np.mean(list(self._spreads_bps)[-10:]))
        ratio = current / self._baseline_spread if self._baseline_spread > 0 else 1.0

        if ratio < 0.7:
            return SpreadRegime.TIGHT
        elif ratio < 1.5:
            return SpreadRegime.NORMAL
        elif ratio < 3.0:
            return SpreadRegime.WIDE
        else:
            return SpreadRegime.GAPPED

    @property
    def current_spread_bps(self) -> float:
        if not self._spreads_bps:
            return 0.0
        return float(np.mean(list(self._spreads_bps)[-10:]))


class AdaptiveRiskManager:
    """
    Computes dynamic risk aversion (lambda) for QUBO parameterization.

    Lambda mapping:
        lambda = base_lambda * vol_multiplier * spread_multiplier * volume_multiplier

    Vol regime effects:
        LOW:     vol_multiplier = 0.5  (can afford to be patient)
        NORMAL:  vol_multiplier = 1.0  (baseline)
        HIGH:    vol_multiplier = 2.0  (front-load to avoid risk)
        EXTREME: vol_multiplier = 4.0  (emergency: execute fast)

    Spread regime effects:
        TIGHT:   spread_multiplier = 1.2  (cheap to cross, slightly aggressive)
        NORMAL:  spread_multiplier = 1.0  (baseline)
        WIDE:    spread_multiplier = 0.6  (expensive to cross, be passive)
        GAPPED:  spread_multiplier = 0.3  (very expensive, maximum patience)
    """

    VOL_MULTIPLIERS = {
        VolatilityRegime.LOW: 0.5,
        VolatilityRegime.NORMAL: 1.0,
        VolatilityRegime.HIGH: 2.0,
        VolatilityRegime.EXTREME: 4.0,
    }

    SPREAD_MULTIPLIERS = {
        SpreadRegime.TIGHT: 1.2,
        SpreadRegime.NORMAL: 1.0,
        SpreadRegime.WIDE: 0.6,
        SpreadRegime.GAPPED: 0.3,
    }

    def __init__(
        self,
        base_lambda: float = 1.0,
        lambda_min: float = 0.1,
        lambda_max: float = 10.0,
        smoothing: float = 0.3
    ):
        self._base_lambda = base_lambda
        self._lambda_min = lambda_min
        self._lambda_max = lambda_max
        self._smoothing = smoothing
        self._current_lambda = base_lambda

        self._vol_estimator = VolatilityEstimator()
        self._spread_estimator = SpreadEstimator()

        self._state = RegimeState()
        self._state_history: deque = deque(maxlen=1000)
        self._regime_start_tick = 0
        self._current_tick = 0

    def update(
        self,
        price: float,
        bid: float,
        ask: float,
        volume: float = 0.0,
        avg_volume: float = 1.0
    ) -> RegimeState:
        self._current_tick += 1

        fast_vol, slow_vol = self._vol_estimator.update(price)
        vol_regime = self._vol_estimator.detect_regime()
        spread_regime = self._spread_estimator.update(bid, ask)

        if vol_regime != self._state.vol_regime or spread_regime != self._state.spread_regime:
            self._regime_start_tick = self._current_tick

        vol_mult = self.VOL_MULTIPLIERS.get(vol_regime, 1.0)
        spread_mult = self.SPREAD_MULTIPLIERS.get(spread_regime, 1.0)

        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        vol_adjustment = 1.0
        if volume_ratio > 2.0:
            vol_adjustment = 0.8
        elif volume_ratio < 0.3:
            vol_adjustment = 1.3

        raw_lambda = self._base_lambda * vol_mult * spread_mult * vol_adjustment
        raw_lambda = np.clip(raw_lambda, self._lambda_min, self._lambda_max)

        self._current_lambda = (
            self._smoothing * raw_lambda
            + (1 - self._smoothing) * self._current_lambda
        )

        vol_ratio = self._vol_estimator.vol_ratio
        confidence = min(1.0, abs(vol_ratio - 1.0) * 2)

        self._state = RegimeState(
            vol_regime=vol_regime,
            spread_regime=spread_regime,
            lambda_value=self._current_lambda,
            realized_vol=fast_vol,
            ewma_vol=slow_vol,
            vol_of_vol=self._vol_estimator.vol_of_vol,
            avg_spread_bps=self._spread_estimator.current_spread_bps,
            volume_ratio=volume_ratio,
            regime_confidence=confidence,
            regime_duration_ticks=self._current_tick - self._regime_start_tick,
            timestamp=self._current_tick
        )

        self._state_history.append(self._state)
        return self._state

    @property
    def current_lambda(self) -> float:
        return self._current_lambda

    @property
    def current_state(self) -> RegimeState:
        return self._state

    def get_qubo_params(self) -> Dict[str, float]:
        """
        Get QUBO parameters adjusted for current regime.

        These override the static QUBOConfig values.
        """
        s = self._state

        impact_scale = 1.0
        if s.vol_regime == VolatilityRegime.HIGH:
            impact_scale = 1.5
        elif s.vol_regime == VolatilityRegime.EXTREME:
            impact_scale = 2.5

        timing_weight = 0.3
        if s.vol_regime in (VolatilityRegime.HIGH, VolatilityRegime.EXTREME):
            timing_weight = 0.5

        return {
            "risk_aversion": self._current_lambda,
            "impact_scale": impact_scale,
            "timing_weight": timing_weight,
            "vol_regime": s.vol_regime.value,
            "spread_regime": s.spread_regime.value,
            "realized_vol": s.realized_vol,
            "regime_confidence": s.regime_confidence
        }

    def get_regime_summary(self) -> Dict:
        if not self._state_history:
            return {}

        lambdas = [s.lambda_value for s in self._state_history]
        vol_regimes = [s.vol_regime.value for s in self._state_history]

        from collections import Counter
        regime_counts = Counter(vol_regimes)

        return {
            "current_lambda": self._current_lambda,
            "mean_lambda": float(np.mean(lambdas)),
            "min_lambda": float(np.min(lambdas)),
            "max_lambda": float(np.max(lambdas)),
            "lambda_std": float(np.std(lambdas)),
            "regime_distribution": dict(regime_counts),
            "total_ticks": len(self._state_history),
            "current_regime": self._state.vol_regime.value,
            "regime_duration": self._state.regime_duration_ticks
        }
