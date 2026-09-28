"""Volatility/spread regime detection and the regime-dependent risk aversion lambda.

lambda = clip(base * m_vol * m_spread * m_volume, lambda_min, lambda_max), smoothed by an EWMA.
In Almgren-Chriss terms a higher lambda front-loads execution; a lower one approaches TWAP.
"""

from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

import numpy as np

_MIN_OBSERVATIONS = 20
_VOL_HISTORY = 500
_SPREAD_RECENT = 10
_MIN_VARIANCE = 1e-12


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


@dataclass(frozen=True)
class RegimeState:
    """Regime classification and lambda after one tick."""

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
    """Fast (alpha=0.06) and slow (alpha=0.01) EWMA of squared log returns.

    Both are initialised to the sample variance of the first 20 returns; the regime is
    read from the ratio sqrt(fast/slow): <0.7 low, <1.3 normal, <2 high, else extreme.
    """

    def __init__(self, fast_alpha: float = 0.06, slow_alpha: float = 0.01, vol_window: int = 100):
        self._fast_alpha = fast_alpha
        self._slow_alpha = slow_alpha
        self._fast_var = 0.0
        self._slow_var = 0.0
        self._initialized = False
        self._returns: deque[float] = deque(maxlen=vol_window)
        self._prev_price: float | None = None
        self._vol_history: deque[float] = deque(maxlen=_VOL_HISTORY)

    def update(self, price: float) -> tuple[float, float]:
        """Return (fast vol, slow vol); zeros until initialised."""
        if self._prev_price is None:
            self._prev_price = price
            return 0.0, 0.0

        ret = float(np.log(price / self._prev_price))
        self._prev_price = price
        self._returns.append(ret)

        if not self._initialized:
            if len(self._returns) >= _MIN_OBSERVATIONS:
                init_var = float(np.var(list(self._returns)))
                self._fast_var = init_var
                self._slow_var = init_var
                self._initialized = True
            return 0.0, 0.0

        r2 = ret * ret
        self._fast_var = (1 - self._fast_alpha) * self._fast_var + self._fast_alpha * r2
        self._slow_var = (1 - self._slow_alpha) * self._slow_var + self._slow_alpha * r2
        fast_vol = float(np.sqrt(self._fast_var))
        slow_vol = float(np.sqrt(self._slow_var))
        self._vol_history.append(fast_vol)
        return fast_vol, slow_vol

    @property
    def fast_vol(self) -> float:
        return float(np.sqrt(self._fast_var)) if self._initialized else 0.0

    @property
    def slow_vol(self) -> float:
        return float(np.sqrt(self._slow_var)) if self._initialized else 0.0

    @property
    def vol_ratio(self) -> float:
        if self._slow_var > _MIN_VARIANCE:
            return float(np.sqrt(self._fast_var / self._slow_var))
        return 1.0

    @property
    def vol_of_vol(self) -> float:
        if len(self._vol_history) < _MIN_OBSERVATIONS:
            return 0.0
        return float(np.std(list(self._vol_history)))

    def detect_regime(self) -> VolatilityRegime:
        ratio = self.vol_ratio
        if ratio < 0.7:
            return VolatilityRegime.LOW
        if ratio < 1.3:
            return VolatilityRegime.NORMAL
        if ratio < 2.0:
            return VolatilityRegime.HIGH
        return VolatilityRegime.EXTREME


class SpreadEstimator:
    """Spread regime from the mean of the last 10 spreads relative to the median of the first 20.

    Ratio <0.7 tight, <1.5 normal, <3 wide, else gapped.
    """

    def __init__(self, window: int = 100) -> None:
        self._spreads_bps: deque[float] = deque(maxlen=window)
        self._baseline_spread: float | None = None

    def update(self, bid: float, ask: float) -> SpreadRegime:
        mid = (bid + ask) / 2
        if mid <= 0:
            return SpreadRegime.NORMAL

        self._spreads_bps.append((ask - bid) / mid * 10_000)
        if len(self._spreads_bps) < _MIN_OBSERVATIONS:
            return SpreadRegime.NORMAL
        if self._baseline_spread is None:
            self._baseline_spread = float(np.median(list(self._spreads_bps)))

        ratio = (
            self.current_spread_bps / self._baseline_spread if self._baseline_spread > 0 else 1.0
        )
        if ratio < 0.7:
            return SpreadRegime.TIGHT
        if ratio < 1.5:
            return SpreadRegime.NORMAL
        if ratio < 3.0:
            return SpreadRegime.WIDE
        return SpreadRegime.GAPPED

    @property
    def current_spread_bps(self) -> float:
        if not self._spreads_bps:
            return 0.0
        return float(np.mean(list(self._spreads_bps)[-_SPREAD_RECENT:]))


class AdaptiveRiskManager:
    """Regime-dependent risk aversion lambda for the HFT QUBO.

    Multipliers: volatility low/normal/high/extreme = 0.5/1/2/4; spread tight/normal/wide/gapped
    = 1.2/1/0.6/0.3; volume 0.8 when volume > 2x average, 1.3 when < 0.3x.
    """

    VOL_MULTIPLIERS: ClassVar[dict[VolatilityRegime, float]] = {
        VolatilityRegime.LOW: 0.5,
        VolatilityRegime.NORMAL: 1.0,
        VolatilityRegime.HIGH: 2.0,
        VolatilityRegime.EXTREME: 4.0,
    }
    SPREAD_MULTIPLIERS: ClassVar[dict[SpreadRegime, float]] = {
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
        smoothing: float = 0.3,
    ) -> None:
        self._base_lambda = base_lambda
        self._lambda_min = lambda_min
        self._lambda_max = lambda_max
        self._smoothing = smoothing
        self._current_lambda = base_lambda
        self._vol_estimator = VolatilityEstimator()
        self._spread_estimator = SpreadEstimator()
        self._state = RegimeState()
        self._state_history: deque[RegimeState] = deque(maxlen=1000)
        self._regime_start_tick = 0
        self._current_tick = 0

    @staticmethod
    def _volume_multiplier(volume_ratio: float) -> float:
        if volume_ratio > 2.0:
            return 0.8
        if volume_ratio < 0.3:
            return 1.3
        return 1.0

    def update(
        self, price: float, bid: float, ask: float, volume: float = 0.0, avg_volume: float = 1.0
    ) -> RegimeState:
        self._current_tick += 1
        fast_vol, slow_vol = self._vol_estimator.update(price)
        vol_regime = self._vol_estimator.detect_regime()
        spread_regime = self._spread_estimator.update(bid, ask)
        if vol_regime != self._state.vol_regime or spread_regime != self._state.spread_regime:
            self._regime_start_tick = self._current_tick

        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        raw_lambda = (
            self._base_lambda
            * self.VOL_MULTIPLIERS[vol_regime]
            * self.SPREAD_MULTIPLIERS[spread_regime]
            * self._volume_multiplier(volume_ratio)
        )
        raw_lambda = float(np.clip(raw_lambda, self._lambda_min, self._lambda_max))
        self._current_lambda = (
            self._smoothing * raw_lambda + (1 - self._smoothing) * self._current_lambda
        )

        self._state = RegimeState(
            vol_regime=vol_regime,
            spread_regime=spread_regime,
            lambda_value=self._current_lambda,
            realized_vol=fast_vol,
            ewma_vol=slow_vol,
            vol_of_vol=self._vol_estimator.vol_of_vol,
            avg_spread_bps=self._spread_estimator.current_spread_bps,
            volume_ratio=volume_ratio,
            regime_confidence=min(1.0, abs(self._vol_estimator.vol_ratio - 1.0) * 2),
            regime_duration_ticks=self._current_tick - self._regime_start_tick,
            timestamp=self._current_tick,
        )
        self._state_history.append(self._state)
        return self._state

    @property
    def current_lambda(self) -> float:
        return self._current_lambda

    @property
    def current_state(self) -> RegimeState:
        return self._state

    def get_qubo_params(self) -> dict[str, float | str]:
        """Risk aversion, impact scale (1 / 1.5 / 2.5) and timing weight (0.3 / 0.5) by regime."""
        s = self._state
        impact_scale = {VolatilityRegime.HIGH: 1.5, VolatilityRegime.EXTREME: 2.5}.get(
            s.vol_regime, 1.0
        )
        stressed = s.vol_regime in (VolatilityRegime.HIGH, VolatilityRegime.EXTREME)
        return {
            "risk_aversion": self._current_lambda,
            "impact_scale": impact_scale,
            "timing_weight": 0.5 if stressed else 0.3,
            "vol_regime": s.vol_regime.value,
            "spread_regime": s.spread_regime.value,
            "realized_vol": s.realized_vol,
            "regime_confidence": s.regime_confidence,
        }

    def get_regime_summary(self) -> dict[str, Any]:
        """Lambda statistics and volatility-regime counts over the stored history."""
        if not self._state_history:
            return {}
        lambdas = [s.lambda_value for s in self._state_history]
        return {
            "current_lambda": self._current_lambda,
            "mean_lambda": float(np.mean(lambdas)),
            "min_lambda": float(np.min(lambdas)),
            "max_lambda": float(np.max(lambdas)),
            "lambda_std": float(np.std(lambdas)),
            "regime_distribution": dict(Counter(s.vol_regime.value for s in self._state_history)),
            "total_ticks": len(self._state_history),
            "current_regime": self._state.vol_regime.value,
            "regime_duration": self._state.regime_duration_ticks,
        }
