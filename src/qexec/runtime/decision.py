"""Cost-benefit rule for invoking the (expensive) slow-path optimizer.

invoke iff order_size >= min_order_size, the optimizer is available, the expected latency
is within max_latency_ms, and E[improvement] / latency_cost > lambda_tradeoff.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Prior used when there is no improvement history (an assumption, not a measurement).
DEFAULT_IMPROVEMENT_PCT = 0.01
_IMPACT_VOL_SCALE = 0.1
_LATENCY_VOL_SCALE = 0.001


@dataclass(frozen=True)
class DecisionConfig:
    """Thresholds and cost coefficients of the decision rule."""

    lambda_tradeoff: float = 1.0
    min_order_size: int = 1000
    max_latency_ms: float = 1000.0
    improvement_history_size: int = 20
    latency_cost_per_ms: float = 0.001
    volatility_impact_weight: float = 0.5


@dataclass(frozen=True)
class MarketState:
    """Market conditions the decision is conditioned on."""

    current_price: float
    bid_ask_spread: float
    market_depth: int
    recent_volatility: float
    volume_rate: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def spread_bps(self) -> float:
        return (self.bid_ask_spread / self.current_price) * 10_000

    @classmethod
    def from_market_data(cls, market_data: pd.DataFrame) -> "MarketState":
        """Latest bar's price, spread and volume; volatility is the std of simple returns
        (0.01 for a single bar)."""
        latest = market_data.iloc[-1]
        if len(market_data) > 1:
            volatility = float(market_data["price"].pct_change().dropna().std())
        else:
            volatility = 0.01
        return cls(
            current_price=float(latest["price"]),
            bid_ask_spread=float(latest.get("spread", latest["price"] * 0.0005)),
            market_depth=int(latest.get("volume", 100_000)),
            recent_volatility=volatility,
            volume_rate=float(market_data["volume"].mean())
            if "volume" in market_data.columns
            else 10_000.0,
        )


class ImprovementTracker:
    """Similarity- and recency-weighted mean of past relative improvements."""

    def __init__(self, max_history: int = 20) -> None:
        self.max_history = max_history
        self._improvement_pcts: deque[float] = deque(maxlen=max_history)
        self._contexts: deque[tuple[int, float]] = deque(maxlen=max_history)

    def __len__(self) -> int:
        return len(self._improvement_pcts)

    def record(
        self, baseline_cost: float, optimized_cost: float, order_size: int, volatility: float
    ) -> None:
        improvement = baseline_cost - optimized_cost
        improvement_pct = improvement / baseline_cost if baseline_cost > 0 else 0.0
        self._improvement_pcts.append(improvement_pct)
        self._contexts.append((order_size, volatility))
        logger.debug("Recorded improvement: $%.2f (%.2f%%)", improvement, 100 * improvement_pct)

    def expected_improvement(
        self, order_size: int, volatility: float, base_cost_estimate: float
    ) -> float:
        """base_cost x weighted mean improvement; weight = similarity(size, vol) x recency."""
        if not self._improvement_pcts:
            return base_cost_estimate * DEFAULT_IMPROVEMENT_PCT

        total_weight = 0.0
        weighted_improvement = 0.0
        n = len(self._contexts)
        for i, ((hist_size, hist_vol), pct) in enumerate(
            zip(self._contexts, self._improvement_pcts, strict=True)
        ):
            size_ratio = min(order_size, hist_size) / max(order_size, hist_size)
            vol_ratio = min(volatility, hist_vol) / max(volatility, hist_vol + 1e-10)
            similarity = (size_ratio + vol_ratio) / 2
            recency = (i + 1) / n
            weight = similarity * recency
            weighted_improvement += weight * pct
            total_weight += weight

        avg_pct = (
            weighted_improvement / total_weight if total_weight > 0 else DEFAULT_IMPROVEMENT_PCT
        )
        return base_cost_estimate * avg_pct

    @property
    def mean_improvement_pct(self) -> float:
        if not self._improvement_pcts:
            return DEFAULT_IMPROVEMENT_PCT
        return float(np.mean(self._improvement_pcts))


@dataclass(frozen=True)
class DecisionResult:
    invoke_optimization: bool
    reason: str
    expected_improvement: float
    latency_cost: float
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)


class OptimizationDecisionEngine:
    """Applies the decision rule and learns expected improvement from recorded outcomes."""

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self.config = config or DecisionConfig()
        self.improvement_tracker = ImprovementTracker(
            max_history=self.config.improvement_history_size
        )
        self._optimizer_available = True
        self.decisions_made = 0
        self.optimizations_invoked = 0

    def set_optimizer_available(self, available: bool) -> None:
        self._optimizer_available = available

    @staticmethod
    def _skip(reason: str) -> DecisionResult:
        return DecisionResult(
            invoke_optimization=False,
            reason=reason,
            expected_improvement=0,
            latency_cost=0,
            confidence=1.0,
        )

    def decide(
        self, order_size: int, market_state: MarketState, optimization_latency_ms: float = 500.0
    ) -> DecisionResult:
        """Hard thresholds first, then the cost-benefit ratio against `lambda_tradeoff`.

        Base cost = N spread/2 + 0.1 N P sigma; E[improvement] = tracker estimate +
        sigma w_vol base cost; latency cost = t c_ms + (t/1000) sigma P N 0.001.
        """
        self.decisions_made += 1
        cfg = self.config
        if order_size < cfg.min_order_size:
            return self._skip(f"Order size {order_size} below minimum {cfg.min_order_size}")
        if not self._optimizer_available:
            return self._skip("Optimizer not available")
        if optimization_latency_ms > cfg.max_latency_ms:
            return self._skip(
                f"Latency {optimization_latency_ms}ms exceeds max {cfg.max_latency_ms}ms"
            )

        sigma = market_state.recent_volatility
        price = market_state.current_price
        base_cost_estimate = (
            order_size * market_state.bid_ask_spread / 2
            + order_size * price * sigma * _IMPACT_VOL_SCALE
        )
        tracked = self.improvement_tracker.expected_improvement(
            order_size=order_size, volatility=sigma, base_cost_estimate=base_cost_estimate
        )
        expected_improvement = tracked + sigma * cfg.volatility_impact_weight * base_cost_estimate
        latency_cost = (
            optimization_latency_ms * cfg.latency_cost_per_ms
            + optimization_latency_ms / 1000 * sigma * price * order_size * _LATENCY_VOL_SCALE
        )

        cost_benefit_ratio = expected_improvement / (latency_cost + 1e-10)
        invoke = cost_benefit_ratio > cfg.lambda_tradeoff
        if invoke:
            self.optimizations_invoked += 1

        return DecisionResult(
            invoke_optimization=invoke,
            reason=(
                f"CBR={cost_benefit_ratio:.2f} {'>' if invoke else '<='} "
                f"lambda={cfg.lambda_tradeoff}"
            ),
            expected_improvement=expected_improvement,
            latency_cost=latency_cost,
            confidence=min(1.0, len(self.improvement_tracker) / 10),
            details={
                "order_size": order_size,
                "volatility": sigma,
                "base_cost_estimate": base_cost_estimate,
                "cost_benefit_ratio": cost_benefit_ratio,
                "lambda": cfg.lambda_tradeoff,
                "depth_ratio": order_size / market_state.market_depth,
            },
        )

    def record_outcome(
        self, baseline_cost: float, optimized_cost: float, order_size: int, volatility: float
    ) -> None:
        self.improvement_tracker.record(
            baseline_cost=baseline_cost,
            optimized_cost=optimized_cost,
            order_size=order_size,
            volatility=volatility,
        )
