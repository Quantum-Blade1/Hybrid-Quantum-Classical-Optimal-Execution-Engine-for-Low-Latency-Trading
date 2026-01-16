"""
Quantum Optimization Decision Layer

Smart decision logic for when to invoke quantum/expensive optimization:

Decision Rule:
    invoke_quantum = (expected_improvement > lambda * latency_cost) 
                     AND (order_size > min_threshold)
                     AND (optimizer_available)

Inputs:
- Order size relative to market depth
- Current volatility (std of recent returns)
- Available optimization time
- Historical improvement from optimization
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Decision Configuration
# =============================================================================

@dataclass
class DecisionConfig:
    """
    Configuration for optimization decision logic.
    
    Attributes:
        lambda_tradeoff: Cost-benefit tradeoff parameter
                        Higher = more conservative (require higher expected improvement)
        min_order_size: Minimum order size to consider optimization
        min_volatility: Minimum volatility to trigger optimization
        max_latency_ms: Maximum acceptable optimization latency
        improvement_history_size: How many historical improvements to track
    """
    lambda_tradeoff: float = 1.0
    min_order_size: int = 1000
    min_depth_ratio: float = 0.01  # Order must be at least 1% of depth
    min_volatility: float = 0.001  # 0.1% min volatility
    max_latency_ms: float = 1000.0  # 1 second max
    improvement_history_size: int = 20
    
    # Cost coefficients
    latency_cost_per_ms: float = 0.001  # $ per ms of delay
    volatility_impact_weight: float = 0.5  # How much vol affects expected improvement


# =============================================================================
# Market State
# =============================================================================

@dataclass
class MarketState:
    """Current market conditions for decision making."""
    current_price: float
    bid_ask_spread: float
    market_depth: int  # Total shares at best bid/ask
    recent_volatility: float  # Std of recent returns
    volume_rate: float  # Shares per minute
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return (self.bid_ask_spread / self.current_price) * 10000
    
    @classmethod
    def from_market_data(cls, market_data: pd.DataFrame) -> 'MarketState':
        """Create MarketState from market data DataFrame."""
        latest = market_data.iloc[-1]
        
        # Calculate volatility from returns
        if len(market_data) > 1:
            returns = market_data['price'].pct_change().dropna()
            volatility = returns.std()
        else:
            volatility = 0.01
        
        return cls(
            current_price=latest['price'],
            bid_ask_spread=latest.get('spread', latest['price'] * 0.0005),
            market_depth=int(latest.get('volume', 100000)),
            recent_volatility=volatility,
            volume_rate=market_data['volume'].mean() if 'volume' in market_data.columns else 10000
        )


# =============================================================================
# Improvement Tracker
# =============================================================================

class ImprovementTracker:
    """
    Tracks historical improvement from optimization.
    
    Compares optimized vs baseline execution costs to estimate
    expected improvement for future decisions.
    """
    
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self._improvements: deque = deque(maxlen=max_history)
        self._contexts: deque = deque(maxlen=max_history)  # (order_size, volatility)
    
    def record(
        self, 
        baseline_cost: float, 
        optimized_cost: float,
        order_size: int,
        volatility: float
    ) -> None:
        """Record an improvement observation."""
        improvement = baseline_cost - optimized_cost
        improvement_pct = improvement / baseline_cost if baseline_cost > 0 else 0
        
        self._improvements.append({
            'absolute': improvement,
            'percentage': improvement_pct,
            'timestamp': datetime.now()
        })
        self._contexts.append((order_size, volatility))
        
        logger.debug(f"Recorded improvement: ${improvement:.2f} ({improvement_pct:.2%})")
    
    def expected_improvement(
        self, 
        order_size: int, 
        volatility: float,
        base_cost_estimate: float
    ) -> float:
        """
        Estimate expected improvement for given conditions.
        
        Uses weighted average of historical improvements,
        weighted by similarity to current conditions.
        """
        if len(self._improvements) == 0:
            # No history - use conservative estimate
            return base_cost_estimate * 0.01  # Assume 1% improvement
        
        # Calculate similarity-weighted average
        total_weight = 0.0
        weighted_improvement = 0.0
        
        for i, (hist_size, hist_vol) in enumerate(self._contexts):
            # Similarity based on order size and volatility
            size_ratio = min(order_size, hist_size) / max(order_size, hist_size)
            vol_ratio = min(volatility, hist_vol) / max(volatility, hist_vol + 1e-10)
            similarity = (size_ratio + vol_ratio) / 2
            
            # Time decay (recent observations more relevant)
            recency = (i + 1) / len(self._contexts)
            weight = similarity * recency
            
            improvement_pct = self._improvements[i]['percentage']
            weighted_improvement += weight * improvement_pct
            total_weight += weight
        
        avg_improvement_pct = weighted_improvement / total_weight if total_weight > 0 else 0.01
        
        return base_cost_estimate * avg_improvement_pct
    
    @property
    def mean_improvement_pct(self) -> float:
        """Mean percentage improvement."""
        if len(self._improvements) == 0:
            return 0.01
        return np.mean([imp['percentage'] for imp in self._improvements])


# =============================================================================
# Decision Engine
# =============================================================================

@dataclass
class DecisionResult:
    """Result of optimization decision."""
    invoke_optimization: bool
    reason: str
    expected_improvement: float
    latency_cost: float
    confidence: float  # 0-1 confidence in decision
    details: Dict = field(default_factory=dict)


class OptimizationDecisionEngine:
    """
    Smart decision engine for when to invoke optimization.
    
    Decision Rule:
        invoke = (E[improvement] > λ * latency_cost) 
                 AND (order_size > min_threshold)
                 AND (optimizer_available)
    """
    
    def __init__(self, config: Optional[DecisionConfig] = None):
        self.config = config or DecisionConfig()
        self.improvement_tracker = ImprovementTracker(
            max_history=self.config.improvement_history_size
        )
        
        # State
        self._optimizer_available = True
        self._last_optimization_time: Optional[datetime] = None
        
        # Statistics
        self.decisions_made = 0
        self.optimizations_invoked = 0
    
    def set_optimizer_available(self, available: bool) -> None:
        """Update optimizer availability status."""
        self._optimizer_available = available
    
    def decide(
        self,
        order_size: int,
        market_state: MarketState,
        optimization_latency_ms: float = 500.0
    ) -> DecisionResult:
        """
        Decide whether to invoke optimization.
        
        Args:
            order_size: Size of order to execute
            market_state: Current market conditions
            optimization_latency_ms: Expected optimization time
            
        Returns:
            DecisionResult with recommendation
        """
        self.decisions_made += 1
        
        # Check hard thresholds first
        
        # 1. Minimum order size
        if order_size < self.config.min_order_size:
            return DecisionResult(
                invoke_optimization=False,
                reason=f"Order size {order_size} below minimum {self.config.min_order_size}",
                expected_improvement=0,
                latency_cost=0,
                confidence=1.0
            )
        
        # 2. Optimizer availability
        if not self._optimizer_available:
            return DecisionResult(
                invoke_optimization=False,
                reason="Optimizer not available",
                expected_improvement=0,
                latency_cost=0,
                confidence=1.0
            )
        
        # 3. Max latency constraint
        if optimization_latency_ms > self.config.max_latency_ms:
            return DecisionResult(
                invoke_optimization=False,
                reason=f"Latency {optimization_latency_ms}ms exceeds max {self.config.max_latency_ms}ms",
                expected_improvement=0,
                latency_cost=0,
                confidence=1.0
            )
        
        # Calculate expected improvement
        # Base cost estimate: order_size * spread_cost + impact_cost
        base_spread_cost = order_size * market_state.bid_ask_spread / 2
        base_impact_cost = order_size * market_state.current_price * market_state.recent_volatility * 0.1
        base_cost_estimate = base_spread_cost + base_impact_cost
        
        # Adjust expected improvement based on volatility
        vol_adjusted_improvement = self.improvement_tracker.expected_improvement(
            order_size=order_size,
            volatility=market_state.recent_volatility,
            base_cost_estimate=base_cost_estimate
        )
        
        # Higher volatility = higher potential for improvement
        vol_bonus = market_state.recent_volatility * self.config.volatility_impact_weight * base_cost_estimate
        expected_improvement = vol_adjusted_improvement + vol_bonus
        
        # Calculate latency cost
        # Cost of delay during volatile period
        latency_cost = (
            optimization_latency_ms * self.config.latency_cost_per_ms +
            optimization_latency_ms / 1000 * market_state.recent_volatility * market_state.current_price * order_size * 0.001
        )
        
        # Apply decision rule
        cost_benefit_ratio = expected_improvement / (latency_cost + 1e-10)
        invoke = cost_benefit_ratio > self.config.lambda_tradeoff
        
        # Confidence based on historical data
        confidence = min(1.0, len(self.improvement_tracker._improvements) / 10)
        
        if invoke:
            self.optimizations_invoked += 1
        
        details = {
            'order_size': order_size,
            'volatility': market_state.recent_volatility,
            'base_cost_estimate': base_cost_estimate,
            'cost_benefit_ratio': cost_benefit_ratio,
            'lambda': self.config.lambda_tradeoff,
            'depth_ratio': order_size / market_state.market_depth
        }
        
        reason = (
            f"CBR={cost_benefit_ratio:.2f} {'>' if invoke else '<='} λ={self.config.lambda_tradeoff}"
        )
        
        return DecisionResult(
            invoke_optimization=invoke,
            reason=reason,
            expected_improvement=expected_improvement,
            latency_cost=latency_cost,
            confidence=confidence,
            details=details
        )
    
    def record_outcome(
        self,
        baseline_cost: float,
        optimized_cost: float,
        order_size: int,
        volatility: float
    ) -> None:
        """Record outcome for future learning."""
        self.improvement_tracker.record(
            baseline_cost=baseline_cost,
            optimized_cost=optimized_cost,
            order_size=order_size,
            volatility=volatility
        )


# =============================================================================
# Lambda Sensitivity Analysis
# =============================================================================

def test_lambda_sensitivity(
    lambda_values: List[float] = [0.5, 1.0, 2.0],
    num_scenarios: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """
    Test decision engine with different lambda values.
    
    Args:
        lambda_values: Lambda values to test
        num_scenarios: Number of random scenarios
        seed: Random seed
        
    Returns:
        DataFrame with results for each lambda
    """
    np.random.seed(seed)
    
    results = []
    
    for lam in lambda_values:
        config = DecisionConfig(lambda_tradeoff=lam)
        engine = OptimizationDecisionEngine(config)
        
        # Seed with some historical improvements
        for _ in range(5):
            engine.improvement_tracker.record(
                baseline_cost=100 + np.random.normal(0, 10),
                optimized_cost=95 + np.random.normal(0, 8),
                order_size=5000,
                volatility=0.02
            )
        
        invoke_count = 0
        total_expected = 0
        total_latency = 0
        
        for _ in range(num_scenarios):
            # Random market conditions
            market_state = MarketState(
                current_price=100 + np.random.normal(0, 5),
                bid_ask_spread=0.05 + np.random.exponential(0.02),
                market_depth=int(100000 + np.random.normal(0, 20000)),
                recent_volatility=0.001 + np.random.exponential(0.01),
                volume_rate=10000 + np.random.normal(0, 2000)
            )
            
            order_size = int(1000 + np.random.exponential(5000))
            latency_ms = 200 + np.random.exponential(300)
            
            decision = engine.decide(
                order_size=order_size,
                market_state=market_state,
                optimization_latency_ms=latency_ms
            )
            
            if decision.invoke_optimization:
                invoke_count += 1
                total_expected += decision.expected_improvement
                total_latency += decision.latency_cost
        
        results.append({
            'lambda': lam,
            'invoke_rate': invoke_count / num_scenarios,
            'avg_expected_improvement': total_expected / max(1, invoke_count),
            'avg_latency_cost': total_latency / max(1, invoke_count),
            'total_invocations': invoke_count
        })
    
    return pd.DataFrame(results)


# =============================================================================
# Demo
# =============================================================================

def run_decision_demo():
    """Demonstrate decision layer."""
    
    print("\n" + "="*70)
    print(" Quantum Optimization Decision Layer Demo")
    print("="*70)
    
    # Create decision engine
    config = DecisionConfig(lambda_tradeoff=1.0)
    engine = OptimizationDecisionEngine(config)
    
    # Seed with historical improvements
    print("\n Seeding with historical improvement data...")
    for i in range(10):
        baseline = 100 + np.random.normal(0, 15)
        improvement_pct = 0.05 + np.random.normal(0, 0.02)  # ~5% improvement
        optimized = baseline * (1 - improvement_pct)
        engine.record_outcome(
            baseline_cost=baseline,
            optimized_cost=optimized,
            order_size=5000 + i * 500,
            volatility=0.015 + np.random.normal(0, 0.005)
        )
    
    print(f" Historical avg improvement: {engine.improvement_tracker.mean_improvement_pct:.1%}")
    
    # Test scenarios
    print("\n Testing decision scenarios:")
    print("-" * 70)
    
    scenarios = [
        {"order_size": 500, "vol": 0.01, "latency": 500, "name": "Small order"},
        {"order_size": 5000, "vol": 0.005, "latency": 500, "name": "Normal, low vol"},
        {"order_size": 5000, "vol": 0.025, "latency": 500, "name": "Normal, high vol"},
        {"order_size": 20000, "vol": 0.02, "latency": 500, "name": "Large order"},
        {"order_size": 5000, "vol": 0.02, "latency": 2000, "name": "High latency"},
    ]
    
    for s in scenarios:
        market = MarketState(
            current_price=150.0,
            bid_ask_spread=0.05,
            market_depth=500000,
            recent_volatility=s["vol"],
            volume_rate=50000
        )
        
        decision = engine.decide(
            order_size=s["order_size"],
            market_state=market,
            optimization_latency_ms=s["latency"]
        )
        
        invoke_str = "✓ INVOKE" if decision.invoke_optimization else "✗ SKIP"
        print(f" {s['name']:<20} {invoke_str:<10} | {decision.reason}")
    
    # Lambda sensitivity
    print("\n Lambda Sensitivity Analysis:")
    print("-" * 70)
    
    sensitivity_df = test_lambda_sensitivity([0.5, 1.0, 2.0], num_scenarios=200)
    print(sensitivity_df.to_string(index=False))
    
    print("\n Interpretation:")
    print("   λ=0.5 (aggressive): More frequent optimization, higher compute cost")
    print("   λ=1.0 (balanced):   Standard cost-benefit balance")
    print("   λ=2.0 (conservative): Optimize only when clear benefit")
    
    print("\n" + "="*70)
    
    return engine, sensitivity_df


if __name__ == "__main__":
    engine, df = run_decision_demo()
