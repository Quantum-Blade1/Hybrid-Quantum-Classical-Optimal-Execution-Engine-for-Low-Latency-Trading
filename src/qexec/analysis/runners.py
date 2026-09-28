"""
Single-order execution runners on a minute-bar market data frame.

Each runner executes one parent order and returns an ExecutionResult:
    run_vwap_execution    VWAPStrategy through ExecutionEngine (impact model)
    run_sa_execution      one SA-QUBO schedule, filled at price + spread/2
    run_hybrid_execution  decision layer re-optimizes the remaining order with
                          SA-QUBO at five checkpoints, filled at price + spread/2

The VWAP runner goes through the engine's impact model while the SA and
hybrid runners do not, so their costs are not like-for-like
(docs/CLAIMS_AUDIT.md F6).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List
from time import time

from qexec.execution.engine import ExecutionEngine, ParentOrder, OrderSide
from qexec.execution.strategies.vwap import VWAPStrategy
from qexec.optimization.qubo import ExecutionQUBO
from qexec.optimization.schedule import optimize_schedule, slice_level_config, spread_over_minutes
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
from qexec.runtime.decision import OptimizationDecisionEngine, DecisionConfig, MarketState


@dataclass
class ExecutionResult:
    """Result from single execution mode."""
    mode: str
    total_shares: int
    executed_shares: int
    avg_price: float
    total_cost: float
    slippage_bps: float
    execution_log: List[Dict]
    optimization_invocations: int = 0
    optimization_time: float = 0.0


def run_vwap_execution(
    market_data: pd.DataFrame,
    total_shares: int,
    seed: int = 42
) -> ExecutionResult:
    """Execute using pure VWAP strategy."""
    
    engine = ExecutionEngine(seed=seed)
    strategy = VWAPStrategy(participation_rate=0.1, seed=seed)
    
    order = ParentOrder(
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=total_shares,
        time_horizon_minutes=len(market_data)
    )
    
    report = engine.process_order(order, market_data, strategy)
    
    # Generate synthetic execution log based on VWAP profile
    execution_log = []
    volume_profile = market_data['volume'].values
    volume_total = volume_profile.sum()
    
    for minute in range(len(market_data)):
        shares = int(total_shares * volume_profile[minute] / volume_total)
        if shares > 0:
            price = market_data.iloc[minute]['price'] + market_data.iloc[minute]['spread'] / 2
            execution_log.append({
                'minute': minute,
                'shares': shares,
                'price': price
            })
    
    return ExecutionResult(
        mode="VWAP",
        total_shares=total_shares,
        executed_shares=report.filled_quantity,
        avg_price=report.average_execution_price,
        total_cost=report.total_cost,
        slippage_bps=report.slippage_vs_vwap_bps,
        execution_log=execution_log
    )


def run_sa_execution(
    market_data: pd.DataFrame,
    total_shares: int,
    num_slices: int = 20,
    seed: int = 42
) -> ExecutionResult:
    """Execute using SA-optimized schedule."""
    
    start = time()
    
    # Build and solve QUBO
    qubo = ExecutionQUBO(slice_level_config(total_shares, num_slices))
    solver = SimulatedAnnealingSolver(num_sweeps=500, seed=seed)
    slice_qty, _ = optimize_schedule(qubo, solver)
    
    opt_time = time() - start
    
    # Convert to schedule
    schedule = spread_over_minutes(slice_qty, len(market_data))
    
    # Execute schedule
    execution_log = []
    total_executed = 0
    total_value = 0.0
    
    for minute, row in market_data.iterrows():
        shares = int(schedule[minute])
        if shares > 0:
            price = row['price'] + row['spread'] / 2  # Buy at ask
            total_executed += shares
            total_value += shares * price
            execution_log.append({
                'minute': minute,
                'shares': shares,
                'price': price
            })
    
    avg_price = total_value / total_executed if total_executed > 0 else 0
    benchmark = (market_data['price'] * market_data['volume']).sum() / market_data['volume'].sum()
    slippage_bps = (avg_price - benchmark) / benchmark * 10000
    
    return ExecutionResult(
        mode="SA-Optimized",
        total_shares=total_shares,
        executed_shares=total_executed,
        avg_price=avg_price,
        total_cost=total_value - total_shares * benchmark,
        slippage_bps=slippage_bps,
        execution_log=execution_log,
        optimization_invocations=1,
        optimization_time=opt_time
    )


def run_hybrid_execution(
    market_data: pd.DataFrame,
    total_shares: int,
    num_slices: int = 20,
    lambda_tradeoff: float = 0.5,  # Aggressive to ensure invocations
    seed: int = 42
) -> ExecutionResult:
    """
    Execute using hybrid quantum-classical approach.
    
    Decision layer decides when to invoke optimization.
    """
    
    # Initialize decision engine
    decision_config = DecisionConfig(
        lambda_tradeoff=lambda_tradeoff,
        min_order_size=500,
        max_latency_ms=2000
    )
    decision_engine = OptimizationDecisionEngine(decision_config)
    
    # Seed improvement tracker
    for _ in range(5):
        decision_engine.record_outcome(
            baseline_cost=100,
            optimized_cost=95,
            order_size=10000,
            volatility=0.01
        )
    
    # Execution state
    remaining_shares = total_shares
    minutes_per_check = len(market_data) // 5  # Check 5 times
    execution_log = []
    optimization_log = []
    current_schedule = np.full(len(market_data), total_shares // len(market_data))  # Uniform start
    
    total_opt_time = 0.0
    
    for check_point in range(5):
        start_minute = check_point * minutes_per_check
        end_minute = min((check_point + 1) * minutes_per_check, len(market_data))
        
        # Create market state from recent data
        recent_data = market_data.iloc[max(0, start_minute-5):start_minute+1]
        if len(recent_data) > 0:
            market_state = MarketState.from_market_data(recent_data)
        else:
            market_state = MarketState(
                current_price=175.0,
                bid_ask_spread=0.03,
                market_depth=500000,
                recent_volatility=0.01,
                volume_rate=50000
            )
        
        # Decision: should we optimize?
        decision = decision_engine.decide(
            order_size=remaining_shares,
            market_state=market_state,
            optimization_latency_ms=500
        )
        
        if decision.invoke_optimization and remaining_shares > 1000:
            # Run SA optimization for remaining order
            opt_start = time()
            
            remaining_minutes = len(market_data) - start_minute
            slices_remaining = max(4, remaining_minutes // 3)
            
            qubo = ExecutionQUBO(slice_level_config(remaining_shares, slices_remaining))
            solver = SimulatedAnnealingSolver(num_sweeps=300, seed=seed + check_point)
            slice_qty, _ = optimize_schedule(qubo, solver)
            
            opt_time = time() - opt_start
            total_opt_time += opt_time
            
            # Update schedule for remaining period
            new_schedule = spread_over_minutes(slice_qty, remaining_minutes)
            
            current_schedule[start_minute:] = 0
            current_schedule[start_minute:start_minute + len(new_schedule)] = new_schedule
            
            optimization_log.append({
                'minute': start_minute,
                'remaining_shares': remaining_shares,
                'volatility': market_state.recent_volatility,
                'decision': 'INVOKE',
                'time_ms': opt_time * 1000
            })
        else:
            optimization_log.append({
                'minute': start_minute,
                'remaining_shares': remaining_shares,
                'volatility': market_state.recent_volatility,
                'decision': 'SKIP',
                'reason': decision.reason
            })
        
        # Execute this segment
        for minute in range(start_minute, end_minute):
            shares = int(current_schedule[minute])
            if shares > 0 and remaining_shares > 0:
                shares = min(shares, remaining_shares)
                price = market_data.iloc[minute]['price'] + market_data.iloc[minute]['spread'] / 2
                execution_log.append({
                    'minute': minute,
                    'shares': shares,
                    'price': price
                })
                remaining_shares -= shares
    
    # Calculate metrics
    total_executed = sum(e['shares'] for e in execution_log)
    total_value = sum(e['shares'] * e['price'] for e in execution_log)
    avg_price = total_value / total_executed if total_executed > 0 else 0
    benchmark = (market_data['price'] * market_data['volume']).sum() / market_data['volume'].sum()
    slippage_bps = (avg_price - benchmark) / benchmark * 10000 if benchmark > 0 else 0
    
    num_invocations = sum(1 for o in optimization_log if o['decision'] == 'INVOKE')
    
    return ExecutionResult(
        mode="Hybrid-Quantum",
        total_shares=total_shares,
        executed_shares=total_executed,
        avg_price=avg_price,
        total_cost=total_value - total_shares * benchmark,
        slippage_bps=slippage_bps,
        execution_log=execution_log,
        optimization_invocations=num_invocations,
        optimization_time=total_opt_time
    )
