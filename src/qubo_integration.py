"""
Integrated QUBO Execution Optimizer

This module connects the QUBO optimization framework with the execution engine,
providing an end-to-end workflow for optimal trade execution:

1. Takes market data and parent order
2. Formulates QUBO for optimal slicing
3. Solves using simulated annealing
4. Executes the optimized schedule
5. Compares results vs VWAP/TWAP
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from time import time

from .qubo_execution import QUBOConfig, ExecutionQUBO
from .qubo_solvers import SimulatedAnnealingSolver, QUBOResult
from .execution_engine import ExecutionEngine, ParentOrder, OrderSide, ExecutionReport
from .base_strategy import BaseStrategy, ExecutionMetrics
from .vwap_strategy import VWAPStrategy
from .twap_strategy import TWAPStrategy
from .order_book import OrderBook


# =============================================================================
# QUBO-Based Strategy
# =============================================================================

class QUBOStrategy(BaseStrategy):
    """
    Execution strategy that uses QUBO optimization.
    
    This strategy formulates the execution problem as a QUBO,
    solves it using simulated annealing, and returns the optimized
    schedule.
    
    Key parameters:
    - num_time_slices: Number of time buckets to divide the order into
    - num_venues: Number of execution venues
    - quantity_levels: Discrete quantity options per slice
    """
    
    strategy_name = "QUBO"
    
    def __init__(
        self,
        num_time_slices: int = 20,
        num_venues: int = 1,
        quantity_levels: Optional[List[int]] = None,
        sa_sweeps: int = 1000,
        seed: Optional[int] = None
    ):
        """
        Initialize QUBO strategy.
        
        Args:
            num_time_slices: Number of execution buckets
            num_venues: Number of venues (default 1 for simplicity)
            quantity_levels: Quantity options per slice
            sa_sweeps: Simulated annealing iterations
            seed: Random seed
        """
        self.num_time_slices = num_time_slices
        self.num_venues = num_venues
        self.quantity_levels = quantity_levels
        self.sa_sweeps = sa_sweeps
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Store optimization results
        self.qubo: Optional[ExecutionQUBO] = None
        self.qubo_result: Optional[QUBOResult] = None
        self.optimization_time: float = 0.0
        self._execution_slices: List = []
    
    def calculate_schedule(
        self,
        total_shares: int,
        market_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate execution schedule using QUBO optimization.
        
        Args:
            total_shares: Total shares to execute
            market_data: Market data DataFrame
            
        Returns:
            Schedule array with shares per minute
        """
        start_time = time()
        
        num_minutes = len(market_data)
        
        # Determine quantity levels if not provided
        if self.quantity_levels is None:
            # Create levels based on order size
            base_qty = total_shares // (self.num_time_slices * 2)
            self.quantity_levels = [
                0,
                base_qty,
                base_qty * 2,
                base_qty * 3,
                base_qty * 4
            ]
        
        # Create QUBO configuration
        config = QUBOConfig(
            total_shares=total_shares,
            num_time_slices=self.num_time_slices,
            num_venues=self.num_venues,
            quantity_levels=self.quantity_levels,
            impact_weight=0.4,
            timing_weight=0.3,
            transaction_weight=0.3,
            equality_penalty=1000.0,
            capacity_penalty=500.0,
            max_shares_per_slice=total_shares // 3
        )
        
        # Build QUBO
        self.qubo = ExecutionQUBO(config)
        Q = self.qubo.build_qubo_matrix()
        
        # Solve with simulated annealing
        solver = SimulatedAnnealingSolver(
            num_sweeps=self.sa_sweeps,
            initial_temp=10.0,
            final_temp=0.01,
            cooling_rate=0.95,
            seed=self.seed
        )
        
        self.qubo_result = solver.solve(Q)
        self.optimization_time = time() - start_time
        
        # Convert QUBO solution to schedule
        solution_df = self.qubo.interpret_solution(self.qubo_result.solution)
        
        # Map QUBO time slices to market data minutes
        schedule = np.zeros(num_minutes)
        minutes_per_slice = num_minutes // self.num_time_slices
        
        for _, row in solution_df.iterrows():
            t = row["time_slice"]
            q = row["quantity"]
            
            # Distribute quantity across minutes in this slice
            start_min = t * minutes_per_slice
            end_min = min((t + 1) * minutes_per_slice, num_minutes)
            
            if end_min > start_min:
                shares_per_min = q // (end_min - start_min)
                remainder = q % (end_min - start_min)
                
                for m in range(start_min, end_min):
                    schedule[m] = shares_per_min
                schedule[start_min] += remainder  # Add remainder to first minute
        
        return schedule
    
    def get_execution_summary(self) -> pd.DataFrame:
        """Get summary of execution slices."""
        if not self._execution_slices:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            "minute": s.minute,
            "target_qty": s.target_quantity,
            "filled_qty": s.filled_quantity,
            "exec_price": s.execution_price,
            "market_impact": s.market_impact
        } for s in self._execution_slices])
    
    def get_optimization_stats(self) -> Dict:
        """Get QUBO optimization statistics."""
        if self.qubo_result is None:
            return {}
        
        costs = self.qubo.calculate_solution_cost(self.qubo_result.solution)
        validation = self.qubo.validate_solution(self.qubo_result.solution)
        
        return {
            "qubo_energy": self.qubo_result.energy,
            "sa_evaluations": self.qubo_result.num_evaluations,
            "sa_iterations": self.qubo_result.iterations,
            "optimization_time_s": self.optimization_time,
            "constraints_satisfied": validation["all_satisfied"],
            **costs
        }


# =============================================================================
# Integrated Workflow
# =============================================================================

@dataclass
class StrategyComparison:
    """Results from comparing multiple execution strategies."""
    vwap_report: ExecutionReport
    twap_report: ExecutionReport
    qubo_report: ExecutionReport
    
    # QUBO optimization stats
    qubo_optimization_time: float
    qubo_energy: float
    qubo_constraints_satisfied: bool
    
    # Computed metrics
    best_strategy: str = ""
    cost_savings: float = 0.0
    
    def __post_init__(self):
        """Compute comparison metrics."""
        costs = {
            "VWAP": self.vwap_report.total_cost,
            "TWAP": self.twap_report.total_cost,
            "QUBO": self.qubo_report.total_cost
        }
        
        self.best_strategy = min(costs, key=costs.get)
        baseline = (costs["VWAP"] + costs["TWAP"]) / 2
        self.cost_savings = baseline - costs["QUBO"]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to comparison DataFrame."""
        reports = {
            "VWAP": self.vwap_report,
            "TWAP": self.twap_report,
            "QUBO": self.qubo_report
        }
        
        data = []
        metrics = [
            ("Avg Exec Price ($)", "average_execution_price", ":.4f"),
            ("Benchmark Price ($)", "benchmark_vwap", ":.4f"),
            ("Slippage (bps)", "slippage_vs_vwap_bps", ":+.2f"),
            ("Total Cost ($)", "total_cost", ":.2f"),
            ("Spread Cost ($)", "spread_cost", ":.2f"),
            ("Impact Cost ($)", "impact_cost", ":.2f"),
            ("Timing Risk ($)", "timing_risk", ":.4f"),
            ("Fill Rate (%)", "fill_rate", ".1%"),
            ("Child Orders", "num_child_orders", "d"),
            ("Exec Time (min)", "execution_time_minutes", "d")
        ]
        
        for name, attr, fmt in metrics:
            row = {"Metric": name}
            for strategy, report in reports.items():
                val = getattr(report, attr)
                if fmt == ".1%":
                    row[strategy] = f"{val:.1%}"
                elif fmt == "d":
                    row[strategy] = str(int(val))
                elif fmt == ":.4f":
                    row[strategy] = f"{val:.4f}"
                elif fmt == ":.2f":
                    row[strategy] = f"{val:.2f}"
                elif fmt == ":+.2f":
                    row[strategy] = f"{val:+.2f}"
                else:
                    row[strategy] = str(val)
            data.append(row)
        
        return pd.DataFrame(data)


def run_integrated_comparison(
    parent_order: ParentOrder,
    market_data: pd.DataFrame,
    qubo_time_slices: int = 20,
    qubo_sa_sweeps: int = 1000,
    seed: int = 42,
    verbose: bool = True
) -> StrategyComparison:
    """
    Run integrated comparison of QUBO vs VWAP vs TWAP.
    
    Args:
        parent_order: Order to execute
        market_data: Market data for simulation
        qubo_time_slices: Number of QUBO time buckets
        qubo_sa_sweeps: SA iterations for QUBO
        seed: Random seed
        verbose: Print progress
        
    Returns:
        StrategyComparison with all results
    """
    if verbose:
        print("\n" + "=" * 70)
        print(" Integrated QUBO Execution Optimization")
        print("=" * 70)
        print(f" Order: {parent_order.side.value.upper()} {parent_order.total_quantity:,} {parent_order.symbol}")
        print(f" Market Data: {len(market_data)} minutes")
    
    # Initialize execution engines
    vwap_engine = ExecutionEngine(seed=seed)
    twap_engine = ExecutionEngine(seed=seed)
    qubo_engine = ExecutionEngine(seed=seed)
    
    # Initialize strategies
    vwap_strategy = VWAPStrategy(participation_rate=0.1, seed=seed)
    twap_strategy = TWAPStrategy(interval_minutes=1, seed=seed)
    qubo_strategy = QUBOStrategy(
        num_time_slices=qubo_time_slices,
        sa_sweeps=qubo_sa_sweeps,
        seed=seed
    )
    
    # Execute with VWAP
    if verbose:
        print("\n--- Executing VWAP ---")
    vwap_order = ParentOrder(
        symbol=parent_order.symbol,
        side=parent_order.side,
        total_quantity=parent_order.total_quantity,
        time_horizon_minutes=len(market_data)
    )
    vwap_report = vwap_engine.process_order(vwap_order, market_data, vwap_strategy)
    if verbose:
        print(f" Cost: ${vwap_report.total_cost:.2f}, Slippage: {vwap_report.slippage_vs_vwap_bps:+.2f} bps")
    
    # Execute with TWAP
    if verbose:
        print("\n--- Executing TWAP ---")
    twap_order = ParentOrder(
        symbol=parent_order.symbol,
        side=parent_order.side,
        total_quantity=parent_order.total_quantity,
        time_horizon_minutes=len(market_data)
    )
    twap_report = twap_engine.process_order(twap_order, market_data, twap_strategy)
    if verbose:
        print(f" Cost: ${twap_report.total_cost:.2f}, Slippage: {twap_report.slippage_vs_vwap_bps:+.2f} bps")
    
    # Execute with QUBO
    if verbose:
        print("\n--- Executing QUBO ---")
        print(f" Optimizing with {qubo_time_slices} time slices, {qubo_sa_sweeps} SA sweeps...")
    
    qubo_order = ParentOrder(
        symbol=parent_order.symbol,
        side=parent_order.side,
        total_quantity=parent_order.total_quantity,
        time_horizon_minutes=len(market_data)
    )
    qubo_report = qubo_engine.process_order(qubo_order, market_data, qubo_strategy)
    
    qubo_stats = qubo_strategy.get_optimization_stats()
    if verbose:
        print(f" Optimization time: {qubo_stats.get('optimization_time_s', 0):.3f}s")
        print(f" Cost: ${qubo_report.total_cost:.2f}, Slippage: {qubo_report.slippage_vs_vwap_bps:+.2f} bps")
    
    # Create comparison result
    comparison = StrategyComparison(
        vwap_report=vwap_report,
        twap_report=twap_report,
        qubo_report=qubo_report,
        qubo_optimization_time=qubo_stats.get("optimization_time_s", 0),
        qubo_energy=qubo_stats.get("qubo_energy", 0),
        qubo_constraints_satisfied=qubo_stats.get("constraints_satisfied", False)
    )
    
    if verbose:
        print("\n" + "=" * 70)
        print(f" Best Strategy: {comparison.best_strategy}")
        print(f" QUBO Cost Savings vs Baseline: ${comparison.cost_savings:.2f}")
        print("=" * 70)
    
    return comparison


def plot_strategy_comparison(
    comparison: StrategyComparison,
    market_data: pd.DataFrame,
    vwap_engine: ExecutionEngine,
    twap_engine: ExecutionEngine,
    qubo_engine: ExecutionEngine,
    save_path: Optional[str] = None
) -> None:
    """
    Plot strategy comparison visualization.
    
    Args:
        comparison: Strategy comparison results
        market_data: Market data
        vwap_engine: VWAP execution engine
        twap_engine: TWAP execution engine
        qubo_engine: QUBO execution engine
        save_path: Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Price and Execution
    ax1 = axes[0, 0]
    ax1.plot(market_data["price"], label="Market Price", color="blue", alpha=0.7)
    ax1.axhline(comparison.vwap_report.benchmark_vwap, color="gray", 
                linestyle="--", alpha=0.5, label="VWAP Benchmark")
    ax1.axhline(comparison.vwap_report.average_execution_price, color="green",
                linestyle="-", alpha=0.8, label=f"VWAP Avg: ${comparison.vwap_report.average_execution_price:.2f}")
    ax1.axhline(comparison.twap_report.average_execution_price, color="orange",
                linestyle="-", alpha=0.8, label=f"TWAP Avg: ${comparison.twap_report.average_execution_price:.2f}")
    ax1.axhline(comparison.qubo_report.average_execution_price, color="red",
                linestyle="-", alpha=0.8, label=f"QUBO Avg: ${comparison.qubo_report.average_execution_price:.2f}")
    ax1.set_xlabel("Minute")
    ax1.set_ylabel("Price ($)")
    ax1.set_title("Execution Price Comparison")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cost Comparison Bar Chart
    ax2 = axes[0, 1]
    strategies = ["VWAP", "TWAP", "QUBO"]
    costs = [comparison.vwap_report.total_cost, 
             comparison.twap_report.total_cost,
             comparison.qubo_report.total_cost]
    colors = ["green", "orange", "red"]
    bars = ax2.bar(strategies, costs, color=colors, alpha=0.7)
    ax2.set_ylabel("Total Cost ($)")
    ax2.set_title("Total Execution Cost")
    for bar, cost in zip(bars, costs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"${cost:.2f}", ha="center", va="bottom", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    
    # Plot 3: Slippage Comparison
    ax3 = axes[1, 0]
    slippages = [comparison.vwap_report.slippage_vs_vwap_bps,
                 comparison.twap_report.slippage_vs_vwap_bps,
                 comparison.qubo_report.slippage_vs_vwap_bps]
    bars = ax3.bar(strategies, slippages, color=colors, alpha=0.7)
    ax3.set_ylabel("Slippage (bps)")
    ax3.set_title("Slippage vs VWAP Benchmark")
    ax3.axhline(0, color="black", linestyle="-", linewidth=0.5)
    for bar, slip in zip(bars, slippages):
        y_pos = bar.get_height() + 0.2 if bar.get_height() >= 0 else bar.get_height() - 0.5
        ax3.text(bar.get_x() + bar.get_width()/2, y_pos,
                f"{slip:+.2f}", ha="center", va="bottom" if bar.get_height() >= 0 else "top", fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")
    
    # Plot 4: Cost Breakdown
    ax4 = axes[1, 1]
    x = np.arange(len(strategies))
    width = 0.25
    
    spread_costs = [comparison.vwap_report.spread_cost,
                    comparison.twap_report.spread_cost,
                    comparison.qubo_report.spread_cost]
    impact_costs = [comparison.vwap_report.impact_cost,
                    comparison.twap_report.impact_cost,
                    comparison.qubo_report.impact_cost]
    
    ax4.bar(x - width/2, spread_costs, width, label="Spread Cost", color="lightblue")
    ax4.bar(x + width/2, impact_costs, width, label="Impact Cost", color="salmon")
    ax4.set_xticks(x)
    ax4.set_xticklabels(strategies)
    ax4.set_ylabel("Cost ($)")
    ax4.set_title("Cost Breakdown by Component")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    
    plt.show()
