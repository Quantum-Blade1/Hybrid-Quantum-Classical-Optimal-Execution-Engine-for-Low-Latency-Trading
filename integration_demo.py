"""
Integrated QUBO Execution Demo

Demonstrates the complete workflow:
1. Generate market data
2. Formulate QUBO for optimal execution
3. Solve with simulated annealing
4. Execute optimized schedule
5. Compare against VWAP and TWAP
"""

import sys
from datetime import datetime

sys.path.insert(0, ".")

from src.market_data import MarketDataSimulator, MarketParams
from src.execution_engine import ParentOrder, OrderSide
from src.qubo_integration import (
    run_integrated_comparison,
    plot_strategy_comparison
)
from src.execution_engine import ExecutionEngine
from src.vwap_strategy import VWAPStrategy
from src.twap_strategy import TWAPStrategy
from src.qubo_integration import QUBOStrategy


def print_header(text: str):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def run_demo(
    order_size: int = 5000,
    num_minutes: int = 60,
    qubo_slices: int = 10,
    sa_sweeps: int = 500,
    seed: int = 42
):
    """
    Run the integrated QUBO execution demo.
    
    Args:
        order_size: Shares to execute
        num_minutes: Trading window length
        qubo_slices: Number of QUBO time buckets
        sa_sweeps: Simulated annealing iterations
        seed: Random seed
    """
    print_header("INTEGRATED QUBO EXECUTION DEMO")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # =========================================================================
    # Step 1: Generate Market Data
    # =========================================================================
    print("\n--- 1. Generating Market Data ---")
    
    params = MarketParams(
        symbol="AAPL",
        initial_price=175.00,
        annual_volatility=0.22
    )
    
    simulator = MarketDataSimulator(
        params=params,
        total_daily_volume=60_000_000,
        seed=seed
    )
    
    market_data = simulator.generate(datetime(2024, 1, 15), num_minutes=num_minutes)
    print(f"Generated {len(market_data)} minutes of market data")
    print(f"Price range: ${market_data['price'].min():.2f} - ${market_data['price'].max():.2f}")
    
    # =========================================================================
    # Step 2: Create Parent Order
    # =========================================================================
    print("\n--- 2. Creating Parent Order ---")
    
    order = ParentOrder(
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=order_size,
        time_horizon_minutes=num_minutes
    )
    
    print(f"Order: {order.side.value.upper()} {order.total_quantity:,} shares")
    print(f"Time Horizon: {order.time_horizon_minutes} minutes")
    
    # =========================================================================
    # Step 3: Run Integrated Comparison
    # =========================================================================
    print("\n--- 3. Running Strategy Comparison ---")
    
    comparison = run_integrated_comparison(
        parent_order=order,
        market_data=market_data,
        qubo_time_slices=qubo_slices,
        qubo_sa_sweeps=sa_sweeps,
        seed=seed,
        verbose=True
    )
    
    # =========================================================================
    # Step 4: Display Detailed Results
    # =========================================================================
    print("\n--- 4. Detailed Results ---")
    
    print("\n{:<25} {:>12} {:>12} {:>12}".format("Metric", "VWAP", "TWAP", "QUBO"))
    print("-" * 65)
    
    print("{:<25} {:>12.4f} {:>12.4f} {:>12.4f}".format(
        "Avg Exec Price ($)",
        comparison.vwap_report.average_execution_price,
        comparison.twap_report.average_execution_price,
        comparison.qubo_report.average_execution_price
    ))
    
    print("{:<25} {:>12.4f} {:>12.4f} {:>12.4f}".format(
        "Benchmark VWAP ($)",
        comparison.vwap_report.benchmark_vwap,
        comparison.twap_report.benchmark_vwap,
        comparison.qubo_report.benchmark_vwap
    ))
    
    print("{:<25} {:>+12.2f} {:>+12.2f} {:>+12.2f}".format(
        "Slippage (bps)",
        comparison.vwap_report.slippage_vs_vwap_bps,
        comparison.twap_report.slippage_vs_vwap_bps,
        comparison.qubo_report.slippage_vs_vwap_bps
    ))
    
    print("{:<25} {:>12.2f} {:>12.2f} {:>12.2f}".format(
        "Total Cost ($)",
        comparison.vwap_report.total_cost,
        comparison.twap_report.total_cost,
        comparison.qubo_report.total_cost
    ))
    
    print("{:<25} {:>12.2f} {:>12.2f} {:>12.2f}".format(
        "Spread Cost ($)",
        comparison.vwap_report.spread_cost,
        comparison.twap_report.spread_cost,
        comparison.qubo_report.spread_cost
    ))
    
    print("{:<25} {:>12.2f} {:>12.2f} {:>12.2f}".format(
        "Impact Cost ($)",
        comparison.vwap_report.impact_cost,
        comparison.twap_report.impact_cost,
        comparison.qubo_report.impact_cost
    ))
    
    print("{:<25} {:>12} {:>12} {:>12}".format(
        "Child Orders",
        comparison.vwap_report.num_child_orders,
        comparison.twap_report.num_child_orders,
        comparison.qubo_report.num_child_orders
    ))
    
    print("{:<25} {:>11.1%} {:>11.1%} {:>11.1%}".format(
        "Fill Rate",
        comparison.vwap_report.fill_rate,
        comparison.twap_report.fill_rate,
        comparison.qubo_report.fill_rate
    ))
    
    # =========================================================================
    # Step 5: Summary
    # =========================================================================
    print("\n--- 5. Summary ---")
    
    print(f"\n  Best Strategy: {comparison.best_strategy}")
    print(f"  QUBO Optimization Time: {comparison.qubo_optimization_time:.3f}s")
    print(f"  QUBO Energy: {comparison.qubo_energy:.4f}")
    print(f"  Constraints Satisfied: {'✓ Yes' if comparison.qubo_constraints_satisfied else '✗ No'}")
    print(f"  Cost Savings vs Baseline: ${comparison.cost_savings:.2f}")
    
    # =========================================================================
    # Step 6: Visualization
    # =========================================================================
    print("\n--- 6. Generating Visualization ---")
    
    try:
        # Re-run with fresh engines for visualization
        vwap_engine = ExecutionEngine(seed=seed)
        twap_engine = ExecutionEngine(seed=seed)
        qubo_engine = ExecutionEngine(seed=seed)
        
        plot_strategy_comparison(
            comparison=comparison,
            market_data=market_data,
            vwap_engine=vwap_engine,
            twap_engine=twap_engine,
            qubo_engine=qubo_engine,
            save_path="qubo_integration_comparison.png"
        )
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    print_header("Demo Complete")
    
    return comparison


if __name__ == "__main__":
    comparison = run_demo(
        order_size=5000,
        num_minutes=60,
        qubo_slices=10,
        sa_sweeps=500,
        seed=42
    )
