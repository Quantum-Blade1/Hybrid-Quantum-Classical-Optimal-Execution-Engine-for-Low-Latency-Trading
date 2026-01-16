"""
Execution Engine Demo

Demonstrates the modular execution engine with comprehensive reporting.
"""

import sys
from datetime import datetime

sys.path.insert(0, ".")

from src.market_data import MarketDataSimulator, MarketParams
from src.execution_engine import ExecutionEngine, ParentOrder, OrderSide
from src.vwap_strategy import VWAPStrategy
from src.twap_strategy import TWAPStrategy


def print_header(text: str, width: int = 70):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(f" {text}")
    print("=" * width)


def print_section(text: str):
    """Print a section header."""
    print(f"\n--- {text} ---")


def run_demo(seed: int = 42):
    """Run the execution engine demo."""
    
    print_header("Execution Engine Framework Demo")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # =========================================================================
    # Step 1: Generate Market Data
    # =========================================================================
    print_section("Generating Market Data")
    
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
    
    market_data = simulator.generate(datetime(2024, 1, 15), num_minutes=390)
    print(f"Generated {len(market_data)} minutes of AAPL market data")
    
    # =========================================================================
    # Step 2: Initialize Execution Engine
    # =========================================================================
    print_section("Initializing Execution Engine")
    
    engine = ExecutionEngine(seed=seed)
    print("Execution engine ready")
    
    # =========================================================================
    # Step 3: Execute Order with VWAP Strategy
    # =========================================================================
    print_section("Executing VWAP Order")
    
    vwap_order = ParentOrder(
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=10_000,
        time_horizon_minutes=390,
        strategy_name="VWAP",
        urgency=0.5
    )
    
    print(f"Parent Order: {vwap_order.side.value.upper()} {vwap_order.total_quantity:,} shares")
    
    vwap_strategy = VWAPStrategy(
        participation_rate=0.10,
        max_slice_pct=0.05,
        seed=seed
    )
    
    vwap_report = engine.process_order(vwap_order, market_data, vwap_strategy)
    
    print("\nVWAP Execution Report:")
    print(vwap_report)
    
    # =========================================================================
    # Step 4: Execute Order with TWAP Strategy
    # =========================================================================
    print_section("Executing TWAP Order")
    
    twap_order = ParentOrder(
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=10_000,
        time_horizon_minutes=390,
        strategy_name="TWAP"
    )
    
    twap_strategy = TWAPStrategy(
        interval_minutes=1,
        max_slice_pct=0.05,
        seed=seed
    )
    
    twap_report = engine.process_order(twap_order, market_data, twap_strategy)
    
    print("\nTWAP Execution Report:")
    print(twap_report)
    
    # =========================================================================
    # Step 5: Compare Results
    # =========================================================================
    print_section("Strategy Comparison")
    
    print("\n{:<25} {:>15} {:>15}".format("Metric", "VWAP", "TWAP"))
    print("-" * 55)
    print("{:<25} {:>15} {:>15}".format(
        "Avg Execution Price",
        f"${vwap_report.average_execution_price:.4f}",
        f"${twap_report.average_execution_price:.4f}"
    ))
    print("{:<25} {:>15} {:>15}".format(
        "Slippage vs VWAP",
        f"{vwap_report.slippage_vs_vwap_bps:+.2f} bps",
        f"{twap_report.slippage_vs_vwap_bps:+.2f} bps"
    ))
    print("{:<25} {:>15} {:>15}".format(
        "Total Cost",
        f"${vwap_report.total_cost:.2f}",
        f"${twap_report.total_cost:.2f}"
    ))
    print("{:<25} {:>15} {:>15}".format(
        "Child Orders",
        str(vwap_report.num_child_orders),
        str(twap_report.num_child_orders)
    ))
    
    # =========================================================================
    # Step 6: Show Detailed Report
    # =========================================================================
    print_section("Detailed VWAP Report")
    
    report_df = vwap_report.to_dataframe()
    print(report_df.to_string(index=False))
    
    # =========================================================================
    # Step 7: Show Child Orders Sample
    # =========================================================================
    print_section("Sample Child Orders (First 10)")
    
    children_df = engine.get_child_orders_df()
    if len(children_df) > 0:
        print(children_df.head(10).to_string(index=False))
    
    # =========================================================================
    # Step 8: Execution History
    # =========================================================================
    print_section("Execution History")
    
    history = engine.get_execution_history()
    print(history.to_string(index=False))
    
    print_section("Demo Complete")
    
    return vwap_report, twap_report


if __name__ == "__main__":
    vwap_report, twap_report = run_demo()
