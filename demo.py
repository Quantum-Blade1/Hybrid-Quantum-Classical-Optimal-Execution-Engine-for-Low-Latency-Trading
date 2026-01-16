"""
Quantum-Classical Trading System - Demo

This demo demonstrates the Phase 1 classical baseline:
1. Generates realistic AAPL market data for a trading day
2. Executes a 10,000 share buy order using VWAP strategy
3. Displays comprehensive execution metrics
4. Optionally visualizes execution quality
"""

import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, ".")

from src.market_data import MarketDataSimulator, MarketParams, calculate_vwap
from src.order_book import OrderBook
from src.vwap_strategy import VWAPStrategy


def print_header(text: str, width: int = 60):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(f" {text}")
    print("=" * width)


def print_section(text: str):
    """Print a section header."""
    print(f"\n--- {text} ---")


def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.2f}"


def format_bps(value: float) -> str:
    """Format value as basis points."""
    return f"{value:+.2f} bps"


def run_demo(
    order_size: int = 10_000,
    side: str = "buy",
    visualize: bool = True,
    seed: int = 42
):
    """
    Run the trading execution demo.
    
    Args:
        order_size: Number of shares to execute
        side: 'buy' or 'sell'
        visualize: Whether to display matplotlib charts
        seed: Random seed for reproducibility
    """
    print_header("Quantum-Classical Trading System - Phase 1 Demo")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Order: {side.upper()} {order_size:,} shares of AAPL")
    
    # =========================================================================
    # Step 1: Generate Market Data
    # =========================================================================
    print_section("Generating Market Data")
    
    params = MarketParams(
        symbol="AAPL",
        initial_price=175.00,
        annual_volatility=0.22,
        annual_drift=0.10
    )
    
    simulator = MarketDataSimulator(
        params=params,
        total_daily_volume=60_000_000,  # AAPL typical daily volume
        seed=seed
    )
    
    trading_date = datetime(2024, 1, 15, 9, 30)
    market_data = simulator.generate(date=trading_date, num_minutes=390)
    
    print(f"Generated {len(market_data)} minutes of intraday data")
    print(f"\nMarket Data Summary:")
    print(f"  Symbol:        {params.symbol}")
    print(f"  Price Range:   {format_currency(market_data['price'].min())} - {format_currency(market_data['price'].max())}")
    print(f"  VWAP:          {format_currency(calculate_vwap(market_data))}")
    print(f"  Total Volume:  {market_data['volume'].sum():,} shares")
    print(f"  Avg Spread:    {format_currency(market_data['spread'].mean())} ({market_data['spread'].mean() / market_data['price'].mean() * 10000:.1f} bps)")
    
    # =========================================================================
    # Step 2: Execute VWAP Order
    # =========================================================================
    print_section("Executing VWAP Order")
    
    strategy = VWAPStrategy(
        participation_rate=0.10,  # Max 10% of each minute's volume
        max_slice_pct=0.05,       # Max 5% of order in single slice
        seed=seed
    )
    
    print(f"Strategy Parameters:")
    print(f"  Participation Rate: 10%")
    print(f"  Max Slice Size:     5% of order")
    
    metrics = strategy.execute(
        total_shares=order_size,
        side=side,
        market_data=market_data
    )
    
    # =========================================================================
    # Step 3: Display Results
    # =========================================================================
    print_section("Execution Results")
    
    print(f"\nExecution Quality:")
    print(f"  Average Price:    {format_currency(metrics.average_execution_price)}")
    print(f"  Benchmark VWAP:   {format_currency(metrics.theoretical_vwap)}")
    print(f"  Slippage:         {format_bps(metrics.slippage_bps)}")
    
    print(f"\nExecution Costs:")
    print(f"  Spread Cost:      {format_currency(metrics.spread_cost)}")
    print(f"  Impact Cost:      {format_currency(metrics.impact_cost)}")
    print(f"  Total Cost:       {format_currency(metrics.total_cost)}")
    
    print(f"\nExecution Statistics:")
    print(f"  Fill Rate:        {metrics.fill_rate * 100:.1f}%")
    print(f"  Shares Filled:    {metrics.filled_shares:,} / {metrics.total_shares:,}")
    print(f"  Execution Slices: {metrics.num_slices}")
    print(f"  Execution Time:   {metrics.execution_time_minutes} minutes")
    
    # Per-share costs
    if metrics.filled_shares > 0:
        cost_per_share = metrics.total_cost / metrics.filled_shares
        print(f"\nCost per Share:     {format_currency(cost_per_share)}")
    
    # =========================================================================
    # Step 4: Visualization (Optional)
    # =========================================================================
    if visualize:
        try:
            import matplotlib.pyplot as plt
            
            print_section("Generating Visualization")
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            # Plot 1: Price and Execution
            ax1 = axes[0]
            ax1.plot(market_data["price"], label="Mid Price", color="blue", alpha=0.7)
            ax1.fill_between(
                range(len(market_data)),
                market_data["bid"],
                market_data["ask"],
                alpha=0.2,
                color="blue",
                label="Bid-Ask"
            )
            
            # Plot execution prices
            exec_summary = strategy.get_execution_summary()
            if not exec_summary.empty:
                ax1.scatter(
                    exec_summary["minute"],
                    exec_summary["exec_price"],
                    color="red",
                    s=exec_summary["filled_qty"] / 10,
                    alpha=0.6,
                    label="Executions"
                )
            
            ax1.axhline(
                metrics.theoretical_vwap,
                color="green",
                linestyle="--",
                label=f"VWAP: {metrics.theoretical_vwap:.2f}"
            )
            ax1.axhline(
                metrics.average_execution_price,
                color="red",
                linestyle="--",
                label=f"Avg Exec: {metrics.average_execution_price:.2f}"
            )
            
            ax1.set_ylabel("Price ($)")
            ax1.set_title(f"AAPL VWAP Execution - {side.upper()} {order_size:,} shares")
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Volume Profile
            ax2 = axes[1]
            ax2.bar(range(len(market_data)), market_data["volume"], alpha=0.5, label="Market Volume")
            
            if not exec_summary.empty:
                exec_vol = np.zeros(len(market_data))
                for _, row in exec_summary.iterrows():
                    exec_vol[int(row["minute"])] = row["filled_qty"]
                ax2.bar(range(len(market_data)), exec_vol * 100, alpha=0.8, color="red", label="Execution Volume (x100)")
            
            ax2.set_ylabel("Volume")
            ax2.set_title("Volume Profile")
            ax2.legend(loc="upper left")
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Cumulative Execution
            ax3 = axes[2]
            if not exec_summary.empty:
                cumulative_shares = exec_summary["filled_qty"].cumsum()
                ax3.plot(exec_summary["minute"], cumulative_shares, color="red", linewidth=2)
                ax3.axhline(order_size, color="gray", linestyle="--", label=f"Target: {order_size:,}")
            
            ax3.set_xlabel("Minute")
            ax3.set_ylabel("Cumulative Shares")
            ax3.set_title("Execution Progress")
            ax3.legend(loc="upper left")
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            output_path = "execution_demo.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Visualization saved to: {output_path}")
            
            plt.show()
            
        except ImportError:
            print("matplotlib not available - skipping visualization")
    
    print_section("Demo Complete")
    
    return metrics


if __name__ == "__main__":
    # Run demo with default parameters
    metrics = run_demo(
        order_size=10_000,
        side="buy",
        visualize=True,
        seed=42
    )
