"""
Strategy Comparison Demo

Compares VWAP and TWAP execution strategies on the same market data,
showing differences in execution schedule, performance, and costs.
"""

import sys
from datetime import datetime

import numpy as np

import os
# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.market_data import MarketDataSimulator, MarketParams
from src.vwap_strategy import VWAPStrategy
from src.twap_strategy import TWAPStrategy, compare_strategies


def print_header(text: str, width: int = 70):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(f" {text}")
    print("=" * width)


def print_section(text: str):
    """Print a section header."""
    print(f"\n--- {text} ---")


def run_comparison(
    order_size: int = 10_000,
    side: str = "buy",
    visualize: bool = True,
    seed: int = 42
):
    """
    Run comparison between VWAP and TWAP strategies.
    
    Args:
        order_size: Number of shares to execute
        side: 'buy' or 'sell'
        visualize: Whether to display matplotlib charts
        seed: Random seed for reproducibility
    """
    print_header("VWAP vs TWAP Strategy Comparison")
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
        total_daily_volume=60_000_000,
        seed=seed
    )
    
    trading_date = datetime(2024, 1, 15, 9, 30)
    market_data = simulator.generate(date=trading_date, num_minutes=390)
    
    print(f"Generated {len(market_data)} minutes of intraday data")
    
    # =========================================================================
    # Step 2: Initialize Strategies
    # =========================================================================
    print_section("Initializing Strategies")
    
    vwap = VWAPStrategy(
        participation_rate=0.10,
        max_slice_pct=0.05,
        seed=seed
    )
    
    twap = TWAPStrategy(
        interval_minutes=1,
        max_slice_pct=0.05,
        seed=seed
    )
    
    print("VWAP: Volume-proportional slices, 10% participation rate")
    print("TWAP: Equal slices at 1-minute intervals")
    
    # =========================================================================
    # Step 3: Execute Both Strategies
    # =========================================================================
    print_section("Executing Strategies")
    
    vwap_metrics = vwap.execute(order_size, side, market_data)
    twap_metrics = twap.execute(order_size, side, market_data)
    
    # =========================================================================
    # Step 4: Compare Results
    # =========================================================================
    print_section("Performance Comparison")
    
    comparison_df = compare_strategies(
        vwap_strategy=VWAPStrategy(participation_rate=0.10, max_slice_pct=0.05, seed=seed),
        twap_strategy=TWAPStrategy(interval_minutes=1, max_slice_pct=0.05, seed=seed),
        total_shares=order_size,
        side=side,
        market_data=market_data
    )
    
    print("\n" + comparison_df.to_string(index=False))
    
    # =========================================================================
    # Step 5: Analysis
    # =========================================================================
    print_section("Analysis")
    
    vwap_vwap_benchmark = (market_data["price"] * market_data["volume"]).sum() / market_data["volume"].sum()
    
    vwap_vs_vwap = (vwap_metrics.average_execution_price - vwap_vwap_benchmark) / vwap_vwap_benchmark * 10000
    twap_vs_vwap = (twap_metrics.average_execution_price - vwap_vwap_benchmark) / vwap_vwap_benchmark * 10000
    
    print(f"\nSlippage vs Market VWAP benchmark:")
    print(f"  VWAP Strategy: {vwap_vs_vwap:+.2f} bps")
    print(f"  TWAP Strategy: {twap_vs_vwap:+.2f} bps")
    
    winner = "VWAP" if abs(vwap_vs_vwap) < abs(twap_vs_vwap) else "TWAP"
    savings = abs(abs(twap_vs_vwap) - abs(vwap_vs_vwap))
    
    print(f"\n{winner} outperformed by {savings:.2f} bps")
    print(f"Cost difference: ${abs(vwap_metrics.total_cost - twap_metrics.total_cost):.2f}")
    
    # =========================================================================
    # Step 6: Visualization
    # =========================================================================
    if visualize:
        try:
            import matplotlib.pyplot as plt
            
            print_section("Generating Visualization")
            
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
            
            # Get execution summaries
            vwap_summary = vwap.get_execution_summary()
            twap_summary = twap.get_execution_summary()
            
            # Plot 1: Price with executions
            ax1 = axes[0]
            ax1.plot(market_data["price"], label="Mid Price", color="blue", alpha=0.7, linewidth=1)
            
            # Mark VWAP executions
            if not vwap_summary.empty:
                ax1.scatter(
                    vwap_summary["minute"],
                    vwap_summary["exec_price"],
                    color="green",
                    s=vwap_summary["filled_qty"] / 5,
                    alpha=0.6,
                    label="VWAP Executions",
                    marker="o"
                )
            
            # Mark TWAP executions  
            if not twap_summary.empty:
                ax1.scatter(
                    twap_summary["minute"],
                    twap_summary["exec_price"],
                    color="red",
                    s=twap_summary["filled_qty"] / 5,
                    alpha=0.6,
                    label="TWAP Executions",
                    marker="s"
                )
            
            ax1.axhline(vwap_vwap_benchmark, color="green", linestyle="--", alpha=0.7, label=f"VWAP: ${vwap_vwap_benchmark:.2f}")
            ax1.axhline(twap_metrics.benchmark_price, color="red", linestyle="--", alpha=0.7, label=f"TWAP: ${twap_metrics.benchmark_price:.2f}")
            
            ax1.set_ylabel("Price ($)")
            ax1.set_title(f"VWAP vs TWAP Execution - {side.upper()} {order_size:,} shares")
            ax1.legend(loc="upper left", fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Execution Schedule Comparison
            ax2 = axes[1]
            
            vwap_schedule = vwap.calculate_schedule(order_size, market_data)
            twap_schedule = twap.calculate_schedule(order_size, market_data)
            
            width = 0.4
            x = np.arange(len(market_data))
            
            # Downsample for visibility (every 10 minutes)
            sample_rate = 10
            x_sampled = x[::sample_rate]
            vwap_sampled = [vwap_schedule[i:i+sample_rate].sum() for i in range(0, len(vwap_schedule), sample_rate)]
            twap_sampled = [twap_schedule[i:i+sample_rate].sum() for i in range(0, len(twap_schedule), sample_rate)]
            
            ax2.bar(x_sampled - width/2, vwap_sampled, width, label="VWAP Schedule", color="green", alpha=0.7)
            ax2.bar(x_sampled + width/2, twap_sampled, width, label="TWAP Schedule", color="red", alpha=0.7)
            
            ax2.set_ylabel("Shares per 10min")
            ax2.set_title("Execution Schedule Comparison (10-min buckets)")
            ax2.legend(loc="upper right")
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Cumulative Execution
            ax3 = axes[2]
            
            if not vwap_summary.empty:
                vwap_cumulative = np.zeros(len(market_data))
                for _, row in vwap_summary.iterrows():
                    vwap_cumulative[int(row["minute"]):] += row["filled_qty"]
                ax3.plot(vwap_cumulative, color="green", linewidth=2, label="VWAP")
            
            if not twap_summary.empty:
                twap_cumulative = np.zeros(len(market_data))
                for _, row in twap_summary.iterrows():
                    twap_cumulative[int(row["minute"]):] += row["filled_qty"]
                ax3.plot(twap_cumulative, color="red", linewidth=2, label="TWAP")
            
            ax3.axhline(order_size, color="gray", linestyle="--", alpha=0.5, label=f"Target: {order_size:,}")
            
            ax3.set_xlabel("Minute")
            ax3.set_ylabel("Cumulative Shares")
            ax3.set_title("Cumulative Execution Progress")
            ax3.legend(loc="lower right")
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            output_path = "strategy_comparison.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Visualization saved to: {output_path}")
            
            plt.show()
            
        except ImportError:
            print("matplotlib not available - skipping visualization")
    
    print_section("Comparison Complete")
    
    return vwap_metrics, twap_metrics


if __name__ == "__main__":
    vwap_results, twap_results = run_comparison(
        order_size=10_000,
        side="buy",
        visualize=True,
        seed=42
    )
