"""
End-to-End Hybrid Quantum-Classical Trading Demo

Scenario: Execute 50,000 shares of AAPL over 1 hour
- Realistic market data with volatility spikes
- Compare three execution modes:
  1. Pure Classical (VWAP)
  2. Hybrid Quantum (with decision layer)
  3. Classical SA
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from time import time
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)


# Local imports
from qexec.market.simulator import MarketDataSimulator, MarketParams
from qexec.execution.engine import ExecutionEngine, ParentOrder, OrderSide
from qexec.execution.strategies.vwap import VWAPStrategy
from qexec.analysis.runners import (
    ExecutionResult, run_vwap_execution, run_sa_execution, run_hybrid_execution
)
from qexec.runtime.decision import (
    OptimizationDecisionEngine, 
    DecisionConfig, 
    MarketState,
    ImprovementTracker
)


# =============================================================================
# Market Data with Volatility Spikes
# =============================================================================

def generate_market_with_volatility_spikes(
    num_minutes: int = 60,
    base_price: float = 175.0,
    base_volatility: float = 0.0002,
    spike_times: List[int] = [15, 35],
    spike_magnitude: float = 5.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate market data with volatility spikes.
    
    Args:
        num_minutes: Total minutes
        base_price: Starting price
        base_volatility: Normal volatility
        spike_times: Minutes when volatility spikes occur
        spike_magnitude: Multiplier for spike volatility
        seed: Random seed
        
    Returns:
        DataFrame with price, volume, spread columns
    """
    np.random.seed(seed)
    
    prices = [base_price]
    volumes = []
    spreads = []
    volatilities = []
    
    for minute in range(num_minutes):
        # Check if volatility spike
        is_spike = any(abs(minute - t) <= 3 for t in spike_times)
        vol = base_volatility * spike_magnitude if is_spike else base_volatility
        volatilities.append(vol)
        
        # Generate price change
        returns = np.random.normal(0, vol)
        new_price = prices[-1] * (1 + returns)
        prices.append(new_price)
        
        # U-shaped volume profile with spike adjustment
        t = minute / num_minutes
        volume_factor = 1.5 - 0.8 * np.sin(np.pi * t)
        if is_spike:
            volume_factor *= 1.5  # More volume during spikes
        base_volume = 1_000_000
        volume = int(base_volume * volume_factor * (0.8 + 0.4 * np.random.random()))
        volumes.append(volume)
        
        # Spread widens during volatility
        base_spread = 0.02
        spread = base_spread * (1 + vol / base_volatility)
        spreads.append(spread)
    
    # Create timestamp index
    base_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    timestamps = [base_time + timedelta(minutes=m) for m in range(num_minutes)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'minute': range(num_minutes),
        'price': prices[1:],
        'volume': volumes,
        'spread': spreads,
        'volatility': volatilities,
        'is_spike': [any(abs(m - t) <= 3 for t in spike_times) for m in range(num_minutes)]
    })


# =============================================================================
# Execution Mode: Pure VWAP
# =============================================================================

# =============================================================================
# Visualization
# =============================================================================

def create_visualization(
    market_data: pd.DataFrame,
    results: Dict[str, ExecutionResult],
    save_path: str = "hybrid_demo_dashboard.png"
) -> None:
    """Create comprehensive visualization dashboard."""
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available")
        return
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # =================================
    # Plot 1: Price with volatility spikes (top row, spans 2 cols)
    # =================================
    ax1 = fig.add_subplot(gs[0, :2])
    
    ax1.plot(market_data['minute'], market_data['price'], 'b-', label='Price', linewidth=1)
    
    # Shade volatility spike periods
    for idx, row in market_data.iterrows():
        if row['is_spike']:
            ax1.axvspan(idx - 0.5, idx + 0.5, alpha=0.2, color='red')
    
    ax1.set_xlabel('Minute')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Market Price with Volatility Spikes (Red Shaded)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # =================================
    # Plot 2: Volatility over time
    # =================================
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(market_data['minute'], market_data['volatility'] * 100, 'r-')
    ax2.set_xlabel('Minute')
    ax2.set_ylabel('Volatility (%)')
    ax2.set_title('Intraday Volatility')
    ax2.grid(True, alpha=0.3)
    
    # =================================
    # Plot 3: Execution comparison (cumulative)
    # =================================
    ax3 = fig.add_subplot(gs[1, :2])
    
    colors = {'VWAP': 'green', 'SA-Optimized': 'orange', 'Hybrid-Quantum': 'blue'}
    
    for mode, result in results.items():
        if result.execution_log:
            df = pd.DataFrame(result.execution_log)
            df['cumulative'] = df['shares'].cumsum()
            ax3.plot(df['minute'], df['cumulative'], 
                    label=mode, color=colors.get(mode, 'gray'), linewidth=2)
    
    ax3.set_xlabel('Minute')
    ax3.set_ylabel('Cumulative Shares')
    ax3.set_title('Execution Progress by Mode')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # =================================
    # Plot 4: Cost comparison bar chart
    # =================================
    ax4 = fig.add_subplot(gs[1, 2])
    
    modes = list(results.keys())
    costs = [results[m].total_cost for m in modes]
    bars = ax4.bar(modes, costs, color=[colors.get(m, 'gray') for m in modes], alpha=0.7)
    ax4.set_ylabel('Total Cost ($)')
    ax4.set_title('Execution Cost Comparison')
    ax4.tick_params(axis='x', rotation=15)
    
    for bar, cost in zip(bars, costs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'${cost:.0f}', ha='center', fontsize=9)
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    # =================================
    # Plot 5: Slippage comparison
    # =================================
    ax5 = fig.add_subplot(gs[2, 0])
    
    slippages = [results[m].slippage_bps for m in modes]
    bars = ax5.bar(modes, slippages, color=[colors.get(m, 'gray') for m in modes], alpha=0.7)
    ax5.set_ylabel('Slippage (bps)')
    ax5.set_title('Slippage vs VWAP Benchmark')
    ax5.axhline(0, color='black', linewidth=0.5)
    ax5.tick_params(axis='x', rotation=15)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # =================================
    # Plot 6: Optimization invocations
    # =================================
    ax6 = fig.add_subplot(gs[2, 1])
    
    invocations = [results[m].optimization_invocations for m in modes]
    opt_times = [results[m].optimization_time * 1000 for m in modes]  # ms
    
    x = np.arange(len(modes))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, invocations, width, label='Invocations', color='steelblue')
    ax6.set_ylabel('Count')
    ax6.set_xticks(x)
    ax6.set_xticklabels(modes, rotation=15)
    ax6.set_title('Optimizer Usage')
    ax6.legend(loc='upper left')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # =================================
    # Plot 7: Summary metrics table
    # =================================
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    table_data = []
    for mode in modes:
        r = results[mode]
        table_data.append([
            mode,
            f'{r.executed_shares:,}',
            f'${r.avg_price:.2f}',
            f'{r.slippage_bps:+.1f}',
            f'{r.optimization_invocations}'
        ])
    
    table = ax7.table(
        cellText=table_data,
        colLabels=['Mode', 'Filled', 'Avg Price', 'Slip (bps)', 'Opt Calls'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax7.set_title('Performance Summary', pad=20)
    
    plt.suptitle('Hybrid Quantum-Classical Trading Execution Demo\n50,000 Shares over 60 Minutes',
                 fontsize=14, fontweight='bold')
    
    try:
        # Try high quality first
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Dashboard saved to: {save_path}")
    except MemoryError:
        print("Warning: MemoryError during high-res plot save. Retrying with low-res...")
        try:
            plt.savefig(save_path, dpi=72, bbox_inches='tight')
            print(f"Dashboard saved (Low Res) to: {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    except Exception as e:
        print(f"Error generating visualization: {e}")
    finally:
        plt.close(fig) # Important: Release memory
    
    # plt.show() # skipped for headless safety


# =============================================================================
# Main Demo
# =============================================================================

def run_end_to_end_demo():
    """Run complete end-to-end hybrid system demo."""
    
    print("\n" + "="*80)
    print(" HYBRID QUANTUM-CLASSICAL TRADING EXECUTION DEMO")
    print(" Scenario: Execute 50,000 shares of AAPL over 1 hour")
    print("="*80)
    
    # Parameters
    total_shares = 50_000
    num_minutes = 60
    seed = 42
    
    # =========================================================================
    # Step 1: Generate Market Data
    # =========================================================================
    print("\n[1/5] Generating market data with volatility spikes...")
    
    market_data = generate_market_with_volatility_spikes(
        num_minutes=num_minutes,
        base_price=175.0,
        spike_times=[15, 35, 50],  # Three volatility events
        spike_magnitude=5.0,
        seed=seed
    )
    
    spike_minutes = market_data[market_data['is_spike']]['minute'].tolist()
    print(f"   - {num_minutes} minutes of data generated")
    print(f"   - Volatility spikes at minutes: {spike_minutes[:5]}...")
    print(f"   - Price range: ${market_data['price'].min():.2f} - ${market_data['price'].max():.2f}")
    
    # =========================================================================
    # Step 2: Run VWAP Execution
    # =========================================================================
    print("\n[2/5] Running pure VWAP execution...")
    
    start = time()
    vwap_result = run_vwap_execution(market_data, total_shares, seed)
    vwap_time = time() - start
    
    print(f"   - Filled: {vwap_result.executed_shares:,} shares")
    print(f"   - Avg price: ${vwap_result.avg_price:.4f}")
    print(f"   - Slippage: {vwap_result.slippage_bps:+.2f} bps")
    print(f"   - Time: {vwap_time:.2f}s")
    
    # =========================================================================
    # Step 3: Run SA-Optimized Execution
    # =========================================================================
    print("\n[3/5] Running SA-optimized execution...")
    
    start = time()
    sa_result = run_sa_execution(market_data, total_shares, num_slices=20, seed=seed)
    sa_time = time() - start
    
    print(f"   - Filled: {sa_result.executed_shares:,} shares")
    print(f"   - Avg price: ${sa_result.avg_price:.4f}")
    print(f"   - Slippage: {sa_result.slippage_bps:+.2f} bps")
    print(f"   - Optimization time: {sa_result.optimization_time*1000:.1f}ms")
    
    # =========================================================================
    # Step 4: Run Hybrid Execution
    # =========================================================================
    print("\n[4/5] Running hybrid quantum-classical execution...")
    
    start = time()
    hybrid_result = run_hybrid_execution(
        market_data, 
        total_shares, 
        num_slices=20,
        lambda_tradeoff=0.3,  # Aggressive to trigger optimizations
        seed=seed
    )
    hybrid_time = time() - start
    
    print(f"   - Filled: {hybrid_result.executed_shares:,} shares")
    print(f"   - Avg price: ${hybrid_result.avg_price:.4f}")
    print(f"   - Slippage: {hybrid_result.slippage_bps:+.2f} bps")
    print(f"   - Optimizer invocations: {hybrid_result.optimization_invocations}")
    print(f"   - Total optimization time: {hybrid_result.optimization_time*1000:.1f}ms")
    
    # =========================================================================
    # Step 5: Comparison Summary
    # =========================================================================
    print("\n[5/5] Generating comparison dashboard...")
    
    results = {
        'VWAP': vwap_result,
        'SA-Optimized': sa_result,
        'Hybrid-Quantum': hybrid_result
    }
    
    # Print summary table
    print("\n" + "="*80)
    print(" PERFORMANCE COMPARISON")
    print("="*80)
    print(f"\n {'Mode':<20} {'Filled':>12} {'Avg Price':>12} {'Slippage':>12} {'Opt Calls':>10}")
    print("-" * 70)
    
    for mode, r in results.items():
        print(f" {mode:<20} {r.executed_shares:>12,} ${r.avg_price:>10.4f} {r.slippage_bps:>+11.2f} {r.optimization_invocations:>10}")
    
    # Find best
    best_mode = min(results.keys(), key=lambda m: results[m].slippage_bps)
    print(f"\n Best performer: {best_mode}")
    
    # Create visualization
    create_visualization(market_data, results, "hybrid_demo_dashboard.png")
    
    print("\n" + "="*80)
    print(" Demo Complete!")
    print("="*80)
    
    return results, market_data


if __name__ == "__main__":
    results, market_data = run_end_to_end_demo()
