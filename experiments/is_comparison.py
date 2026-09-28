"""
IS Comparison Script

Runs VWAP and TWAP strategies.
Calculates Implementation Shortfall for each.
Generates stacked bar chart comparison.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from qexec.market.simulator import MarketDataSimulator, MarketParams
# removed StrategyConfig import
from qexec.execution.strategies.vwap import VWAPStrategy
from qexec.execution.strategies.twap import TWAPStrategy
from qexec.runtime.controller import HybridController
from qexec.execution.engine import ExecutionEngine
from qexec.analysis.shortfall import ISAnalyzer, plot_is_breakdown

def main():
    print("\n" + "="*70)
    print(" Implementation Shortfall Comparison")
    print("="*70)
    
    # 1. Generate Market Data
    # 100k shares, 1 hour
    total_shares = 100000
    n_minutes = 60
    
    params = MarketParams(initial_price=100.0, annual_volatility=0.30) # High vol for impact
    sim = MarketDataSimulator(params=params)
    data = sim.generate(num_minutes=n_minutes)
    
    decision_price = data.iloc[0]['price'] * 0.999 # Say we decided slightly before arrival when price was lower
    # Or just use arrival price as decision price (Delay cost = 0)
    decision_price = data.iloc[0]['price']
    
    print(f"Decision Price: ${decision_price:.2f}")
    
    analyzer = ISAnalyzer(decision_price, total_shares)
    results = {}
    
    # ---------------------------------------------------------
    # 2. Run VWAP
    # ---------------------------------------------------------
    print("\nRunning VWAP...")
    # Config handled via init args now
    engine_vwap = ExecutionEngine()
    strategy_vwap = VWAPStrategy(participation_rate=0.1, order_book=engine_vwap.order_book)
    strategy_vwap.execute(total_shares, 'buy', data)
    
    # Extract execution log from strategy slices
    log_data = []
    for s in strategy_vwap.slices:
        log_data.append({
            'timestamp': s.timestamp,
            'shares': s.filled_quantity,
            'price': s.execution_price
        })
    log_vwap = pd.DataFrame(log_data)
    
    results['VWAP'] = analyzer.analyze(log_vwap, data)
    print(f"  VWAP IS: ${results['VWAP'].total_shortfall:,.0f}")

    # ---------------------------------------------------------
    # 3. Run TWAP
    # ---------------------------------------------------------
    print("\nRunning TWAP...")
    engine_twap = ExecutionEngine()
    strategy_twap = TWAPStrategy(interval_minutes=1, order_book=engine_twap.order_book)
    strategy_twap.execute(total_shares, 'buy', data)
    
    log_data = []
    for s in strategy_twap.slices:
        log_data.append({
            'timestamp': s.timestamp,
            'shares': s.filled_quantity,
            'price': s.execution_price
        })
    log_twap = pd.DataFrame(log_data)
    
    results['TWAP'] = analyzer.analyze(log_twap, data)
    print(f"  TWAP IS: ${results['TWAP'].total_shortfall:,.0f}")
    
    # ---------------------------------------------------------
    # 4. Visualization
    # ---------------------------------------------------------
    print("\nGenerating Chart...")
    plot_is_breakdown(results, "is_comparison.png")
    
    # Print breakdown table
    print("\nBreakdown:")
    rows = []
    for name, res in results.items():
        r = res.to_dict()
        r['Strategy'] = name
        rows.append(r)
    df_res = pd.DataFrame(rows).set_index('Strategy')
    print(df_res.to_string(float_format=lambda x: f"{x:,.0f}"))

if __name__ == "__main__":
    main()
