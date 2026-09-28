"""
Walk-Forward Analysis

Implements a rolling window backtest for optimizing trading strategies.
Process:
1. Divide data into Windows (Train Window -> Test Window)
2. Train: Extract volume profile and volatility from Train Window
3. Test: Execute strategies on Test Window using trained parameters
4. Compare:
    - Baseline VWAP (uses Perfect Foresight of Day's volume - Theoretical Upper Bound?) 
      *Correction*: Standard VWAP usually assumes Day's volume profile matches historical average.
      Let's define:
      - "Static VWAP": Uses a generic U-shape profile (uninformed)
      - "Adaptive VWAP": Uses profile learned from Train Window (informed)
      - "Hybrid": Tuned via Train Window volatility (e.g. Lambda)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Optional

from qexec.market.simulator import MarketDataSimulator, MarketParams
from qexec.execution.engine import ExecutionEngine
from qexec.execution.strategies.vwap import VWAPStrategy
from qexec.analysis.shortfall import ISAnalyzer
from qexec.execution.strategies.fixed import FixedScheduleStrategy
from qexec.optimization.qubo import ExecutionQUBO
from qexec.optimization.schedule import optimize_schedule, slice_level_config
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver

@dataclass
class WindowResult:
    window_id: int
    train_start: str
    test_start: str
    
    # Metrics (Total Shortfall $)
    shortfall_static: float
    shortfall_adaptive: float
    shortfall_hybrid: float
    
    # Other metrics for hybrid tuning
    train_volatility: float

class WalkForwardAnalyzer:
    """Rolling window analyzer for trading strategies."""
    
    def __init__(self, 
                 total_days: int = 10, 
                 train_days: int = 3, 
                 test_days: int = 1,
                 daily_shares: int = 50000):
        self.total_days = total_days
        self.train_days = train_days
        self.test_days = test_days
        self.daily_shares = daily_shares
        
        # Sim parameters
        self.market_params = MarketParams(
            initial_price=100.0, 
            annual_volatility=0.40 # High vol to make adaptation matter
        )
        
    def generate_full_dataset(self) -> List[pd.DataFrame]:
        """Generate simulation data for all days."""
        sim = MarketDataSimulator(self.market_params)
        days_data = []
        
        # Simulate contiguous days (prices chaining)
        current_price = self.market_params.initial_price
        
        for i in range(self.total_days):
            # Update start price
            self.market_params.initial_price = current_price
            sim.params = self.market_params
            
            # Generate 1 day (e.g. 390 minutes)
            day_data = sim.generate(num_minutes=390)
            
            # Add Day ID
            day_data['day'] = i
            days_data.append(day_data)
            
            # Update price for next day (close + overnight drift/jump?)
            # Just take last close
            current_price = day_data.iloc[-1]['price']
            
        return days_data

    def run(self):
        print("Generating market data...")
        all_days = self.generate_full_dataset()
        
        results = []
        
        # Rolling Window Loop
        # Step size = test_days
        num_windows = (self.total_days - self.train_days) // self.test_days
        
        print(f"Starting Walk-Forward Analysis ({num_windows} windows)...")
        print("-" * 60)
        print(f"{'Window':<5} {'Vol':<8} {'Static':<12} {'Adaptive':<12} {'Hybrid':<12} {'Winner'}")
        print("-" * 60)
        
        for w in range(num_windows):
            start_idx = w * self.test_days
            train_end_idx = start_idx + self.train_days
            test_end_idx = train_end_idx + self.test_days
            
            if test_end_idx > len(all_days):
                break
                
            train_data = all_days[start_idx:train_end_idx]
            test_data = all_days[train_end_idx:test_end_idx]
            
            res = self._process_window(w, train_data, test_data)
            results.append(res)
            
            # Identify winner
            scores = {
                'Static': res.shortfall_static, 
                'Adaptive': res.shortfall_adaptive, 
                'Hybrid': res.shortfall_hybrid
            }
            winner = min(scores, key=scores.get)
            
            print(f"{w:<5} {res.train_volatility:<8.1%} "
                  f"${res.shortfall_static:<11,.0f} "
                  f"${res.shortfall_adaptive:<11,.0f} "
                  f"${res.shortfall_hybrid:<11,.0f} "
                  f"{winner}")
            
        return results
        
    def _process_window(self, window_id: int, 
                        train_data: List[pd.DataFrame], 
                        test_data: List[pd.DataFrame]) -> WindowResult:
        
        # 1. TRAIN: Extract parameters from historical days
        # Merge train days
        train_df = pd.concat(train_data)
        
        # A) Volume Profile (average by minute of day)
        # Assuming all days have same # minutes and 0-index corresponds to time
        # Group by index (0-389)
        # Reset index to get 0-389 for each day
        minutes = []
        for d in train_data:
            minutes.append(d['volume'].values)
        
        avg_volume_profile = np.mean(minutes, axis=0) # Average across days
        
        # B) Volatility
        returns = train_df['price'].pct_change().dropna()
        train_vol = returns.std() * np.sqrt(390 * 252) # Annualized approx
        
        # 2. TEST: Run execution on test days using trained params
        # Accumulate shortfall across test days
        
        total_static = 0.0
        total_adaptive = 0.0
        total_hybrid = 0.0
        
        for day in test_data:
            # Setup
            arrival_price = day.iloc[0]['price']
            analyzer = ISAnalyzer(arrival_price, self.daily_shares)
            
            # --- Strategy A: Static VWAP (Uninformed) ---
            # Uses a generic U-Shape or Flat profile. 
            # Let's use Flat (TWAP-ish) or a "bad" profile prediction to simulate uninformed
            # Actually, standard VWAP usually just assumes typical U-shape.
            # Let's generate a theoretical U-shape
            
            # ... Or simpler: Use the actual day's profile but add noise to simulate forecast error?
            # "Static" usually refers to a fixed profile derived long ago. 
            # Let's use a flat profile (TWAP) as the 'Uninformed Base' or a U-shape that doesn't match specific recent trends.
            # Let's use TWAP as baseline "Static".
            # Or use VWAP with a generic U-shape (not learned from recent days).
            
            # Let's use the first day of training as the "Static" profile (outdated info)
            static_profile = train_data[0]['volume'].values
            
            eng_s = ExecutionEngine()
            strat_s = VWAPStrategy(historical_profile=static_profile, order_book=eng_s.order_book)
            report_s = eng_s.process_order(
                self._create_order(), day, strat_s
            )
            # Use IS Analyzer? ExecutionReport has IS components built-in now?
            # Actually ExecutionReport in execution_engine.py calculates Slippage vs Arrival.
            # IS = (AvgExec - Arrival) * Shares (if side=Buy)
            # ExecutionReport.total_cost is Spread + Impact. 
            # But we want Implementation Shortfall (Slippage vs Arrival).
            # Report has 'slippage_vs_arrival_bps'.
            
            # IS ($) = slippage_bps / 10000 * Arrival * Shares
            is_s = report_s.slippage_vs_arrival_bps / 10000 * arrival_price * self.daily_shares
            total_static += is_s
            
            # --- Strategy B: Adaptive VWAP (Informed) ---
            # Uses the average profile from the Train Window (Recent history)
            eng_a = ExecutionEngine()
            strat_a = VWAPStrategy(historical_profile=avg_volume_profile, order_book=eng_a.order_book)
            report_a = eng_a.process_order(
                self._create_order(), day, strat_a
            )
            is_a = report_a.slippage_vs_arrival_bps / 10000 * arrival_price * self.daily_shares
            total_adaptive += is_a
            
            # --- Strategy C: Hybrid (QUBO-Optimized) ---
            # Run actual QUBO optimization via SA to generate schedule,
            # then execute it through the same engine.
            num_slices = min(20, len(day))
            qubo_config = slice_level_config(
                self.daily_shares,
                num_slices,
                volatility=max(0.001, train_vol / np.sqrt(252)),
                impact_coefficient=0.1,
            )
            slice_qty, _ = optimize_schedule(
                ExecutionQUBO(qubo_config), SimulatedAnnealingSolver(num_sweeps=300, seed=42)
            )

            # Each slice's quantity is placed at the slice's first minute.
            hybrid_schedule = np.zeros(len(day))
            minutes_per_slice = max(1, len(day) // num_slices)
            for t, quantity in enumerate(slice_qty):
                if quantity > 0:
                    hybrid_schedule[min(t * minutes_per_slice, len(day) - 1)] += quantity

            # Normalize to match daily shares
            sched_sum = hybrid_schedule.sum()
            if sched_sum > 0:
                hybrid_schedule = hybrid_schedule * (self.daily_shares / sched_sum)
            else:
                hybrid_schedule = np.full(len(day), self.daily_shares / len(day))

            eng_h = ExecutionEngine()
            strat_h = FixedScheduleStrategy(hybrid_schedule)
            report_h = eng_h.process_order(
                self._create_order(), day, strat_h
            )
            is_h = report_h.slippage_vs_arrival_bps / 10000 * arrival_price * self.daily_shares
            total_hybrid += is_h
            
        return WindowResult(
            window_id=window_id,
            train_start="",
            test_start="",
            shortfall_static=total_static,
            shortfall_adaptive=total_adaptive,
            shortfall_hybrid=total_hybrid,
            train_volatility=train_vol
        )

    def _create_order(self):
        from qexec.execution.engine import ParentOrder, OrderSide
        return ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=self.daily_shares,
            time_horizon_minutes=390 # Full day
        )

    def plot_results(self, results: List[WindowResult]):
        """Plot cumulative shortfall."""
        df = pd.DataFrame([vars(r) for r in results])
        
        df['Cum_Static'] = df['shortfall_static'].cumsum()
        df['Cum_Adaptive'] = df['shortfall_adaptive'].cumsum()
        df['Cum_Hybrid'] = df['shortfall_hybrid'].cumsum()
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['window_id'], df['Cum_Static'], label='Static VWAP', linestyle='--')
        plt.plot(df['window_id'], df['Cum_Adaptive'], label='Adaptive VWAP (Recalibrated)')
        plt.plot(df['window_id'], df['Cum_Hybrid'], label='Hybrid Quantum', linewidth=2)
        
        plt.title('Walk-Forward Analysis: Cumulative Implementation Shortfall')
        plt.xlabel('Rolling Window')
        plt.ylabel('Cumulative Cost ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('walk_forward_results.png')
        print("Saved plot: walk_forward_results.png")
