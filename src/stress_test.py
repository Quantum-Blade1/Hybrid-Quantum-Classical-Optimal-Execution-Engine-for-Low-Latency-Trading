"""
Stress Testing Suite

Scenarios:
1. Flash Crash: 50% price drop in 5 minutes
2. Liquidity Crisis: Spreads widen 10x
3. Volatility Spike: Sigma doubles instantaneously
4. Market Outage: Venue goes offline (zero liquidity)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from time import time
import logging

from src.market_data import MarketDataSimulator
# hybrid_demo is at root, assuming running from root
import sys
import os
sys.path.append(os.getcwd())
from hybrid_demo import run_hybrid_execution, run_vwap_execution, ExecutionResult

@dataclass
class StressResult:
    scenario_name: str
    is_hybrid: bool
    total_cost: float
    slippage_bps: float
    filled_shares: int
    fill_rate: float
    max_drawdown: float
    recovery_time_min: float
    crashed: bool = False
    error_msg: str = ""

class StressGenerator:
    """Generates market data for stress scenarios."""
    
    @staticmethod
    def flash_crash(
        num_minutes: int = 60,
        crash_start: int = 20,
        crash_duration: int = 5,
        drop_pct: float = 0.50
    ) -> pd.DataFrame:
        """Simulate flash crash."""
        from src.market_data import MarketParams
        
        sim = MarketDataSimulator(params=MarketParams(initial_price=100.0))
        data = sim.generate(num_minutes=num_minutes)
        data['volatility'] = 0.0002 # Initialize base volatility
        
        # Apply crash
        peak_price = data.iloc[crash_start]['price']
        target_price = peak_price * (1 - drop_pct)
        
        # Crash down
        crash_step = (peak_price - target_price) / crash_duration
        for i in range(crash_duration):
            idx = crash_start + i
            if idx < len(data):
                data.at[idx, 'price'] = peak_price - crash_step * (i + 1)
                data.at[idx, 'volatility'] = 0.05  # Huge vol during crash
                data.at[idx, 'spread'] = 0.50      # Wide spread
        
        # Slow recovery
        recovery_start = crash_start + crash_duration
        for i in range(recovery_start, len(data)):
             # Price slowly drifts back up 20%
             data.at[i, 'price'] = target_price * (1 + 0.005 * (i - recovery_start))
             data.at[i, 'volatility'] = 0.01  # Elevated but dropping
             
        data['is_stress'] = False
        data.loc[crash_start:crash_start+crash_duration+10, 'is_stress'] = True
        
        return data

    @staticmethod
    def liquidity_crisis(num_minutes: int = 60) -> pd.DataFrame:
        """Simulate liquidity drying up (spreads widen)."""
        data = MarketDataSimulator().generate(num_minutes=num_minutes)
        data['volatility'] = 0.0002
        
        # 10x spread widening in middle 20 mins
        start, end = 20, 40
        data['spread'] = 0.02 # Base spread
        
        data.loc[start:end, 'spread'] = 0.20 # 10x
        data.loc[start:end, 'volume'] = data.loc[start:end, 'volume'] * 0.1 # Volume dries up
        
        data['is_stress'] = False
        data.loc[start:end, 'is_stress'] = True
        return data

    @staticmethod
    def volatility_spike(num_minutes: int = 60) -> pd.DataFrame:
        """Simulate massive volatility spike."""
        data = MarketDataSimulator().generate(num_minutes=num_minutes)
        data['volatility'] = 0.0002
        
        # Volatility explodes
        start, end = 15, 45
        for i in range(start, end):
            # Add random large jumps
            jump = np.random.normal(0, 5.0) # $5 jumps
            data.at[i, 'price'] += jump
            data.at[i, 'volatility'] = 0.02 # 100x normal
            
        data['is_stress'] = False
        data.loc[start:end, 'is_stress'] = True
        return data

    @staticmethod
    def market_outage(num_minutes: int = 60) -> pd.DataFrame:
        """Simulate market outage (zero liquidity)."""
        data = MarketDataSimulator().generate(num_minutes=num_minutes)
        data['volatility'] = 0.0002
        
        # Market halts
        start, end = 25, 35
        data.loc[start:end, 'volume'] = 0
        data.loc[start:end, 'spread'] = 1000.0 # Effectively no execution
        data.loc[start:end, 'price'] = data.loc[start-1, 'price'] # Price frozen
        
        data['is_stress'] = False
        data.loc[start:end, 'is_stress'] = True
        return data

class StressRunner:
    """Runs stress tests."""
    
    def run_scenario(self, name: str, data: pd.DataFrame) -> List[StressResult]:
        print(f"\nRunning Scenario: {name}")
        results = []
        
        # 1. Classical (VWAP) - reusing hybrid_demo logic but forcing SA/VWAP
        try:
            # We use hybrid_demo.run_vwap_execution as baseline
            
            start = time()
            res = run_vwap_execution(data, total_shares=50000)
            elapsed = time() - start
            
            results.append(StressResult(
                scenario_name=name,
                is_hybrid=False,
                total_cost=res.total_cost,
                slippage_bps=res.slippage_bps,
                filled_shares=res.executed_shares,
                fill_rate=res.executed_shares / 50000.0,
                max_drawdown=0.0, # TODO: calculate
                recovery_time_min=0.0
            ))
            print(f"  Classical: Filled {res.executed_shares} ({res.slippage_bps:.1f} bps)")
            
        except Exception as e:
            print(f"  Classical crashed: {e}")
            results.append(StressResult(
                scenario_name=name, is_hybrid=False, total_cost=0, slippage_bps=0,
                filled_shares=0, fill_rate=0, max_drawdown=0, recovery_time_min=0,
                crashed=True, error_msg=str(e)
            ))

        # 2. Hybrid
        try:
            start = time()
            res = run_hybrid_execution(data, total_shares=50000, lambda_tradeoff=0.5)
            elapsed = time() - start
            
            results.append(StressResult(
                scenario_name=name,
                is_hybrid=True,
                total_cost=res.total_cost,
                slippage_bps=res.slippage_bps,
                filled_shares=res.executed_shares,
                fill_rate=res.executed_shares / 50000.0,
                max_drawdown=0.0,
                recovery_time_min=0.0
            ))
            print(f"  Hybrid:    Filled {res.executed_shares} ({res.slippage_bps:.1f} bps)")
            
        except Exception as e:
            print(f"  Hybrid crashed: {e}")
            results.append(StressResult(
                scenario_name=name, is_hybrid=True, total_cost=0, slippage_bps=0,
                filled_shares=0, fill_rate=0, max_drawdown=0, recovery_time_min=0,
                crashed=True, error_msg=str(e)
            ))
            
        return results

    def run_suite(self):
        print("="*60)
        print("STRESS TEST SUITE")
        print("="*60)
        
        scenarios = [
            ("Flash Crash", StressGenerator.flash_crash()),
            ("Liquidity Crisis", StressGenerator.liquidity_crisis()),
            ("Volatility Spike", StressGenerator.volatility_spike()),
            ("Market Outage", StressGenerator.market_outage())
        ]
        
        all_results = []
        for name, data in scenarios:
            all_results.extend(self.run_scenario(name, data))
            
        self.report(all_results)
        
    def report(self, results: List[StressResult]):
        print("\n" + "="*60)
        print("STRESS TEST REPORT")
        print("="*60)
        
        # Convert to DataFrame for nice printing
        data = []
        for r in results:
            data.append({
                "Scenario": r.scenario_name,
                "Mode": "Hybrid" if r.is_hybrid else "Classical",
                "Crashed": "YES" if r.crashed else "No",
                "Fill %": f"{r.fill_rate*100:.1f}%",
                "Slippage": f"{r.slippage_bps:.1f}",
                "Cost": f"${r.total_cost:,.0f}"
            })
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        
        # Check for failures
        crashes = df[df["Crashed"] == "YES"]
        if not crashes.empty:
            print("\nCRITICAL FAILURES DETECTED:")
            print(crashes)
        else:
            print("\nAll systems operational. No crashes detected.")

if __name__ == "__main__":
    runner = StressRunner()
    runner.run_suite()
