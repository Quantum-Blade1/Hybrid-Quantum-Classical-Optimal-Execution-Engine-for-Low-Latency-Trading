"""
Implementation Shortfall Analysis

Decomposes execution cost into:
1. Delay Cost: (Arrival Price - Decision Price) * Shares
2. Market Impact: (Exec Price - Mid Price) * Exec Shares
3. Timing Risk: (Mid Price - Arrival Price) * Exec Shares
4. Opportunity Cost: (Closing Price - Decision Price) * Unexecuted Shares
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

@dataclass
class ISC_Components:
    decision_price: float
    arrival_price: float
    total_shares: int
    executed_shares: int
    
    total_shortfall: float
    delay_cost: float
    market_impact: float
    timing_risk: float
    opportunity_cost: float
    
    avg_exec_price: float
    
    def to_dict(self) -> Dict:
        return {
            'Total IS': self.total_shortfall,
            'Delay Cost': self.delay_cost,
            'Market Impact': self.market_impact,
            'Timing Risk': self.timing_risk,
            'Opportunity Cost': self.opportunity_cost 
        }

class ISAnalyzer:
    """Calculates Implementation Shortfall components."""
    
    def __init__(self, decision_price: float, total_orders: int):
        self.decision_price = decision_price
        self.total_orders = total_orders
        
    def analyze(self, execution_log: pd.DataFrame, market_data: pd.DataFrame) -> ISC_Components:
        """
        Analyze execution log to calculate IS components.
        
        Args:
            execution_log: DataFrame with [timestamp, shares, price]
            market_data: DataFrame with [timestamp, price] (mid price)
            
        Returns:
            ISC_Components object
        """
        if execution_log.empty:
            return ISC_Components(0,0,0,0,0,0,0,0,0,0)
        
        # Merge execution log with market data on timestamp
        # Assume closest timestamp match if not exact
        df = pd.merge_asof(
            execution_log.sort_values('timestamp'),
            market_data.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            suffixes=('_exec', '_mkt')
        )
        
        executed_shares = df['shares'].sum()
        unexecuted_shares = self.total_orders - executed_shares
        
        arrival_price = market_data.iloc[0]['price'] # Price at start of window
        last_price = market_data.iloc[-1]['price']
        
        # 1. Delay Cost: (Arrival - Decision) * Total Shares
        # Assumes we intended to trade Total Shares
        delay_cost = (arrival_price - self.decision_price) * self.total_orders
        
        # 2. Market Impact: (Exec Price - Mid Price) * Exec Shares
        # Instantaneous cost paid relative to fair mid price at that moment
        # Note: df['price_exec'] is the fill price, df['price_mkt'] is mid price
        impact_cost = ((df['price_exec'] - df['price_mkt']) * df['shares']).sum()
        
        # 3. Timing Risk (Trend): (Mid Price - Arrival Price) * Exec Shares
        # Cost due to market moving while we were trading
        timing_risk = ((df['price_mkt'] - arrival_price) * df['shares']).sum()
        
        # 4. Opportunity Cost: (Closing Price - Decision Price) * Unexecuted Shares
        opportunity_cost = (last_price - self.decision_price) * unexecuted_shares
        
        # Total IS check
        # IS = (AvgExec - Decision) * Exec + (Close - Decision) * Unexec
        avg_exec_price = (df['price_exec'] * df['shares']).sum() / executed_shares
        
        # IS_executed = (AvgExec - Decision) * Executed
        # IS_executed = Delay + Impact + Timing (roughly, for Executed portion)
        # Let's use the component sum as Total IS
        
        total_shortfall = delay_cost + impact_cost + timing_risk + opportunity_cost
        
        return ISC_Components(
            decision_price=self.decision_price,
            arrival_price=arrival_price,
            total_shares=self.total_orders,
            executed_shares=executed_shares,
            total_shortfall=total_shortfall,
            delay_cost=delay_cost,
            market_impact=impact_cost,
            timing_risk=timing_risk,
            opportunity_cost=opportunity_cost,
            avg_exec_price=avg_exec_price
        )

def plot_is_breakdown(strategies: Dict[str, ISC_Components], filename: str = "is_breakdown.png"):
    """Create stacked bar chart of IS components."""
    
    labels = list(strategies.keys())
    components = ['Delay Cost', 'Market Impact', 'Timing Risk', 'Opportunity Cost']
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    # Prepare data
    data = {comp: [] for comp in components}
    for strat in strategies.values():
        d = strat.to_dict()
        for comp in components:
            data[comp].append(d[comp])
            
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bottom = np.zeros(len(labels))
    
    for i, comp in enumerate(components):
        vals = np.array(data[comp])
        ax.bar(labels, vals, bottom=bottom, label=comp, color=colors[i], width=0.5)
        bottom += vals
        
    ax.set_title('Implementation Shortfall Decomposition')
    ax.set_ylabel('Cost ($)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values
    for i, total in enumerate(bottom):
        ax.text(i, total, f"${total:,.0f}", ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot: {filename}")

if __name__ == "__main__":
    pass
