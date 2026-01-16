"""
Almgren-Chriss Optimal Execution Model

Implements the classic optimal execution framework (Almgren & Chriss, 2000).
Provides closed-form solution for optimal trading trajectory minimizing
Expected Cost + Lambda * Variance.

Model parameters:
- sigma: Daily volatility
- eta: Temporary impact coefficient
- rho: Permanent impact coefficient
- lambda: Risk aversion
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ACConfig:
    total_shares: int
    n_days: float = 1.0  # Fraction of day usually
    n_steps: int = 10
    sigma: float = 0.02  # Daily volatility
    price: float = 100.0
    daily_volume: int = 5_000_000
    
    # Impact coefficients (estimated)
    # Temporary impact: Cost ~ eta * v
    eta: float = 0.05 / 10000  # $0.05 per 10k shares rate
    
    # Permanent impact: Cost ~ rho * X
    rho: float = 0.01 / 10000 
    
    # Risk aversion
    risk_aversion: float = 1e-6

class AlmgrenChrissSolver:
    """Calculates optimal execution trajectory using Almgren-Chriss model."""
    
    def __init__(self, config: ACConfig):
        self.config = config
    
    def compute_trajectory(self) -> pd.DataFrame:
        """
        Compute optimal trading schedule.
        
        Returns:
            DataFrame with [time, shares_held, shares_to_trade]
        """
        X = self.config.total_shares
        T = self.config.n_days
        N = self.config.n_steps
        tau = T / N
        
        # Almgren-Chriss parameters
        # sigma^2 is daily variance of price change (absolute)
        # We need variance per time step? 
        # AC formula uses sigma = volatility of the asset per unit time
        
        # Kappa calculation:
        # kappa = sqrt(lambda * sigma^2 / eta)
        # Assuming linear temporary impact
        
        sig2 = (self.config.sigma * self.config.price) ** 2  # Variance in $^2
        lam = self.config.risk_aversion
        eta = self.config.eta
        
        # Avoid divide by zero
        if abs(eta) < 1e-9:
            kappa = 100.0 # Fast execution
        else:
            kappa = np.sqrt(lam * sig2 / eta)
            
        # Time steps
        t = np.linspace(0, T, N + 1)
        
        # Calculate optimal shares holding x(t)
        # x(t) = X * sinh(kappa(T-t)) / sinh(kappa*T)
        
        # Handle small kappa (risk neutral -> TWAP)
        if kappa * T < 1e-4:
            # Limit is linear (TWAP)
            x_t = X * (1 - t/T)
        else:
            x_t = X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
            
        # Shares to trade in each interval (n_j)
        # n_j = x_{j-1} - x_j
        shares_to_trade = -np.diff(x_t)
        
        # Last element of diff is last step
        
        schedule = pd.DataFrame({
            'step': range(N),
            'time': t[1:], # End of interval
            'shares_held_start': x_t[:-1],
            'shares_held_end': x_t[1:],
            'shares_to_trade': shares_to_trade
        })
        
        return schedule
    
    def calculate_expected_cost(self, trajectory: pd.DataFrame) -> float:
        """Calculate theoretical expected cost E[C]."""
        # E[C] = Permanent + Temporary
        # Perm = 0.5 * gamma * X^2 (gamma = permanent impact)
        # Temp = sum(eta * n_j^2 / tau)
        
        X = self.config.total_shares
        rho = self.config.rho
        eta = self.config.eta
        tau = self.config.n_days / self.config.n_steps
        
        perm_cost = 0.5 * rho * (X ** 2)
        
        n = trajectory['shares_to_trade'].values
        temp_cost = np.sum(eta * (n ** 2) / tau)
        
        return perm_cost + temp_cost
        
    def calculate_variance(self, trajectory: pd.DataFrame) -> float:
        """Calculate variance of cost V[C]."""
        # V[C] = sigma^2 * sum(x_j^2 * tau)
        
        sig2 = (self.config.sigma * self.config.price) ** 2
        tau = self.config.n_days / self.config.n_steps
        
        # x_j is shares held (using approximation as sum of integrals or discrete sum)
        # AC formula: sum_{j=1}^N (tau * sig2 * x_{j-1}^2) 
        # (Actually depends on exact implementation, simple Riemann sum here)
        
        x = trajectory['shares_held_start'].values
        variance = sig2 * np.sum((x ** 2) * tau)
        
        return variance

def run_ac_benchmark_demo():
    """Run demonstration of AC model vs TWAP."""
    total_shares = 10000
    n_days = 1/26  # 15 minutes roughly (assuming 6.5h day)
    n_steps = 15
    
    print("\n" + "="*60)
    print(" Almgren-Chriss Benchmark")
    print("="*60)
    
    # 1. Risk Neutral (Lambda ~ 0) -> Should look like TWAP
    config_rn = ACConfig(
        total_shares=total_shares, n_days=n_days, n_steps=n_steps,
        risk_aversion=1e-10
    )
    solver_rn = AlmgrenChrissSolver(config_rn)
    sched_rn = solver_rn.compute_trajectory()
    
    # 2. High Risk Aversion -> Should front-load execution
    config_ra = ACConfig(
        total_shares=total_shares, n_days=n_days, n_steps=n_steps,
        risk_aversion=1e-4
    )
    solver_ra = AlmgrenChrissSolver(config_ra)
    sched_ra = solver_ra.compute_trajectory()
    
    # Comparison
    print(f"{'Step':<5} {'TWAP (n)':<10} {'AC-RiskNeutral':<15} {'AC-RiskAverse':<15}")
    print("-" * 50)
    
    twap_rate = total_shares / n_steps
    for i in range(n_steps):
        rn_Trade = sched_rn.iloc[i]['shares_to_trade']
        ra_Trade = sched_ra.iloc[i]['shares_to_trade']
        
        print(f"{i:<5} {twap_rate:<10.0f} {rn_Trade:<15.1f} {ra_Trade:<15.1f}")
        
    print("\nanalysis:")
    print("Risk Neutral: Constant rate (matches TWAP)")
    print("Risk Averse:  Front-loaded (trade more early to reduce exposure)")

if __name__ == "__main__":
    run_ac_benchmark_demo()
