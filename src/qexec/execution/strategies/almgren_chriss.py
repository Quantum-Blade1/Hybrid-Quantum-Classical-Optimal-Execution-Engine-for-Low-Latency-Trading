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
