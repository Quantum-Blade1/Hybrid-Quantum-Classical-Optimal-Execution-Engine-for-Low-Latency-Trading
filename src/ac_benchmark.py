"""
Almgren-Chriss vs Hybrid Benchmark

Compares the Quantum-Hybrid optimization against the closed-form
Almgren-Chriss optimal trajectory across different risk aversion levels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

from src.almgren_chriss import AlmgrenChrissSolver, ACConfig
from src.hybrid_async import HybridController
from src.market_data import MarketDataSimulator

def run_comparison(
    total_shares: int = 50000,
    n_steps: int = 20,
    risk_aversion: float = 1e-5
) -> None:
    print(f"\nRunning Comparison (Risk Aversion lambda={risk_aversion:.1e})...")
    
    # 1. Almgren-Chriss Optimal
    ac_config = ACConfig(
        total_shares=total_shares,
        n_days=1/26, # ~15 mins
        n_steps=n_steps,
        risk_aversion=risk_aversion
    )
    ac_solver = AlmgrenChrissSolver(ac_config)
    ac_traj = ac_solver.compute_trajectory()
    ac_schedule = ac_traj['shares_to_trade'].values
    
    # 2. Hybrid (SA) Optimization
    # We need to map lambda to our 'equality_penalty' or similar in QUBO?
    # Actually, our current QUBO formulation uses 'equality_penalty' for constraints,
    # but doesn't explicitely have a 'risk' term in the demo version (it's in the full formulation).
    # For this benchmark, we'll assume the Hybrid system is configured to mimic risk aversion
    # via its internal cost function (e.g. by penalizing late execution if market is volatile).
    # NOTE: The current demo QUBO is mostly impact-minimizing (Risk Neutral-ish).
    # To make it fair, we'll run it as is and see how it compares to AC-RiskNeutral.
    
    controller = HybridController(
        optimizer_type='sa',
        optimizer_interval=10.0, # Run once effectively
        engine_tick_interval=0.01 
    )
    
    # Force single optimization with specific parameters if possible, 
    # but controller runs loop. We'll just execute and capture the result.
    
    # Actually, let's use the optimizer directly to get the schedule
    # to avoid the noise of the execution engine for the trajectory plot.
    params = controller.optimizer._optimize_sa(total_shares, n_steps)
    hybrid_schedule = params
    
    # 3. Calculate metrics
    # Deviation from AC optimal
    mse = np.mean((hybrid_schedule - ac_schedule) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"  RMSE (Hybrid vs AC): {rmse:.1f} shares")
    
    # 4. Plot
    plt.figure(figsize=(10, 6))
    
    steps = np.arange(n_steps)
    plt.plot(steps, ac_schedule, 'b-o', label='Almgren-Chriss (Optimal)')
    plt.plot(steps, hybrid_schedule, 'r--s', label='Hybrid (SA-QUBO)')
    plt.axhline(total_shares / n_steps, color='g', linestyle=':', label='TWAP')
    
    plt.title(f'Trajectory Comparison (Risk Aversion $\\lambda={risk_aversion:.1e}$)')
    plt.xlabel('Time Step')
    plt.ylabel('Shares Traded')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"ac_vs_hybrid_lambda_{risk_aversion:.1e}.png"
    plt.savefig(filename)
    print(f"  Saved plot: {filename}")
    
    # Save raw data
    df = pd.DataFrame({
        'Step': steps,
        'AC_Optimal': ac_schedule,
        'Hybrid_SA': hybrid_schedule,
        'Diff': hybrid_schedule - ac_schedule
    })
    print("\nComparison Table (First 5 steps):")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    # Run with Lambda corresponding to Risk Neutral behavior (where our QUBO is tuned)
    # AC with lambda=1e-9 is effectively TWAP
    run_comparison(risk_aversion=1e-9)
    
    # Also run with higher risk aversion to show AC shifts
    run_comparison(risk_aversion=1e-4)
