
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import time
from typing import List, Dict

# Set academic style
plt.style.use('fast')
plt.rcParams.update({
    'font.family': 'sans-serif', # Often preferred in nature/science
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 14
})

def generate_figure_1_architecture():
    """Generate System Architecture Diagram."""
    print("Generating Figure 1: System Architecture...")
    
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Define styles
    box_style = dict(boxstyle='round,pad=1', fc='white', ec='black', lw=1.5)
    fast_style = dict(boxstyle='round,pad=1', fc='#e6f3ff', ec='#0066cc', lw=1.5) # Blue tint
    slow_style = dict(boxstyle='round,pad=1', fc='#fff0e6', ec='#cc3300', lw=1.5) # Red tint
    
    # Nodes
    # Fast Path (Top)
    ax.text(20, 80, "Market Data\nFeed", ha='center', va='center', bbox=box_style, size=10)
    ax.text(50, 80, "Execution\nEngine\n(Async Fast Path)", ha='center', va='center', bbox=fast_style, size=11, weight='bold')
    ax.text(80, 80, "Exchange /\nOrder Book", ha='center', va='center', bbox=box_style, size=10)
    
    # Slow Path (Bottom)
    ax.text(50, 40, "Hybrid\nOptimizer\n(Slow Path)", ha='center', va='center', bbox=slow_style, size=11, weight='bold')
    ax.text(20, 40, "Policy\nQueue", ha='center', va='center', bbox=box_style, size=10)
    ax.text(80, 40, "Quantum\nBackends\n(QPU / SA)", ha='center', va='center', bbox=dict(boxstyle='round,pad=1', fc='#f2e6ff', ec='#6600cc'), size=10)
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=1.5, color='black')
    dashed_arrow = dict(arrowstyle='->', lw=1.5, color='black', linestyle='--')
    
    # Fast Loop
    ax.annotate("", xy=(30, 80), xytext=(20, 85), arrowprops=arrow_props) # Data -> Engine
    ax.annotate("Ticks", xy=(35, 82), ha='center', size=8)
    
    ax.annotate("", xy=(65, 80), xytext=(80, 85), arrowprops=arrow_props) # Engine -> Exchange (Orders)
    ax.annotate("Orders", xy=(72, 82), ha='center', size=8)
    ax.annotate("", xy=(80, 75), xytext=(65, 78), arrowprops=dict(arrowstyle='->', lw=1.5, ls='--')) # Exchange -> Engine (Fills)
    
    # Slow Loop couplings
    ax.annotate("", xy=(50, 55), xytext=(50, 70), arrowprops=arrow_props) # Engine -> Optimizer (State)
    ax.annotate("State Update\n(Async)", xy=(52, 62), ha='left', size=8)
    
    ax.annotate("", xy=(80, 50), xytext=(65, 45), arrowprops=arrow_props) # Optimizer -> Quantum
    ax.annotate("QUBO", xy=(72, 52), ha='center', size=8)
    
    ax.annotate("", xy=(65, 35), xytext=(80, 30), arrowprops=arrow_props) # Quantum -> Optimizer
    ax.annotate("Bitstring", xy=(72, 28), ha='center', size=8)
    
    ax.annotate("", xy=(20, 50), xytext=(35, 40), arrowprops=arrow_props) # Optimizer -> Queue
    ax.annotate("Policy", xy=(28, 48), ha='center', size=8)
    
    ax.annotate("", xy=(35, 70), xytext=(20, 60), arrowprops=arrow_props) # Queue -> Engine
    ax.annotate("Poll Policy", xy=(25, 68), ha='center', size=8)
    
    # Background Regions
    # Fast Region
    rect_fast = patches.Rectangle((5, 60), 90, 35, linewidth=1, edgecolor='none', facecolor='blue', alpha=0.05)
    ax.add_patch(rect_fast)
    ax.text(10, 92, "Latency-Critical Domain (<10ms)", color='#0066cc', weight='bold')
    
    # Slow Region
    rect_slow = patches.Rectangle((5, 5), 90, 50, linewidth=1, edgecolor='none', facecolor='red', alpha=0.05)
    ax.add_patch(rect_slow)
    ax.text(10, 10, "Compute-Intensive Domain (>100ms)", color='#cc3300', weight='bold')
    
    plt.tight_layout()
    plt.savefig('figure1_architecture.png', dpi=300)
    plt.close()
    print("Saved figure1_architecture.png")


def generate_figure_3_scaling():
    """Generate Scaling Analysis Plot."""
    print("Generating Figure 3: Scaling Analysis...")
    
    from src.qubo_solvers import BruteForceSolver, SimulatedAnnealingSolver
    from src.qubo_execution import QUBOConfig, ExecutionQUBO
    
    problem_sizes = [4, 6, 8, 10, 12, 14, 16]
    times_bf = []
    times_sa = []
    
    print("  Running benchmarks...")
    for n in problem_sizes:
        # Build Problem
        config = QUBOConfig(total_shares=100, num_time_slices=n, num_venues=1, quantity_levels=[0,50,100])
        qubo = ExecutionQUBO(config)
        Q = qubo.build_qubo_matrix()
        
        # 1. Brute Force (Scale limit)
        if n <= 12:
            start = time.time()
            BruteForceSolver().solve(Q)
            times_bf.append(time.time() - start)
        else:
            times_bf.append(np.nan)
            
        # 2. Simulated Annealing
        start = time.time()
        SimulatedAnnealingSolver(num_sweeps=100).solve(Q)
        times_sa.append(time.time() - start)
        
    # Plot
    fig, ax = plt.figure(figsize=(8, 6)), plt.gca()
    
    ax.plot(problem_sizes[:len(times_bf)], times_bf, 'o--', label='Classical Exact (Brute Force)', color='red')
    ax.plot(problem_sizes, times_sa, 's-', label='Classical Heuristic (Sim Ann)', color='blue')
    
    ax.set_yscale('log')
    ax.set_xlabel('Problem Size (N slices)')
    ax.set_ylabel('Time to Solution (s)')
    ax.set_title('Solver Scaling Analysis')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('figure3_scaling.png', dpi=300)
    plt.close()
    print("Saved figure3_scaling.png")

if __name__ == "__main__":
    generate_figure_1_architecture()
    generate_figure_3_scaling()
