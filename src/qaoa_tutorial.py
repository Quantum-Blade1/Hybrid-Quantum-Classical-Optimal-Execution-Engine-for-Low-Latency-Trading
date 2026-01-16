"""
QAOA Tutorial: Max-Cut Problem with Qiskit

============================================================================
QUANTUM APPROXIMATE OPTIMIZATION ALGORITHM (QAOA)
============================================================================

QAOA is a hybrid quantum-classical algorithm for solving combinatorial
optimization problems. It encodes the problem into a quantum circuit,
then uses a classical optimizer to find the best circuit parameters.

QAOA consists of:
1. COST HAMILTONIAN (H_C): Encodes the problem objective
2. MIXER HAMILTONIAN (H_M): Enables exploration of solution space
3. PARAMETERIZED CIRCUIT: Alternates between H_C and H_M applications
4. CLASSICAL OPTIMIZER: Finds optimal parameters (β, γ)

============================================================================
MAX-CUT PROBLEM
============================================================================

Given a graph G = (V, E), partition vertices into two sets to maximize
the number of edges crossing the partition.

For a 4-node graph:
    0 --- 1
    |     |
    3 --- 2

Max-Cut = 4 (cut all edges by placing {0,2} in one set, {1,3} in other)

QUBO formulation:
    max Σ_{(i,j)∈E} x_i(1 - x_j) + (1 - x_i)x_j
    = max Σ_{(i,j)∈E} (x_i + x_j - 2*x_i*x_j)

Ising formulation (for quantum):
    H_C = Σ_{(i,j)∈E} (1 - Z_i Z_j) / 2

============================================================================
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# =============================================================================
# Step 1: Define the Graph
# =============================================================================

@dataclass
class Graph:
    """
    Simple graph representation for Max-Cut.
    
    Vertices are numbered 0 to n-1.
    Edges are tuples (i, j) with i < j.
    """
    num_vertices: int
    edges: List[Tuple[int, int]]
    
    @classmethod
    def create_square(cls) -> 'Graph':
        """
        Create a 4-node square graph:
        
            0 --- 1
            |     |
            3 --- 2
        """
        return cls(
            num_vertices=4,
            edges=[(0, 1), (1, 2), (2, 3), (0, 3)]
        )
    
    @classmethod
    def create_triangle(cls) -> 'Graph':
        """Create a 3-node triangle."""
        return cls(
            num_vertices=3,
            edges=[(0, 1), (1, 2), (0, 2)]
        )
    
    def max_cut_value(self, partition: List[int]) -> int:
        """
        Calculate the Max-Cut value for a given partition.
        
        Args:
            partition: Binary list where partition[i] = 0 or 1
            
        Returns:
            Number of edges crossing the partition
        """
        cut_value = 0
        for i, j in self.edges:
            if partition[i] != partition[j]:
                cut_value += 1
        return cut_value


# =============================================================================
# Step 2: Build QAOA Circuit
# =============================================================================

def build_qaoa_circuit(
    graph: Graph,
    gamma: float,
    beta: float,
    p: int = 1
):
    """
    Build the QAOA circuit for Max-Cut.
    
    The circuit structure is:
    1. Initial state: |+⟩^n (uniform superposition)
    2. For each layer l = 1 to p:
       a) Apply cost unitary: e^{-iγ_l H_C}
       b) Apply mixer unitary: e^{-iβ_l H_M}
    3. Measure in computational basis
    
    Cost Hamiltonian H_C:
        For each edge (i,j), apply: RZZ(2*gamma) on qubits i,j
        This implements e^{-iγ(1-Z_i Z_j)/2}
    
    Mixer Hamiltonian H_M:
        For each qubit i, apply: RX(2*beta)
        This implements e^{-iβ X_i}
    
    Args:
        graph: Graph for Max-Cut
        gamma: Cost layer parameter
        beta: Mixer layer parameter
        p: Number of QAOA layers (depth)
        
    Returns:
        Qiskit QuantumCircuit
    """
    from qiskit import QuantumCircuit
    
    n = graph.num_vertices
    qc = QuantumCircuit(n, n)  # n qubits, n classical bits
    
    # =========================================================================
    # Initial state: Apply Hadamard to all qubits to create |+⟩^n
    # This creates an equal superposition of all 2^n possible partitions
    # =========================================================================
    for i in range(n):
        qc.h(i)
    
    qc.barrier()  # Visual separator
    
    # =========================================================================
    # QAOA layers
    # =========================================================================
    for layer in range(p):
        # -----------------------------------------------------------------
        # Cost Layer: e^{-iγ H_C}
        # 
        # For Max-Cut, H_C = Σ_{(i,j)∈E} (1 - Z_i Z_j) / 2
        # 
        # We implement e^{-iγ(1-Z_i Z_j)/2} for each edge using:
        # - RZZ gate: e^{-iθ Z_i Z_j}
        # - Global phase (ignored)
        # 
        # RZZ(2γ) implements e^{-iγ Z_i Z_j}
        # -----------------------------------------------------------------
        for i, j in graph.edges:
            # RZZ gate on qubits i, j with angle 2*gamma
            qc.rzz(2 * gamma, i, j)
        
        qc.barrier()
        
        # -----------------------------------------------------------------
        # Mixer Layer: e^{-iβ H_M}
        # 
        # H_M = Σ_i X_i (sum of Pauli-X on each qubit)
        # 
        # e^{-iβ X_i} is implemented by RX(2β)
        # -----------------------------------------------------------------
        for i in range(n):
            qc.rx(2 * beta, i)
        
        qc.barrier()
    
    # =========================================================================
    # Measure all qubits
    # =========================================================================
    qc.measure(range(n), range(n))
    
    return qc


# =============================================================================
# Step 3: QAOA Cost Function
# =============================================================================

def qaoa_cost_function(
    params: np.ndarray,
    graph: Graph,
    p: int = 1,
    shots: int = 1000
) -> float:
    """
    QAOA cost function for classical optimizer.
    
    This function:
    1. Builds the QAOA circuit with given parameters
    2. Runs it on the simulator
    3. Computes the expectation value of the Max-Cut cost
    
    IMPORTANT: We NEGATE the cost because optimizers minimize,
    but Max-Cut wants to maximize.
    
    Args:
        params: Array [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
        graph: Graph for Max-Cut
        p: Number of QAOA layers
        shots: Number of measurement shots
        
    Returns:
        Negative expected Max-Cut value (for minimization)
    """
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    
    # Extract gamma and beta from params
    gammas = params[:p]
    betas = params[p:]
    
    # For p=1, we use scalar values
    gamma = gammas[0] if p == 1 else gammas
    beta = betas[0] if p == 1 else betas
    
    # Build circuit
    qc = build_qaoa_circuit(graph, gamma, beta, p)
    
    # Run on simulator
    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()
    
    # Calculate expected Max-Cut value
    expected_value = 0.0
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        # Convert bitstring to partition (note: Qiskit reverses bit order)
        partition = [int(b) for b in bitstring[::-1]]
        cut_value = graph.max_cut_value(partition)
        expected_value += cut_value * count / total_shots
    
    # Return negative because we minimize, but want to maximize cut
    return -expected_value


# =============================================================================
# Step 4: Optimize QAOA Parameters
# =============================================================================

def optimize_qaoa(
    graph: Graph,
    p: int = 1,
    shots: int = 1000,
    maxiter: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Run QAOA optimization using COBYLA.
    
    COBYLA (Constrained Optimization BY Linear Approximations) is a
    gradient-free optimizer suitable for noisy quantum cost functions.
    
    Args:
        graph: Graph for Max-Cut
        p: Number of QAOA layers
        shots: Shots per circuit evaluation
        maxiter: Maximum optimizer iterations
        verbose: Print progress
        
    Returns:
        Dictionary with:
        - optimal_params: Best (gamma, beta) values
        - optimal_value: Best expected cut value
        - best_partition: Most likely optimal partition
        - counts: Final measurement statistics
    """
    from scipy.optimize import minimize
    
    if verbose:
        print(f"\n{'='*60}")
        print(f" QAOA Optimization for Max-Cut (p={p})")
        print(f"{'='*60}")
        print(f" Graph: {graph.num_vertices} vertices, {len(graph.edges)} edges")
    
    # Initial parameters
    # gamma ∈ [0, 2π], beta ∈ [0, π]
    np.random.seed(42)
    x0 = np.concatenate([
        np.random.uniform(0, 2*np.pi, p),  # gammas
        np.random.uniform(0, np.pi, p)      # betas
    ])
    
    if verbose:
        print(f" Initial params: gamma={x0[:p]}, beta={x0[p:]}")
    
    # Optimization callback
    iteration = [0]
    def callback(xk):
        iteration[0] += 1
        if verbose and iteration[0] % 10 == 0:
            cost = qaoa_cost_function(xk, graph, p, shots)
            print(f"   Iter {iteration[0]}: expected cut = {-cost:.3f}")
    
    # Run COBYLA optimization
    result = minimize(
        qaoa_cost_function,
        x0,
        args=(graph, p, shots),
        method='COBYLA',
        options={'maxiter': maxiter, 'rhobeg': 0.5},
        callback=callback
    )
    
    optimal_params = result.x
    optimal_value = -result.fun  # Negate back to get max cut value
    
    if verbose:
        print(f"\n Optimization complete!")
        print(f" Optimal gamma: {optimal_params[:p]}")
        print(f" Optimal beta:  {optimal_params[p:]}")
        print(f" Expected Max-Cut: {optimal_value:.3f}")
    
    # Get final measurement results
    gamma = optimal_params[0] if p == 1 else optimal_params[:p]
    beta = optimal_params[p] if p == 1 else optimal_params[p:]
    
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    
    qc = build_qaoa_circuit(graph, gamma, beta, p)
    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    final_result = simulator.run(compiled, shots=shots*10).result()
    counts = final_result.get_counts()
    
    # Find most common result
    best_bitstring = max(counts, key=counts.get)
    best_partition = [int(b) for b in best_bitstring[::-1]]
    best_cut = graph.max_cut_value(best_partition)
    
    if verbose:
        print(f"\n Most likely partition: {best_partition}")
        print(f" Actual Max-Cut value: {best_cut}")
        print(f"\n Top 5 measurement results:")
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
        for bitstring, count in sorted_counts[:5]:
            partition = [int(b) for b in bitstring[::-1]]
            cut = graph.max_cut_value(partition)
            print(f"   {bitstring} (cut={cut}): {count} shots ({100*count/sum(counts.values()):.1f}%)")
    
    return {
        'optimal_params': optimal_params,
        'optimal_gammas': optimal_params[:p],
        'optimal_betas': optimal_params[p:],
        'optimal_value': optimal_value,
        'best_partition': best_partition,
        'best_cut': best_cut,
        'counts': counts,
        'num_iterations': iteration[0]
    }


# =============================================================================
# Step 5: Visualization
# =============================================================================

def visualize_qaoa_results(
    graph: Graph,
    results: Dict,
    save_path: Optional[str] = None
):
    """
    Visualize QAOA optimization results.
    
    Creates a figure with:
    1. Graph with optimal partition coloring
    2. Measurement probability distribution
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Graph with partition
    ax1 = axes[0]
    
    # Position nodes in a circle
    n = graph.num_vertices
    angles = [2 * np.pi * i / n for i in range(n)]
    positions = [(np.cos(a), np.sin(a)) for a in angles]
    
    # Draw edges
    for i, j in graph.edges:
        x1, y1 = positions[i]
        x2, y2 = positions[j]
        partition = results['best_partition']
        # Color edges that are cut
        color = 'red' if partition[i] != partition[j] else 'gray'
        linewidth = 3 if partition[i] != partition[j] else 1
        ax1.plot([x1, x2], [y1, y2], c=color, linewidth=linewidth, zorder=1)
    
    # Draw nodes
    partition = results['best_partition']
    for i, (x, y) in enumerate(positions):
        color = 'blue' if partition[i] == 0 else 'orange'
        ax1.scatter(x, y, s=500, c=color, zorder=2, edgecolors='black', linewidth=2)
        ax1.text(x, y, str(i), ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title(f"Max-Cut Solution (Cut = {results['best_cut']})\nBlue=Set0, Orange=Set1, Red=Cut Edges")
    ax1.axis('off')
    
    # Plot 2: Measurement distribution
    ax2 = axes[1]
    
    counts = results['counts']
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:10]
    
    bitstrings = [bs for bs, _ in sorted_counts]
    probs = [c / sum(counts.values()) for _, c in sorted_counts]
    cuts = [graph.max_cut_value([int(b) for b in bs[::-1]]) for bs in bitstrings]
    
    colors = ['green' if c == results['best_cut'] else 'lightblue' for c in cuts]
    bars = ax2.bar(range(len(bitstrings)), probs, color=colors)
    
    ax2.set_xticks(range(len(bitstrings)))
    ax2.set_xticklabels([f"{bs}\n(cut={c})" for bs, c in zip(bitstrings, cuts)], fontsize=8)
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('Bitstring')
    ax2.set_title('Measurement Distribution (Top 10)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


# =============================================================================
# Main Demo
# =============================================================================

def run_qaoa_demo():
    """Run the complete QAOA Max-Cut demo."""
    
    print("\n" + "="*70)
    print(" QAOA Tutorial: Solving Max-Cut with Qiskit")
    print("="*70)
    
    # Create graph
    graph = Graph.create_square()
    print(f"\n Graph: 4-node square")
    print(f" Edges: {graph.edges}")
    print(f" Optimal Max-Cut: 4 (can cut all edges)")
    
    # Run QAOA optimization
    results = optimize_qaoa(
        graph=graph,
        p=1,        # Single QAOA layer
        shots=1000,
        maxiter=50,
        verbose=True
    )
    
    # Visualize
    print("\n Generating visualization...")
    visualize_qaoa_results(graph, results, save_path="qaoa_maxcut.png")
    
    print("\n" + "="*70)
    print(" Demo Complete!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = run_qaoa_demo()
