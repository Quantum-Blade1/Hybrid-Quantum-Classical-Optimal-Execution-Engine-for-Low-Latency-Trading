"""
Comprehensive Benchmark & Figure Generation for Springer Journal Paper

Generates publication-quality figures (PDF, 300 DPI) covering:
  1. System Architecture & QUBO Formulation  (Figs 1-4)
  2. Classical vs Quantum Solver Performance  (Figs 5-9)
  3. Market Microstructure Analysis           (Figs 10-12)
  4. Adaptive Risk Aversion                   (Figs 15-18)
  5. Execution Performance & Walk-Forward     (Figs 19-21)
  6. HFT Pipeline                             (Fig 27)
  7. Quantum-Specific                         (Fig 28)

Figures 13, 14, 22-26, 29 and 30 were removed because they plotted
hand-typed or randomly generated numbers rather than experiment output
(see docs/CLAIMS_AUDIT.md). Numbering is kept to avoid renaming files.

Springer formatting:
  - Single column: 3.5 in wide
  - Double column: 7.0 in wide
  - Font: 8-10 pt serif
  - 300 DPI PDF output

Usage:
    python experiments/journal_figures.py
"""

import numpy as np
import time
import os
import logging

logging.disable(logging.INFO)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches


SPRINGER_RC = {
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
}
plt.rcParams.update(SPRINGER_RC)

OUTPUT_DIR = "paper/figures"
SEED = 42


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    return path


# ═════════════════════════════════════════════════════════════════════
# SECTION 1: System Architecture & QUBO Formulation (Figs 1-4)
# ═════════════════════════════════════════════════════════════════════

def fig01_system_architecture():
    """Fig 1: High-level system architecture diagram."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title("Hybrid Quantum-Classical Execution Architecture", fontsize=11, fontweight='bold')

    boxes = [
        (0.5, 4.5, 2.2, 1.0, "Market Data\nTick Stream", "#E3F2FD"),
        (0.5, 2.5, 2.2, 1.0, "Microstructure\nAnalyzer", "#E8F5E9"),
        (0.5, 0.5, 2.2, 1.0, "Adaptive Risk\nManager", "#FFF3E0"),
        (3.5, 3.5, 2.2, 1.2, "QUBO\nFormulation\n(6-term cost)", "#F3E5F5"),
        (3.5, 1.2, 2.2, 1.2, "Quantum/Classical\nSolver\n(QAOA / SA)", "#FCE4EC"),
        (6.5, 4.0, 2.8, 0.8, "Fast Path (<1ms)", "#E0F7FA"),
        (6.5, 2.5, 2.8, 0.8, "Policy Queue", "#FFF9C4"),
        (6.5, 1.0, 2.8, 0.8, "Slow Path (100ms-5s)", "#FFEBEE"),
        (6.5, -0.2, 2.8, 0.8, "Execution Orders", "#E8EAF6"),
    ]

    for x, y, w, h, text, color in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='#333', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=7, fontweight='bold')

    arrows = [
        (1.6, 4.5, 0, -0.4),
        (1.6, 2.5, 0, -0.4),
        (2.7, 3.0, 0.7, 0.8),
        (2.7, 1.0, 0.7, 0.5),
        (5.7, 4.0, 0.7, 0.2),
        (5.7, 2.0, 0.7, 0.2),
        (7.9, 4.0, 0, -0.5),
        (7.9, 2.5, 0, -0.5),
        (7.9, 1.0, 0, -0.5),
    ]

    for x, y, dx, dy in arrows:
        ax.annotate('', xy=(x+dx, y+dy), xytext=(x, y),
                     arrowprops=dict(arrowstyle='->', color='#555', lw=1.0))

    return save_fig(fig, "fig01_system_architecture.pdf")


def fig02_qubo_matrix_structure():
    """Fig 2: QUBO matrix structure and sparsity pattern."""
    from qexec.optimization.qubo import QUBOConfig, ExecutionQUBO

    config = QUBOConfig(
        total_shares=1000, num_time_slices=5, num_venues=1,
        quantity_levels=[0, 100, 200, 300],
        equality_penalty=100.0, capacity_penalty=50.0
    )
    qubo = ExecutionQUBO(config)
    Q = qubo.build_qubo_matrix()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    im1 = ax1.imshow(np.abs(Q), cmap='YlOrRd', aspect='equal')
    ax1.set_title("QUBO Matrix |Q| (Execution)")
    ax1.set_xlabel("Variable Index")
    ax1.set_ylabel("Variable Index")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    ax2.spy(Q, markersize=1.5, color='#1565C0')
    ax2.set_title(f"Sparsity Pattern\n({Q.shape[0]}x{Q.shape[0]}, "
                  f"nnz={np.count_nonzero(Q)})")
    ax2.set_xlabel("Variable Index")

    fig.tight_layout()
    return save_fig(fig, "fig02_qubo_matrix_structure.pdf")


def fig03_hft_qubo_cost_decomposition():
    """Fig 3: HFT QUBO 6-term cost function breakdown."""
    from qexec.optimization.hft_qubo import HFTQUBOConfig, HFTExecutionQUBO
    from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver

    config = HFTQUBOConfig(
        total_shares=2000, num_tick_slices=10, num_venues=3,
        quantity_levels=[0, 100, 250, 500],
        kyle_lambda=0.001, vpin=0.4, adverse_selection_cost=0.0002
    )
    qubo = HFTExecutionQUBO(config)
    Q = qubo.build_qubo_matrix()

    solver = SimulatedAnnealingSolver(num_sweeps=500, seed=SEED)
    result = solver.solve(Q, verbose=False)
    costs = qubo.calculate_cost_breakdown(result.solution)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    labels = ["Market\nImpact", "Timing\nRisk", "Transaction\nCost",
              "Adverse\nSelection", "Inventory\nRisk", "Info\nLeakage"]
    values = [costs['impact_cost'], costs['timing_cost'],
              costs['transaction_cost'], costs['adverse_selection_cost'],
              costs['inventory_risk_cost'], costs['information_leakage_cost']]
    colors = ['#1976D2', '#388E3C', '#F57C00', '#D32F2F', '#7B1FA2', '#00796B']

    bars = ax1.bar(range(len(labels)), values, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("Cost Contribution")
    ax1.set_title("HFT QUBO Cost Decomposition")

    total = sum(abs(v) for v in values)
    if total > 0:
        fracs = [abs(v)/total for v in values]
    else:
        fracs = [1/6]*6
    wedges, texts, autotexts = ax2.pie(
        fracs, labels=None, autopct='%1.0f%%',
        colors=colors, pctdistance=0.75, startangle=90,
        textprops={'fontsize': 7}
    )
    ax2.legend(labels, loc='center left', bbox_to_anchor=(0.9, 0.5), fontsize=6)
    ax2.set_title("Cost Distribution")

    fig.tight_layout()
    return save_fig(fig, "fig03_hft_qubo_cost_decomposition.pdf")


def fig04_qubo_to_ising_conversion():
    """Fig 4: QUBO to Ising conversion verification."""
    from qexec.optimization.qubo import QUBOConfig, ExecutionQUBO
    from qexec.optimization.ising import qubo_to_ising, binary_to_spins

    sizes = [4, 6, 8, 10, 12, 14, 16]
    errors = []
    num_terms = []

    for n in sizes:
        Q = np.random.default_rng(SEED).standard_normal((n, n))
        Q = (Q + Q.T) / 2
        ising = qubo_to_ising(Q)

        max_err = 0
        for _ in range(100):
            x = np.random.randint(0, 2, n)
            qubo_val = float(x @ Q @ x)
            z = binary_to_spins(x)
            ising_val = ising.evaluate(z)
            max_err = max(max_err, abs(qubo_val - ising_val))
        errors.append(max_err)

        nt = int(np.sum(np.abs(ising.h) > 1e-10))
        nt += int(np.sum(np.abs(ising.J) > 1e-10))
        num_terms.append(nt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    ax1.semilogy(sizes, [e + 1e-16 for e in errors], 'o-', color='#D32F2F',
                 label='Max |QUBO - Ising|')
    ax1.axhline(y=1e-10, color='green', linestyle='--', alpha=0.7, label='Tolerance')
    ax1.set_xlabel("Problem Size (qubits)")
    ax1.set_ylabel("Conversion Error")
    ax1.set_title("QUBO-Ising Equivalence")
    ax1.legend()

    ax2.bar(sizes, num_terms, color='#1565C0', alpha=0.7, edgecolor='black', linewidth=0.5)
    expected = [n + n*(n-1)//2 for n in sizes]
    ax2.plot(sizes, expected, 'r--', label='Max terms $n + n(n{-}1)/2$')
    ax2.set_xlabel("Problem Size (qubits)")
    ax2.set_ylabel("Hamiltonian Terms")
    ax2.set_title("Ising Term Scaling")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    return save_fig(fig, "fig04_qubo_ising_conversion.pdf")


# ═════════════════════════════════════════════════════════════════════
# SECTION 2: Classical vs Quantum Solver Performance (Figs 5-9)
# ═════════════════════════════════════════════════════════════════════

def fig05_sa_convergence():
    """Fig 5: Simulated annealing convergence curves."""
    from qexec.optimization.qubo import QUBOConfig, ExecutionQUBO
    from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver

    config = QUBOConfig(
        total_shares=400, num_time_slices=4, num_venues=1,
        quantity_levels=[0, 100, 200], equality_penalty=100.0
    )
    qubo = ExecutionQUBO(config)
    Q = qubo.build_qubo_matrix()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    sweep_counts = [100, 300, 500, 1000, 2000]
    colors_sweep = plt.cm.viridis(np.linspace(0.2, 0.9, len(sweep_counts)))
    final_energies = []

    for sweeps, color in zip(sweep_counts, colors_sweep):
        solver = SimulatedAnnealingSolver(num_sweeps=sweeps, seed=SEED)
        result = solver.solve(Q, verbose=False)
        if result.history:
            ax1.plot(result.history, color=color, label=f'{sweeps} sweeps',
                     alpha=0.8)
        final_energies.append(result.energy)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Best Energy")
    ax1.set_title("SA Convergence")
    ax1.legend(fontsize=6)

    ax2.plot(sweep_counts, final_energies, 'o-', color='#D32F2F', linewidth=1.5)
    ax2.set_xlabel("Number of Sweeps")
    ax2.set_ylabel("Final Energy")
    ax2.set_title("Energy vs Sweep Count")
    ax2.set_xscale('log')

    fig.tight_layout()
    return save_fig(fig, "fig05_sa_convergence.pdf")


def fig06_solver_comparison():
    """Fig 6: Brute-force vs SA vs Greedy comparison."""
    from qexec.optimization.qubo import QUBOConfig, ExecutionQUBO
    from qexec.optimization.solvers.exact import BruteForceSolver
    from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
    from qexec.optimization.solvers.greedy import GreedySolver

    sizes = [4, 6, 8, 10, 12]
    sa_ratios = []
    greedy_ratios = []
    sa_times = []
    bf_times = []
    greedy_times = []

    for n_vars in sizes:
        n_slices = max(2, n_vars // 3)
        levels = [0] + [100 * i for i in range(1, n_vars // n_slices + 1)]
        config = QUBOConfig(
            total_shares=400, num_time_slices=n_slices, num_venues=1,
            quantity_levels=levels, equality_penalty=100.0
        )
        qubo = ExecutionQUBO(config)
        Q = qubo.build_qubo_matrix()
        actual_n = Q.shape[0]

        if actual_n <= 20:
            bf = BruteForceSolver()
            bf_result = bf.solve(Q, verbose=False)
            bf_times.append(bf_result.solve_time)
            opt = bf_result.energy
        else:
            bf_times.append(None)
            sa_long = SimulatedAnnealingSolver(num_sweeps=5000, seed=SEED)
            opt = sa_long.solve(Q, verbose=False).energy

        sa = SimulatedAnnealingSolver(num_sweeps=500, seed=SEED)
        sa_result = sa.solve(Q, verbose=False)
        sa_times.append(sa_result.solve_time)
        sa_ratios.append(opt / sa_result.energy if abs(sa_result.energy) > 1e-10 else 1.0)

        gr = GreedySolver(seed=SEED)
        gr_result = gr.solve(Q, verbose=False)
        greedy_times.append(gr_result.solve_time)
        greedy_ratios.append(opt / gr_result.energy if abs(gr_result.energy) > 1e-10 else 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    ax1.plot(sizes, [min(r, 1.0) for r in sa_ratios], 's-', color='#1976D2',
             label='Sim. Annealing')
    ax1.plot(sizes, [min(r, 1.0) for r in greedy_ratios], '^-', color='#F57C00',
             label='Greedy')
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Optimal')
    ax1.set_xlabel("QUBO Variables")
    ax1.set_ylabel("Approximation Ratio")
    ax1.set_title("Solution Quality")
    ax1.set_ylim(0.5, 1.05)
    ax1.legend(fontsize=7)

    ax2.semilogy(sizes, sa_times, 's-', color='#1976D2', label='SA (500 sweeps)')
    ax2.semilogy(sizes, greedy_times, '^-', color='#F57C00', label='Greedy')
    bf_valid = [(s, t) for s, t in zip(sizes, bf_times) if t is not None]
    if bf_valid:
        ax2.semilogy([x[0] for x in bf_valid], [x[1] for x in bf_valid],
                     'o-', color='#D32F2F', label='Brute Force')
    ax2.set_xlabel("QUBO Variables")
    ax2.set_ylabel("Solve Time (s)")
    ax2.set_title("Computational Cost")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    return save_fig(fig, "fig06_solver_comparison.pdf")


def fig07_qaoa_vs_sa():
    """Fig 7: QAOA vs SA energy and time comparison."""
    from qexec.optimization.qubo import QUBOConfig, ExecutionQUBO
    from qexec.optimization.solvers.qaoa import QAOASolver
    from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver

    config = QUBOConfig(
        total_shares=400, num_time_slices=4, num_venues=1,
        quantity_levels=[0, 100, 200], equality_penalty=100.0
    )
    qubo = ExecutionQUBO(config)
    Q = qubo.build_qubo_matrix()

    num_runs = 5
    qaoa_energies, sa_energies = [], []
    qaoa_times, sa_times = [], []

    for i in range(num_runs):
        qaoa = QAOASolver(p=2, shots=500, maxiter=30, seed=SEED + i)
        qr = qaoa.solve(Q, verbose=False)
        qaoa_energies.append(qr.energy)
        qaoa_times.append(qr.solve_time)

        sa = SimulatedAnnealingSolver(num_sweeps=500, seed=SEED + i)
        t0 = time.time()
        sr = sa.solve(Q, verbose=False)
        sa_times.append(time.time() - t0)
        sa_energies.append(sr.energy)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 2.5))

    bp1 = ax1.boxplot([qaoa_energies, sa_energies], patch_artist=True, widths=0.5)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['QAOA\n(p=2)', 'SA\n(500sw)'])
    bp1['boxes'][0].set_facecolor('#4CAF50')
    bp1['boxes'][1].set_facecolor('#2196F3')
    for b in bp1['boxes']:
        b.set_alpha(0.6)
    ax1.set_ylabel("Energy")
    ax1.set_title("Energy Distribution")

    ax2.bar(['QAOA', 'SA'], [np.mean(qaoa_times), np.mean(sa_times)],
            color=['#4CAF50', '#2196F3'], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel("Time (s)")
    ax2.set_title("Mean Solve Time")

    ax3.scatter(qaoa_times, qaoa_energies, c='#4CAF50', marker='o', label='QAOA', alpha=0.7, s=30)
    ax3.scatter(sa_times, sa_energies, c='#2196F3', marker='s', label='SA', alpha=0.7, s=30)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Energy")
    ax3.set_title("Time-Energy Tradeoff")
    ax3.legend(fontsize=7)

    fig.tight_layout()
    return save_fig(fig, "fig07_qaoa_vs_sa.pdf")


def fig08_qaoa_landscape():
    """Fig 8: QAOA parameter landscape for p=1."""
    from qexec.optimization.qubo import QUBOConfig, ExecutionQUBO
    from qexec.optimization.ising import qubo_to_ising, build_qaoa_circuit_from_ising
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    config = QUBOConfig(
        total_shares=200, num_time_slices=2, num_venues=1,
        quantity_levels=[0, 100], equality_penalty=100.0
    )
    qubo = ExecutionQUBO(config)
    Q = qubo.build_qubo_matrix()
    n = Q.shape[0]
    ising = qubo_to_ising(Q)
    simulator = AerSimulator()

    gamma_range = np.linspace(0, 2 * np.pi, 20)
    beta_range = np.linspace(0, np.pi, 20)
    landscape = np.zeros((len(beta_range), len(gamma_range)))

    for gi, g in enumerate(gamma_range):
        for bi, b in enumerate(beta_range):
            qc = build_qaoa_circuit_from_ising(ising, [g], [b], 1)
            compiled = transpile(qc, simulator)
            result = simulator.run(compiled, shots=200).result()
            counts = result.get_counts()

            exp_e = 0.0
            for bs_str, count in counts.items():
                binary = np.array([int(c) for c in bs_str[::-1]])
                if len(binary) == n:
                    exp_e += float(binary @ Q @ binary) * count / 200
            landscape[bi, gi] = exp_e

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(landscape, extent=[0, 2*np.pi, 0, np.pi],
                   aspect='auto', cmap='RdYlBu_r', origin='lower')
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(r"QAOA Energy Landscape ($p=1$)")
    fig.colorbar(im, ax=ax, label="Expected Energy", shrink=0.9)

    min_idx = np.unravel_index(np.argmin(landscape), landscape.shape)
    ax.plot(gamma_range[min_idx[1]], beta_range[min_idx[0]],
            'w*', markersize=10, markeredgecolor='black', markeredgewidth=0.5)

    fig.tight_layout()
    return save_fig(fig, "fig08_qaoa_landscape.pdf")


def fig09_solution_quality_vs_size():
    """Fig 9: How solution quality scales with problem size."""
    from qexec.optimization.qubo import QUBOConfig, ExecutionQUBO
    from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
    from qexec.optimization.solvers.exact import BruteForceSolver

    sizes = [4, 6, 8, 10, 12, 16, 20]
    sa_gaps = []
    fill_rates = []

    for n in sizes:
        n_slices = max(2, n // 2)
        levels = list(range(0, n // n_slices + 1))
        levels = [l * 50 for l in levels]
        config = QUBOConfig(
            total_shares=500, num_time_slices=n_slices, num_venues=1,
            quantity_levels=levels, equality_penalty=100.0
        )
        qubo = ExecutionQUBO(config)
        Q = qubo.build_qubo_matrix()
        actual_n = Q.shape[0]

        if actual_n <= 20:
            bf = BruteForceSolver()
            opt_e = bf.solve(Q, verbose=False).energy
        else:
            long_sa = SimulatedAnnealingSolver(num_sweeps=5000, seed=SEED)
            opt_e = long_sa.solve(Q, verbose=False).energy

        gaps = []
        fills = []
        for run in range(5):
            sa = SimulatedAnnealingSolver(num_sweeps=500, seed=SEED + run)
            sr = sa.solve(Q, verbose=False)
            gap = (sr.energy - opt_e) / abs(opt_e) * 100 if abs(opt_e) > 1e-10 else 0
            gaps.append(gap)

            schedule = qubo.interpret_solution(sr.solution)
            total = sum(row['quantity'] for _, row in schedule.iterrows()) if len(schedule) > 0 else 0
            fills.append(total / 500)

        sa_gaps.append((np.mean(gaps), np.std(gaps)))
        fill_rates.append((np.mean(fills), np.std(fills)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    means = [g[0] for g in sa_gaps]
    stds = [g[1] for g in sa_gaps]
    ax1.errorbar(sizes, means, yerr=stds, fmt='o-', color='#D32F2F',
                 capsize=3, label='SA optimality gap')
    ax1.set_xlabel("QUBO Variables")
    ax1.set_ylabel("Gap from Optimal (%)")
    ax1.set_title("Optimality Gap Scaling")
    ax1.legend()

    means_f = [f[0] for f in fill_rates]
    stds_f = [f[1] for f in fill_rates]
    ax2.errorbar(sizes, means_f, yerr=stds_f, fmt='s-', color='#1976D2',
                 capsize=3, label='Fill rate')
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    ax2.set_xlabel("QUBO Variables")
    ax2.set_ylabel("Fill Rate")
    ax2.set_title("Constraint Satisfaction")
    ax2.set_ylim(0, 1.2)
    ax2.legend()

    fig.tight_layout()
    return save_fig(fig, "fig09_solution_quality_scaling.pdf")


# ═════════════════════════════════════════════════════════════════════
# SECTION 3: Market Microstructure (Figs 10-14)
# ═════════════════════════════════════════════════════════════════════

def _generate_micro_data(n_ticks=600):
    """Generate synthetic tick data with regime changes."""
    np.random.seed(SEED)
    price = 100.0
    prices, bids, asks, volumes, sides = [], [], [], [], []
    for i in range(n_ticks):
        if i < 200:
            vol = 0.001
        elif i < 400:
            vol = 0.005
        else:
            vol = 0.0015
        ret = np.random.normal(0, vol)
        price *= (1 + ret)
        spread = max(0.01, abs(ret) * price * 2 + 0.01)
        prices.append(price)
        bids.append(price - spread/2)
        asks.append(price + spread/2)
        volumes.append(max(10, int(np.random.exponential(500))))
        sides.append("buy" if ret >= 0 else "sell")
    return prices, bids, asks, volumes, sides


def fig10_kyle_lambda():
    """Fig 10: Kyle's lambda estimation over time with regime changes."""
    from qexec.microstructure.kyle import KyleLambdaEstimator

    prices, bids, asks, volumes, sides = _generate_micro_data()
    estimator = KyleLambdaEstimator(window_size=50)
    lambdas = []

    for i in range(1, len(prices)):
        dp = prices[i] - prices[i-1]
        sv = volumes[i] if sides[i] == "buy" else -volumes[i]
        estimator.update(dp, sv)
        lambdas.append(estimator.lambda_value)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4), sharex=True)

    ax1.plot(prices, color='#1565C0', linewidth=0.8)
    ax1.axvspan(200, 400, alpha=0.15, color='red', label='High Vol')
    ax1.set_ylabel("Price ($)")
    ax1.set_title("Kyle's Lambda Estimation")
    ax1.legend(fontsize=7)

    ax2.plot(range(1, len(lambdas)+1), lambdas, color='#D32F2F', linewidth=0.8)
    ax2.axvspan(200, 400, alpha=0.15, color='red')
    ax2.set_xlabel("Tick")
    ax2.set_ylabel(r"Kyle's $\lambda$")
    ax2.set_title("Price Impact Coefficient")

    fig.tight_layout()
    return save_fig(fig, "fig10_kyle_lambda.pdf")


def fig11_vpin_estimation():
    """Fig 11: VPIN estimation with toxicity threshold."""
    from qexec.microstructure.vpin import VPINEstimator

    prices, bids, asks, volumes, sides = _generate_micro_data()
    estimator = VPINEstimator(bucket_size=200, num_buckets=20)
    vpins = []

    for i in range(len(prices)):
        prev_p = prices[i - 1] if i > 0 else prices[i]
        v = estimator.update(prices[i], volumes[i], prev_p)
        vpins.append(v)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(vpins, color='#7B1FA2', linewidth=0.8, label='VPIN')
    ax.axhline(y=0.7, color='red', linestyle='--', linewidth=0.8, label='Toxic threshold')
    ax.axvspan(200, 400, alpha=0.15, color='red', label='High vol regime')
    ax.fill_between(range(len(vpins)), vpins, 0.7,
                     where=[v > 0.7 for v in vpins], color='red', alpha=0.2)
    ax.set_xlabel("Tick")
    ax.set_ylabel("VPIN")
    ax.set_title("Volume-Synchronized Probability\nof Informed Trading")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)

    fig.tight_layout()
    return save_fig(fig, "fig11_vpin_estimation.pdf")


def fig12_microstructure_dashboard():
    """Fig 12: Combined microstructure metrics dashboard."""
    from qexec.microstructure.analyzer import MicrostructureAnalyzer

    prices, bids, asks, volumes, sides = _generate_micro_data()
    analyzer = MicrostructureAnalyzer(kyle_window=50, vpin_bucket_size=200,
                                      vpin_buckets=20, tick_size=0.01)

    states = []
    for i in range(len(prices)):
        state = analyzer.process_tick(prices[i], volumes[i], bids[i], asks[i],
                                       sides[i], timestamp_ns=i*100000)
        states.append(state)

    fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharex=True)

    axes[0, 0].plot([s.kyle_lambda for s in states], color='#D32F2F', linewidth=0.8)
    axes[0, 0].set_ylabel(r"Kyle's $\lambda$")
    axes[0, 0].set_title("Price Impact")

    axes[0, 1].plot([s.vpin for s in states], color='#7B1FA2', linewidth=0.8)
    axes[0, 1].axhline(y=0.7, color='red', linestyle='--', linewidth=0.6)
    axes[0, 1].set_ylabel("VPIN")
    axes[0, 1].set_title("Order Flow Toxicity")

    axes[1, 0].plot([s.adverse_selection_cost for s in states],
                     color='#F57C00', linewidth=0.8)
    axes[1, 0].set_ylabel("AS Cost")
    axes[1, 0].set_title("Adverse Selection")
    axes[1, 0].set_xlabel("Tick")

    spreads = [(a - b) / ((a + b)/2) * 10000 for a, b in zip(asks, bids)]
    axes[1, 1].plot(spreads, color='#00796B', linewidth=0.8)
    axes[1, 1].set_ylabel("Spread (bps)")
    axes[1, 1].set_title("Bid-Ask Spread")
    axes[1, 1].set_xlabel("Tick")

    for ax_row in axes:
        for ax in ax_row:
            ax.axvspan(200, 400, alpha=0.1, color='red')

    fig.suptitle("Market Microstructure Dashboard", fontsize=11, fontweight='bold')
    fig.tight_layout()
    return save_fig(fig, "fig12_microstructure_dashboard.pdf")


# ═════════════════════════════════════════════════════════════════════
# SECTION 4: Adaptive Risk Aversion (Figs 15-18)
# ═════════════════════════════════════════════════════════════════════

def fig15_lambda_adaptation():
    """Fig 15: Dynamic lambda adaptation across volatility regimes."""
    from qexec.microstructure.regime import AdaptiveRiskManager

    prices, bids, asks, volumes, _ = _generate_micro_data()
    manager = AdaptiveRiskManager(base_lambda=1.0)
    lambdas = []
    regimes = []

    for i in range(len(prices)):
        state = manager.update(prices[i], bids[i], asks[i], volumes[i],
                                float(np.mean(volumes[:max(1, i)])))
        lambdas.append(state.lambda_value)
        regimes.append(state.vol_regime.value)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4), sharex=True)

    ax1.plot(prices, color='#1565C0', linewidth=0.8)
    ax1.set_ylabel("Price ($)")
    ax1.set_title("Adaptive Risk Aversion (Lambda)")
    ax1.axvspan(200, 400, alpha=0.15, color='red', label='High vol')
    ax1.legend(fontsize=7)

    ax2.plot(lambdas, color='#D32F2F', linewidth=0.8)
    regime_colors = {'low': '#4CAF50', 'normal': '#2196F3',
                      'high': '#FF9800', 'extreme': '#D32F2F'}
    for i in range(1, len(regimes)):
        ax2.axvspan(i-1, i, alpha=0.15, color=regime_colors.get(regimes[i], '#999'))

    ax2.set_xlabel("Tick")
    ax2.set_ylabel(r"$\lambda$ (Risk Aversion)")

    patches = [mpatches.Patch(color=c, alpha=0.3, label=r)
               for r, c in regime_colors.items()]
    ax2.legend(handles=patches, fontsize=6, ncol=2)

    fig.tight_layout()
    return save_fig(fig, "fig15_lambda_adaptation.pdf")


def fig16_volatility_regime_detection():
    """Fig 16: Dual-timescale EWMA volatility regime detection."""
    from qexec.microstructure.regime import VolatilityEstimator

    prices, _, _, _, _ = _generate_micro_data()
    estimator = VolatilityEstimator(fast_alpha=0.06, slow_alpha=0.01)
    fast_vols, slow_vols, ratios = [], [], []

    for p in prices:
        f, s = estimator.update(p)
        fast_vols.append(f)
        slow_vols.append(s)
        ratios.append(estimator.vol_ratio)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4), sharex=True)

    ax1.plot(fast_vols, color='#D32F2F', linewidth=0.8, label=r'Fast EWMA ($\alpha$=0.06)')
    ax1.plot(slow_vols, color='#1976D2', linewidth=0.8, label=r'Slow EWMA ($\alpha$=0.01)')
    ax1.set_ylabel("Volatility")
    ax1.set_title("Dual-Timescale EWMA Volatility")
    ax1.legend(fontsize=6)
    ax1.axvspan(200, 400, alpha=0.15, color='red')

    ax2.plot(ratios, color='#7B1FA2', linewidth=0.8)
    ax2.axhline(y=0.7, color='green', linestyle='--', linewidth=0.6, alpha=0.7)
    ax2.axhline(y=1.3, color='#FF9800', linestyle='--', linewidth=0.6, alpha=0.7)
    ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=0.6, alpha=0.7)
    ax2.text(10, 0.5, 'LOW', fontsize=6, color='green')
    ax2.text(10, 1.0, 'NORMAL', fontsize=6, color='gray')
    ax2.text(10, 1.5, 'HIGH', fontsize=6, color='#FF9800')
    ax2.text(10, 2.2, 'EXTREME', fontsize=6, color='red')
    ax2.set_xlabel("Tick")
    ax2.set_ylabel("Fast/Slow Ratio")
    ax2.set_title("Regime Detection Signal")
    ax2.axvspan(200, 400, alpha=0.15, color='red')

    fig.tight_layout()
    return save_fig(fig, "fig16_volatility_regime_detection.pdf")


def fig17_regime_distribution():
    """Fig 17: Regime distribution and regime-conditioned lambda."""
    from qexec.microstructure.regime import AdaptiveRiskManager

    prices, bids, asks, volumes, _ = _generate_micro_data(n_ticks=1000)
    manager = AdaptiveRiskManager(base_lambda=1.0)
    regime_lambdas = {'low': [], 'normal': [], 'high': [], 'extreme': []}

    for i in range(len(prices)):
        state = manager.update(prices[i], bids[i], asks[i], volumes[i],
                                float(np.mean(volumes[:max(1, i)])))
        regime_lambdas[state.vol_regime.value].append(state.lambda_value)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    regime_names = ['Low', 'Normal', 'High', 'Extreme']
    counts = [len(regime_lambdas[r.lower()]) for r in regime_names]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#D32F2F']
    ax1.bar(regime_names, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel("Tick Count")
    ax1.set_title("Regime Distribution")

    box_data = [regime_lambdas[r.lower()] for r in regime_names if regime_lambdas[r.lower()]]
    box_labels_valid = [r for r in regime_names if regime_lambdas[r.lower()]]
    box_colors_valid = [c for r, c in zip(regime_names, colors) if regime_lambdas[r.lower()]]

    if box_data:
        bp = ax2.boxplot(box_data, patch_artist=True, widths=0.5)
        ax2.set_xticks(range(1, len(box_labels_valid) + 1))
        ax2.set_xticklabels(box_labels_valid)
        for patch, color in zip(bp['boxes'], box_colors_valid):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax2.set_ylabel(r"$\lambda$ Value")
    ax2.set_title(r"Regime-Conditioned $\lambda$")

    fig.tight_layout()
    return save_fig(fig, "fig17_regime_distribution.pdf")


def fig18_qubo_param_sensitivity():
    """Fig 18: Sensitivity of execution cost to QUBO parameters."""
    from qexec.optimization.hft_qubo import HFTQUBOConfig, HFTExecutionQUBO
    from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver

    lambdas = np.linspace(0.2, 5.0, 15)
    vpins = np.linspace(0.0, 0.9, 15)

    cost_vs_lambda = []
    for lam in lambdas:
        config = HFTQUBOConfig(
            total_shares=1000, num_tick_slices=5, num_venues=2,
            quantity_levels=[0, 100, 250, 500],
            impact_weight=0.25 * lam, vpin=0.3
        )
        qubo = HFTExecutionQUBO(config)
        Q = qubo.build_qubo_matrix()
        sa = SimulatedAnnealingSolver(num_sweeps=300, seed=SEED)
        r = sa.solve(Q, verbose=False)
        cost_vs_lambda.append(r.energy)

    cost_vs_vpin = []
    for vp in vpins:
        config = HFTQUBOConfig(
            total_shares=1000, num_tick_slices=5, num_venues=2,
            quantity_levels=[0, 100, 250, 500], vpin=vp
        )
        qubo = HFTExecutionQUBO(config)
        Q = qubo.build_qubo_matrix()
        sa = SimulatedAnnealingSolver(num_sweeps=300, seed=SEED)
        r = sa.solve(Q, verbose=False)
        cost_vs_vpin.append(r.energy)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    ax1.plot(lambdas, cost_vs_lambda, 'o-', color='#D32F2F', markersize=3)
    ax1.set_xlabel(r"Risk Aversion $\lambda$")
    ax1.set_ylabel("Optimal Cost")
    ax1.set_title(r"Cost Sensitivity to $\lambda$")

    ax2.plot(vpins, cost_vs_vpin, 's-', color='#7B1FA2', markersize=3)
    ax2.set_xlabel("VPIN")
    ax2.set_ylabel("Optimal Cost")
    ax2.set_title("Cost Sensitivity to VPIN")

    fig.tight_layout()
    return save_fig(fig, "fig18_qubo_param_sensitivity.pdf")


# ═════════════════════════════════════════════════════════════════════
# SECTION 5: Execution Performance & Walk-Forward (Figs 19-23)
# ═════════════════════════════════════════════════════════════════════

def fig19_execution_schedule_comparison():
    """Fig 19: TWAP vs VWAP vs Almgren-Chriss vs QUBO execution schedules."""
    from qexec.execution.strategies.almgren_chriss import ACConfig, AlmgrenChrissSolver
    from qexec.optimization.qubo import QUBOConfig, ExecutionQUBO
    from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver

    total = 10000
    n_steps = 15

    twap = np.full(n_steps, total / n_steps)

    u_profile = np.array([1.5 - 0.8 * abs(i - n_steps/2) / (n_steps/2) for i in range(n_steps)])
    u_profile = u_profile / u_profile.sum()
    vwap = u_profile * total

    ac_config = ACConfig(total_shares=total, n_steps=n_steps, risk_aversion=1e-4)
    ac_solver = AlmgrenChrissSolver(ac_config)
    ac_sched = ac_solver.compute_trajectory()
    ac = ac_sched['shares_to_trade'].values

    qubo_config = QUBOConfig(
        total_shares=total, num_time_slices=n_steps, num_venues=1,
        quantity_levels=[0, total//(n_steps*2), total//n_steps],
        equality_penalty=100.0, impact_coefficient=0.1
    )
    qubo_obj = ExecutionQUBO(qubo_config)
    Q = qubo_obj.build_qubo_matrix()
    sa = SimulatedAnnealingSolver(num_sweeps=500, seed=SEED)
    r = sa.solve(Q, verbose=False)
    sol_df = qubo_obj.interpret_solution(r.solution)
    qubo_sched = np.zeros(n_steps)
    if len(sol_df) > 0:
        for _, row in sol_df.iterrows():
            t = int(row['time_slice'])
            if t < n_steps:
                qubo_sched[t] += row['quantity']
    qubo_sched = qubo_sched * (total / max(1, qubo_sched.sum()))

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = range(n_steps)
    ax.step(x, twap, where='mid', label='TWAP', color='#9E9E9E', linewidth=1.0)
    ax.step(x, vwap, where='mid', label='VWAP', color='#2196F3', linewidth=1.0)
    ax.step(x, ac, where='mid', label='Almgren-Chriss', color='#4CAF50', linewidth=1.0)
    ax.step(x, qubo_sched, where='mid', label='QUBO-Optimized',
            color='#D32F2F', linewidth=1.5)
    ax.set_xlabel("Time Slice")
    ax.set_ylabel("Shares")
    ax.set_title("Execution Schedule Comparison")
    ax.legend(fontsize=7)

    fig.tight_layout()
    return save_fig(fig, "fig19_execution_schedule_comparison.pdf")


def fig20_almgren_chriss_frontier():
    """Fig 20: Almgren-Chriss efficient frontier (cost vs risk)."""
    from qexec.execution.strategies.almgren_chriss import ACConfig, AlmgrenChrissSolver

    risk_aversions = np.logspace(-10, -3, 20)
    costs, variances = [], []

    for lam in risk_aversions:
        config = ACConfig(total_shares=10000, n_steps=15, risk_aversion=lam)
        solver = AlmgrenChrissSolver(config)
        traj = solver.compute_trajectory()
        c = solver.calculate_expected_cost(traj)
        v = solver.calculate_variance(traj)
        costs.append(c)
        variances.append(v)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    sc = ax.scatter(np.sqrt(variances), costs, c=np.log10(risk_aversions),
                     cmap='RdYlBu_r', s=30, edgecolors='black', linewidths=0.5)
    ax.plot(np.sqrt(variances), costs, 'k-', alpha=0.3, linewidth=0.5)
    ax.set_xlabel(r"Risk (Std Dev of Cost, $\sqrt{V[C]}$)")
    ax.set_ylabel("Expected Cost E[C]")
    ax.set_title("Almgren-Chriss Efficient Frontier")
    cbar = fig.colorbar(sc, ax=ax, label=r"$\log_{10}(\lambda)$", shrink=0.9)
    ax.annotate('Risk Neutral\n(TWAP)', xy=(np.sqrt(variances[0]), costs[0]),
                fontsize=6, ha='left')
    ax.annotate('Risk Averse\n(Front-loaded)', xy=(np.sqrt(variances[-1]), costs[-1]),
                fontsize=6, ha='right')

    fig.tight_layout()
    return save_fig(fig, "fig20_almgren_chriss_frontier.pdf")


def fig21_walk_forward_shortfall():
    """Fig 21: Walk-forward cumulative implementation shortfall."""
    from qexec.analysis.walk_forward import WalkForwardAnalyzer

    analyzer = WalkForwardAnalyzer(total_days=8, train_days=3, test_days=1,
                                    daily_shares=10000)
    results = analyzer.run()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    windows = [r.window_id for r in results]
    static = [r.shortfall_static for r in results]
    adaptive = [r.shortfall_adaptive for r in results]
    hybrid = [r.shortfall_hybrid for r in results]

    cum_s = np.cumsum(static)
    cum_a = np.cumsum(adaptive)
    cum_h = np.cumsum(hybrid)

    ax1.plot(windows, cum_s, '--', color='#9E9E9E', label='Static VWAP')
    ax1.plot(windows, cum_a, '-', color='#2196F3', label='Adaptive VWAP')
    ax1.plot(windows, cum_h, '-', color='#D32F2F', linewidth=1.5, label='Hybrid QUBO')
    ax1.set_xlabel("Rolling Window")
    ax1.set_ylabel("Cumulative IS ($)")
    ax1.set_title("Walk-Forward Analysis")
    ax1.legend(fontsize=7)

    x = np.arange(len(windows))
    w = 0.25
    ax2.bar(x - w, static, w, label='Static', color='#9E9E9E', alpha=0.7)
    ax2.bar(x, adaptive, w, label='Adaptive', color='#2196F3', alpha=0.7)
    ax2.bar(x + w, hybrid, w, label='Hybrid', color='#D32F2F', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'W{i}' for i in windows])
    ax2.set_xlabel("Window")
    ax2.set_ylabel("IS ($)")
    ax2.set_title("Per-Window Shortfall")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    return save_fig(fig, "fig21_walk_forward_shortfall.pdf")


# ═════════════════════════════════════════════════════════════════════
# SECTION 6: Latency & HFT Pipeline (Figs 24-27)
# ═════════════════════════════════════════════════════════════════════

def fig27_venue_routing():
    """Fig 27: Multi-venue routing decisions from HFT QUBO."""
    from qexec.optimization.hft_qubo import HFTQUBOConfig, HFTExecutionQUBO
    from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver

    config = HFTQUBOConfig(
        total_shares=3000, num_tick_slices=15, num_venues=3,
        quantity_levels=[0, 100, 250, 500],
        kyle_lambda=0.001, vpin=0.4
    )
    qubo = HFTExecutionQUBO(config)
    Q = qubo.build_qubo_matrix()
    sa = SimulatedAnnealingSolver(num_sweeps=500, seed=SEED)
    result = sa.solve(Q, verbose=False)
    solution = qubo.interpret_solution(result.solution)

    venue_ticks = {v: np.zeros(15) for v in ['Lit', 'Dark', 'ECN']}
    for entry in solution['schedule']:
        t = entry['tick']
        v = entry['venue']
        if t < 15 and v in venue_ticks:
            venue_ticks[v][t] += entry['quantity']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    x = np.arange(15)
    w = 0.8
    bottom = np.zeros(15)
    colors_v = {'Lit': '#D32F2F', 'Dark': '#4CAF50', 'ECN': '#2196F3'}
    for venue, data in venue_ticks.items():
        ax1.bar(x, data, w, bottom=bottom, label=venue,
                color=colors_v[venue], alpha=0.7, edgecolor='black', linewidth=0.3)
        bottom += data

    ax1.set_xlabel("Tick Slice")
    ax1.set_ylabel("Shares")
    ax1.set_title("Venue Routing Schedule")
    ax1.legend(fontsize=7)

    venue_totals = {v: float(d.sum()) for v, d in venue_ticks.items()}
    total = sum(venue_totals.values())
    if total > 0:
        fracs = [venue_totals[v] / total for v in ['Lit', 'Dark', 'ECN']]
    else:
        fracs = [0.33, 0.33, 0.34]
    ax2.pie(fracs, labels=['Lit', 'Dark', 'ECN'], autopct='%1.0f%%',
            colors=[colors_v['Lit'], colors_v['Dark'], colors_v['ECN']],
            pctdistance=0.7, textprops={'fontsize': 8})
    ax2.set_title("Venue Volume Split")

    fig.tight_layout()
    return save_fig(fig, "fig27_venue_routing.pdf")


# ═════════════════════════════════════════════════════════════════════
# SECTION 7: IBM Hardware / Quantum-Specific (Figs 28-30)
# ═════════════════════════════════════════════════════════════════════

def fig28_qaoa_circuit_depth():
    """Fig 28: QAOA circuit depth and gate count scaling."""
    from qexec.optimization.ising import qubo_to_ising, build_qaoa_circuit_from_ising

    sizes = [4, 6, 8, 10, 12]
    depths_p1, depths_p2, depths_p3 = [], [], []
    gates_p1, gates_p2, gates_p3 = [], [], []

    for n in sizes:
        Q = np.random.default_rng(SEED).standard_normal((n, n))
        Q = (Q + Q.T) / 2
        ising = qubo_to_ising(Q)

        for p, d_list, g_list in [(1, depths_p1, gates_p1),
                                    (2, depths_p2, gates_p2),
                                    (3, depths_p3, gates_p3)]:
            g = np.ones(p) * 0.5
            b = np.ones(p) * 0.3
            qc = build_qaoa_circuit_from_ising(ising, g, b, p)
            d_list.append(qc.depth())
            g_list.append(sum(qc.count_ops().values()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    ax1.plot(sizes, depths_p1, 'o-', color='#4CAF50', label='p=1')
    ax1.plot(sizes, depths_p2, 's-', color='#2196F3', label='p=2')
    ax1.plot(sizes, depths_p3, '^-', color='#D32F2F', label='p=3')
    ax1.set_xlabel("Number of Qubits")
    ax1.set_ylabel("Circuit Depth")
    ax1.set_title("QAOA Circuit Depth")
    ax1.legend()

    ax2.plot(sizes, gates_p1, 'o-', color='#4CAF50', label='p=1')
    ax2.plot(sizes, gates_p2, 's-', color='#2196F3', label='p=2')
    ax2.plot(sizes, gates_p3, '^-', color='#D32F2F', label='p=3')
    ax2.set_xlabel("Number of Qubits")
    ax2.set_ylabel("Total Gate Count")
    ax2.set_title("QAOA Gate Count")
    ax2.legend()

    fig.tight_layout()
    return save_fig(fig, "fig28_qaoa_circuit_depth.pdf")


# ═════════════════════════════════════════════════════════════════════
# Main: Generate All Figures
# ═════════════════════════════════════════════════════════════════════

ALL_FIGURES = [
    ("Section 1: Architecture & QUBO", [
        fig01_system_architecture,
        fig02_qubo_matrix_structure,
        fig03_hft_qubo_cost_decomposition,
        fig04_qubo_to_ising_conversion,
    ]),
    ("Section 2: Solver Performance", [
        fig05_sa_convergence,
        fig06_solver_comparison,
        fig07_qaoa_vs_sa,
        fig08_qaoa_landscape,
        fig09_solution_quality_vs_size,
    ]),
    ("Section 3: Microstructure", [
        fig10_kyle_lambda,
        fig11_vpin_estimation,
        fig12_microstructure_dashboard,
    ]),
    ("Section 4: Adaptive Risk", [
        fig15_lambda_adaptation,
        fig16_volatility_regime_detection,
        fig17_regime_distribution,
        fig18_qubo_param_sensitivity,
    ]),
    ("Section 5: Execution Performance", [
        fig19_execution_schedule_comparison,
        fig20_almgren_chriss_frontier,
        fig21_walk_forward_shortfall,
    ]),
    ("Section 6: Latency & HFT", [
        fig27_venue_routing,
    ]),
    ("Section 7: Quantum Hardware", [
        fig28_qaoa_circuit_depth,
    ]),
]


def generate_all_figures():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_paths = []
    total_start = time.time()

    print("=" * 70)
    num_figures = sum(len(funcs) for _, funcs in ALL_FIGURES)
    print(f" Generating {num_figures} Journal Figures (Springer Format)")
    print("=" * 70)

    for section_name, funcs in ALL_FIGURES:
        print(f"\n {section_name}")
        print(" " + "─" * 50)
        for func in funcs:
            t0 = time.time()
            try:
                path = func()
                elapsed = time.time() - t0
                print(f"   {func.__name__:<45} {elapsed:5.1f}s  OK")
                all_paths.append(path)
            except Exception as e:
                elapsed = time.time() - t0
                print(f"   {func.__name__:<45} {elapsed:5.1f}s  FAIL: {e}")

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f" Generated {len(all_paths)}/{num_figures} figures in {total_time:.1f}s")
    print(f" Output directory: {OUTPUT_DIR}/")
    print(f"{'=' * 70}")

    return all_paths


if __name__ == "__main__":
    generate_all_figures()
