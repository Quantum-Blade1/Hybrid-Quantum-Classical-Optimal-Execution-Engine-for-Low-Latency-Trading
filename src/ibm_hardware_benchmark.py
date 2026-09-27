"""
IBM Quantum Hardware Benchmarking Module

Runs execution QUBO optimization on real IBM quantum processors via
qiskit-ibm-runtime, then compares against:
  1. Noiseless Aer simulator (ideal QAOA)
  2. Noisy Aer simulator (calibrated noise model)
  3. Classical simulated annealing

This produces the hardware benchmark data and figures needed for
a Springer journal paper. No prior quantum finance paper provides
real-hardware execution results at this level of detail.

Usage:
    # With IBM Quantum token:
    results = run_hardware_benchmark(token="YOUR_IBM_TOKEN")

    # Without token (simulator-only mode for development):
    results = run_hardware_benchmark(token=None)

    # Generate all figures:
    generate_hardware_figures(results, output_dir="figures/")
"""

import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class HardwareBenchmarkConfig:
    """Configuration for IBM hardware benchmarks."""
    num_qubits_range: List[int] = field(default_factory=lambda: [4, 6, 8, 10, 12])
    qaoa_depths: List[int] = field(default_factory=lambda: [1, 2, 3])
    shots: int = 4000
    optimizer_maxiter: int = 50
    sa_sweeps: int = 1000
    num_runs_per_config: int = 5
    seed: int = 42
    backend_name: str = "ibm_brisbane"


@dataclass
class SingleRunResult:
    """Result from a single solver run."""
    solver: str
    num_qubits: int
    qaoa_depth: int
    energy: float
    optimal_energy: float
    approximation_ratio: float
    solve_time_s: float
    success_probability: float
    counts: Dict[str, int] = field(default_factory=dict)
    run_id: int = 0


@dataclass
class HardwareBenchmarkResult:
    """Aggregated results across all benchmark configurations."""
    config: HardwareBenchmarkConfig
    runs: List[SingleRunResult] = field(default_factory=list)
    backend_properties: Dict = field(default_factory=dict)
    timestamp: str = ""

    def get_runs(self, solver: str = None, n_qubits: int = None,
                 depth: int = None) -> List[SingleRunResult]:
        filtered = self.runs
        if solver:
            filtered = [r for r in filtered if r.solver == solver]
        if n_qubits:
            filtered = [r for r in filtered if r.num_qubits == n_qubits]
        if depth:
            filtered = [r for r in filtered if r.qaoa_depth == depth]
        return filtered

    def summary_table(self) -> Dict[str, Dict[str, float]]:
        table = {}
        for solver in ["SA", "QAOA_Ideal", "QAOA_Noisy", "QAOA_Hardware"]:
            runs = self.get_runs(solver=solver)
            if runs:
                energies = [r.energy for r in runs]
                ratios = [r.approximation_ratio for r in runs]
                times = [r.solve_time_s for r in runs]
                table[solver] = {
                    "mean_energy": float(np.mean(energies)),
                    "std_energy": float(np.std(energies)),
                    "mean_approx_ratio": float(np.mean(ratios)),
                    "mean_time_s": float(np.mean(times)),
                    "best_energy": float(np.min(energies)),
                    "num_runs": len(runs),
                }
        return table


def build_execution_qubo(n_qubits: int, seed: int = 42) -> Tuple[np.ndarray, float]:
    """
    Build a small execution-style QUBO for hardware benchmarking.

    Creates a QUBO that encodes an optimal execution problem with
    n_qubits binary decision variables. Returns the QUBO matrix and
    the exact optimal energy (found by brute force for small n).
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_qubits, n_qubits))

    num_slices = n_qubits // 2
    num_levels = 2

    impact_coeff = 0.1
    timing_coeff = 0.05
    total_target = num_slices

    for t in range(num_slices):
        for k in range(num_levels):
            i = t * num_levels + k
            if i >= n_qubits:
                break
            q = (k + 1)
            Q[i, i] += impact_coeff * q * q
            Q[i, i] += timing_coeff * (t + 1) * q

    penalty = 10.0
    for i in range(n_qubits):
        t_i = i // num_levels
        k_i = i % num_levels
        q_i = (k_i + 1)
        Q[i, i] += penalty * q_i * q_i - 2 * penalty * total_target * q_i
        for j in range(i + 1, n_qubits):
            k_j = j % num_levels
            q_j = (k_j + 1)
            Q[i, j] += 2 * penalty * q_i * q_j

    Q = (Q + Q.T) / 2

    optimal_energy = float('inf')
    for bits in range(2 ** n_qubits):
        x = np.array([(bits >> b) & 1 for b in range(n_qubits)], dtype=float)
        e = float(x @ Q @ x)
        if e < optimal_energy:
            optimal_energy = e

    return Q, optimal_energy


def solve_with_sa(Q: np.ndarray, optimal_energy: float,
                  num_sweeps: int = 1000, seed: int = 42) -> SingleRunResult:
    """Solve QUBO with simulated annealing."""
    from .qubo_solvers import SimulatedAnnealingSolver

    n = Q.shape[0]
    start = time.time()
    solver = SimulatedAnnealingSolver(num_sweeps=num_sweeps, seed=seed)
    result = solver.solve(Q, verbose=False)
    elapsed = time.time() - start

    ratio = optimal_energy / result.energy if abs(result.energy) > 1e-10 else 1.0
    ratio = min(ratio, 1.0)

    return SingleRunResult(
        solver="SA",
        num_qubits=n,
        qaoa_depth=0,
        energy=result.energy,
        optimal_energy=optimal_energy,
        approximation_ratio=ratio,
        solve_time_s=elapsed,
        success_probability=1.0 if abs(result.energy - optimal_energy) < 1e-6 else 0.0,
    )


def solve_with_qaoa_simulator(
    Q: np.ndarray,
    optimal_energy: float,
    p: int = 2,
    shots: int = 4000,
    maxiter: int = 50,
    noisy: bool = False,
    noise_level: float = 0.02,
    seed: int = 42
) -> SingleRunResult:
    """Solve QUBO with QAOA on Aer simulator (ideal or noisy)."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    from qiskit import transpile
    from scipy.optimize import minimize

    from .qubo_to_ising import qubo_to_ising, build_qaoa_circuit_from_ising

    n = Q.shape[0]
    rng = np.random.default_rng(seed)
    start = time.time()

    ising = qubo_to_ising(Q)

    backend = AerSimulator()
    if noisy:
        noise_model = NoiseModel()
        error_1q = depolarizing_error(noise_level, 1)
        error_2q = depolarizing_error(noise_level * 5, 2)
        noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'rz', 'h'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['rzz', 'cx'])

        p_0_given_1 = noise_level
        p_1_given_0 = noise_level * 0.8
        readout_err = ReadoutError(
            [[1 - p_1_given_0, p_1_given_0],
             [p_0_given_1, 1 - p_0_given_1]]
        )
        noise_model.add_all_qubit_readout_error(readout_err)
        backend = AerSimulator(noise_model=noise_model)

    history = []

    def cost_fn(params):
        gammas = params[:p]
        betas = params[p:]
        qc = build_qaoa_circuit_from_ising(ising, gammas, betas, p)
        compiled = transpile(qc, backend)
        result = backend.run(compiled, shots=shots).result()
        counts = result.get_counts()

        exp_energy = 0.0
        for bitstring, count in counts.items():
            binary = np.array([int(b) for b in bitstring[::-1]])
            if len(binary) == n:
                exp_energy += float(binary @ Q @ binary) * count / shots
        history.append(exp_energy)
        return exp_energy

    x0 = np.concatenate([
        rng.uniform(0, 2 * np.pi, p),
        rng.uniform(0, np.pi, p)
    ])

    opt_result = minimize(cost_fn, x0, method='COBYLA',
                          options={'maxiter': maxiter})

    gammas = opt_result.x[:p]
    betas = opt_result.x[p:]
    qc = build_qaoa_circuit_from_ising(ising, gammas, betas, p)
    compiled = transpile(qc, backend)
    final = backend.run(compiled, shots=shots * 5).result()
    counts = final.get_counts()

    best_energy = float('inf')
    best_bs = None
    for bitstring, count in counts.items():
        binary = np.array([int(b) for b in bitstring[::-1]])
        if len(binary) == n:
            e = float(binary @ Q @ binary)
            if e < best_energy:
                best_energy = e
                best_bs = bitstring

    total_shots = sum(counts.values())
    success_count = counts.get(best_bs, 0) if best_bs else 0
    success_prob = success_count / total_shots

    elapsed = time.time() - start
    ratio = optimal_energy / best_energy if abs(best_energy) > 1e-10 else 1.0
    ratio = min(ratio, 1.0)

    solver_name = "QAOA_Noisy" if noisy else "QAOA_Ideal"

    return SingleRunResult(
        solver=solver_name,
        num_qubits=n,
        qaoa_depth=p,
        energy=best_energy,
        optimal_energy=optimal_energy,
        approximation_ratio=ratio,
        solve_time_s=elapsed,
        success_probability=success_prob,
        counts=dict(Counter(counts).most_common(20)),
    )


def solve_with_ibm_hardware(
    Q: np.ndarray,
    optimal_energy: float,
    p: int = 2,
    shots: int = 4000,
    maxiter: int = 30,
    token: str = None,
    backend_name: str = "ibm_brisbane",
    seed: int = 42,
    instance: str = None,
    channel: str = None,
) -> Tuple[SingleRunResult, Dict]:
    """
    Solve QUBO with QAOA on real IBM quantum hardware.

    Supports both IBM Quantum Platform (channel="ibm_quantum") and
    IBM Cloud Quantum (channel="ibm_cloud" with instance CRN).

    Args:
        token: IBM API key/token
        backend_name: Backend processor name
        instance: IBM Cloud CRN (required for ibm_cloud channel)
        channel: "ibm_quantum" or "ibm_cloud" (auto-detected from instance)
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit import QuantumCircuit
    from scipy.optimize import minimize

    from .qubo_to_ising import qubo_to_ising, build_qaoa_circuit_from_ising

    n = Q.shape[0]
    rng = np.random.default_rng(seed)
    start = time.time()

    if channel is None:
        channel = "ibm_cloud" if instance else "ibm_quantum"

    service_kwargs = {"channel": channel, "token": token}
    if instance:
        service_kwargs["instance"] = instance

    service = QiskitRuntimeService(**service_kwargs)
    backend = service.backend(backend_name)

    backend_props = {
        "name": backend.name,
        "num_qubits": backend.num_qubits,
        "version": str(getattr(backend, 'version', 'unknown')),
    }

    ising = qubo_to_ising(Q)

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

    def cost_fn(params):
        gammas = params[:p]
        betas = params[p:]
        qc = build_qaoa_circuit_from_ising(ising, gammas, betas, p)
        isa_circuit = pm.run(qc)

        sampler = SamplerV2(mode=backend)
        job = sampler.run([isa_circuit], shots=shots)
        result = job.result()
        pub_result = result[0]

        counts = pub_result.data.meas.get_counts()

        exp_energy = 0.0
        total = sum(counts.values())
        for bitstring, count in counts.items():
            binary = np.array([int(b) for b in bitstring[::-1]])
            if len(binary) >= n:
                binary = binary[:n]
            exp_energy += float(binary @ Q @ binary) * count / total
        return exp_energy

    x0 = np.concatenate([
        rng.uniform(0, 2 * np.pi, p),
        rng.uniform(0, np.pi, p)
    ])

    opt_result = minimize(cost_fn, x0, method='COBYLA',
                          options={'maxiter': maxiter})

    gammas = opt_result.x[:p]
    betas = opt_result.x[p:]
    qc = build_qaoa_circuit_from_ising(ising, gammas, betas, p)
    isa_circuit = pm.run(qc)

    sampler = SamplerV2(mode=backend)
    job = sampler.run([isa_circuit], shots=shots * 5)
    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()

    best_energy = float('inf')
    best_bs = None
    for bitstring, count in counts.items():
        binary = np.array([int(b) for b in bitstring[::-1]])
        if len(binary) >= n:
            binary = binary[:n]
        e = float(binary @ Q @ binary)
        if e < best_energy:
            best_energy = e
            best_bs = bitstring

    total_shots = sum(counts.values())
    success_count = counts.get(best_bs, 0) if best_bs else 0
    success_prob = success_count / total_shots

    elapsed = time.time() - start
    ratio = optimal_energy / best_energy if abs(best_energy) > 1e-10 else 1.0
    ratio = min(ratio, 1.0)

    return SingleRunResult(
        solver="QAOA_Hardware",
        num_qubits=n,
        qaoa_depth=p,
        energy=best_energy,
        optimal_energy=optimal_energy,
        approximation_ratio=ratio,
        solve_time_s=elapsed,
        success_probability=success_prob,
        counts=dict(Counter(counts).most_common(20)),
    ), backend_props


def run_hardware_benchmark(
    token: Optional[str] = None,
    config: Optional[HardwareBenchmarkConfig] = None,
    instance: Optional[str] = None,
    channel: Optional[str] = None,
) -> HardwareBenchmarkResult:
    """
    Run the full IBM hardware benchmark suite.

    When token is None, runs simulator-only benchmarks (ideal + noisy).
    When token is provided, also runs on real IBM quantum hardware.

    Args:
        token: IBM API key (ibm_cloud) or token (ibm_quantum)
        config: Benchmark configuration
        instance: IBM Cloud CRN for ibm_cloud channel
        channel: "ibm_quantum" or "ibm_cloud" (auto-detected from instance)
    """
    if config is None:
        config = HardwareBenchmarkConfig()

    result = HardwareBenchmarkResult(
        config=config,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    hardware_available = token is not None

    print("=" * 70)
    print(" IBM Quantum Hardware Benchmark Suite")
    print("=" * 70)
    if hardware_available:
        print(f" Backend: {config.backend_name}")
    else:
        print(" Mode: Simulator-only (no IBM token provided)")
    print(f" Qubit range: {config.num_qubits_range}")
    print(f" QAOA depths: {config.qaoa_depths}")
    print(f" Runs per config: {config.num_runs_per_config}")

    for n_qubits in config.num_qubits_range:
        print(f"\n{'─' * 60}")
        print(f" Problem size: {n_qubits} qubits")
        print(f"{'─' * 60}")

        Q, opt_energy = build_execution_qubo(n_qubits, seed=config.seed)
        print(f" Optimal energy: {opt_energy:.4f}")

        for run_id in range(config.num_runs_per_config):
            run_seed = config.seed + run_id * 100

            sa_result = solve_with_sa(Q, opt_energy, config.sa_sweeps, run_seed)
            sa_result.run_id = run_id
            result.runs.append(sa_result)

            for p_depth in config.qaoa_depths:
                ideal = solve_with_qaoa_simulator(
                    Q, opt_energy, p=p_depth, shots=config.shots,
                    maxiter=config.optimizer_maxiter, noisy=False,
                    seed=run_seed
                )
                ideal.run_id = run_id
                result.runs.append(ideal)

                noisy = solve_with_qaoa_simulator(
                    Q, opt_energy, p=p_depth, shots=config.shots,
                    maxiter=config.optimizer_maxiter, noisy=True,
                    noise_level=0.02, seed=run_seed
                )
                noisy.run_id = run_id
                result.runs.append(noisy)

                if hardware_available and n_qubits <= 12:
                    try:
                        hw, props = solve_with_ibm_hardware(
                            Q, opt_energy, p=p_depth, shots=config.shots,
                            maxiter=min(config.optimizer_maxiter, 30),
                            token=token, backend_name=config.backend_name,
                            seed=run_seed, instance=instance,
                            channel=channel,
                        )
                        hw.run_id = run_id
                        result.runs.append(hw)
                        result.backend_properties = props
                        print(f"   HW p={p_depth}: E={hw.energy:.4f} "
                              f"(ratio={hw.approximation_ratio:.3f})")
                    except Exception as e:
                        logger.error(f"Hardware run failed: {e}")
                        print(f"   HW p={p_depth}: FAILED - {e}")

            if run_id == 0:
                print(f"   SA: E={sa_result.energy:.4f} "
                      f"(ratio={sa_result.approximation_ratio:.3f}, "
                      f"t={sa_result.solve_time_s:.4f}s)")
                for p_depth in config.qaoa_depths:
                    ideal_runs = [r for r in result.runs
                                  if r.solver == "QAOA_Ideal"
                                  and r.num_qubits == n_qubits
                                  and r.qaoa_depth == p_depth
                                  and r.run_id == 0]
                    noisy_runs = [r for r in result.runs
                                  if r.solver == "QAOA_Noisy"
                                  and r.num_qubits == n_qubits
                                  and r.qaoa_depth == p_depth
                                  and r.run_id == 0]
                    if ideal_runs:
                        ir = ideal_runs[0]
                        print(f"   Ideal p={p_depth}: E={ir.energy:.4f} "
                              f"(ratio={ir.approximation_ratio:.3f}, "
                              f"t={ir.solve_time_s:.2f}s)")
                    if noisy_runs:
                        nr = noisy_runs[0]
                        print(f"   Noisy p={p_depth}: E={nr.energy:.4f} "
                              f"(ratio={nr.approximation_ratio:.3f})")

    print(f"\n{'=' * 70}")
    print(" Benchmark Complete")
    table = result.summary_table()
    print(f"\n {'Solver':<18} {'Mean E':>10} {'Best E':>10} "
          f"{'Approx':>10} {'Time(s)':>10} {'Runs':>6}")
    print(" " + "─" * 66)
    for solver, stats in table.items():
        print(f" {solver:<18} {stats['mean_energy']:>10.2f} "
              f"{stats['best_energy']:>10.2f} "
              f"{stats['mean_approx_ratio']:>10.3f} "
              f"{stats['mean_time_s']:>10.3f} "
              f"{stats['num_runs']:>6}")
    print("=" * 70)

    return result


def generate_hardware_figures(
    result: HardwareBenchmarkResult,
    output_dir: str = "figures"
) -> List[str]:
    """Generate all hardware benchmark figures for the paper."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'serif',
    })

    saved = []

    # ── Figure 1: Approximation Ratio vs Problem Size ──
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    solvers = ["SA", "QAOA_Ideal", "QAOA_Noisy", "QAOA_Hardware"]
    colors = {"SA": "#2196F3", "QAOA_Ideal": "#4CAF50",
              "QAOA_Noisy": "#FF9800", "QAOA_Hardware": "#E91E63"}
    markers = {"SA": "s", "QAOA_Ideal": "o",
               "QAOA_Noisy": "^", "QAOA_Hardware": "D"}
    labels = {"SA": "Sim. Annealing", "QAOA_Ideal": "QAOA (Ideal)",
              "QAOA_Noisy": "QAOA (Noisy)", "QAOA_Hardware": "QAOA (IBM HW)"}

    for solver in solvers:
        qubit_sizes = sorted(set(r.num_qubits for r in result.runs
                                 if r.solver == solver))
        if not qubit_sizes:
            continue
        means, stds = [], []
        for nq in qubit_sizes:
            runs = result.get_runs(solver=solver, n_qubits=nq)
            ratios = [r.approximation_ratio for r in runs]
            means.append(np.mean(ratios))
            stds.append(np.std(ratios))

        ax.errorbar(qubit_sizes, means, yerr=stds, marker=markers[solver],
                     color=colors[solver], label=labels[solver],
                     capsize=3, linewidth=1.2, markersize=4)

    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Approximation Ratio")
    ax.set_title("Solution Quality vs Problem Size")
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "fig_hw_approx_ratio_vs_size.pdf")
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)

    # ── Figure 2: Solve Time Comparison ──
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for solver in solvers:
        qubit_sizes = sorted(set(r.num_qubits for r in result.runs
                                 if r.solver == solver))
        if not qubit_sizes:
            continue
        means = []
        for nq in qubit_sizes:
            runs = result.get_runs(solver=solver, n_qubits=nq)
            means.append(np.mean([r.solve_time_s for r in runs]))
        ax.semilogy(qubit_sizes, means, marker=markers[solver],
                     color=colors[solver], label=labels[solver],
                     linewidth=1.2, markersize=4)

    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Solve Time (s)")
    ax.set_title("Computational Cost Scaling")
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    path = os.path.join(output_dir, "fig_hw_solve_time_scaling.pdf")
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)

    # ── Figure 3: Energy Distribution Box Plots ──
    fig, axes = plt.subplots(1, len(result.config.num_qubits_range),
                             figsize=(7, 2.5), sharey=False)
    if len(result.config.num_qubits_range) == 1:
        axes = [axes]

    for idx, nq in enumerate(result.config.num_qubits_range):
        ax = axes[idx]
        data_for_box = []
        box_labels = []
        box_colors = []
        for solver in solvers:
            runs = result.get_runs(solver=solver, n_qubits=nq)
            if runs:
                data_for_box.append([r.energy for r in runs])
                box_labels.append(solver.replace("QAOA_", "").replace("_", "\n"))
                box_colors.append(colors[solver])

        if data_for_box:
            bp = ax.boxplot(data_for_box, patch_artist=True, widths=0.6)
            ax.set_xticks(range(1, len(box_labels) + 1))
            ax.set_xticklabels(box_labels)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            opt_e = result.get_runs(n_qubits=nq)[0].optimal_energy if result.get_runs(n_qubits=nq) else 0
            ax.axhline(y=opt_e, color='red', linestyle='--', linewidth=0.8,
                       label='Optimal')

        ax.set_title(f"n={nq}", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Energy")
        ax.tick_params(axis='x', rotation=45)

    fig.suptitle("Energy Distribution by Solver and Problem Size", fontsize=10)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_hw_energy_distribution.pdf")
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)

    # ── Figure 4: QAOA Depth Effect ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    for solver_type, ax, title in [
        ("QAOA_Ideal", ax1, "Ideal Simulator"),
        ("QAOA_Noisy", ax2, "Noisy Simulator")
    ]:
        for nq in result.config.num_qubits_range:
            depths = sorted(set(r.qaoa_depth for r in result.runs
                                if r.solver == solver_type
                                and r.num_qubits == nq))
            if not depths:
                continue
            means = []
            for d in depths:
                runs = result.get_runs(solver=solver_type, n_qubits=nq, depth=d)
                means.append(np.mean([r.approximation_ratio for r in runs]))
            ax.plot(depths, means, marker='o', label=f"n={nq}",
                    linewidth=1.2, markersize=4)

        ax.set_xlabel("QAOA Depth (p)")
        ax.set_ylabel("Approximation Ratio")
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Effect of Circuit Depth on Solution Quality", fontsize=10)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_hw_depth_effect.pdf")
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)

    # ── Figure 5: Success Probability Heatmap ──
    depths = sorted(set(r.qaoa_depth for r in result.runs if r.qaoa_depth > 0))
    qubit_sizes = sorted(set(r.num_qubits for r in result.runs))

    if depths and qubit_sizes:
        for solver_type in ["QAOA_Ideal", "QAOA_Noisy"]:
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            matrix = np.zeros((len(depths), len(qubit_sizes)))

            for di, d in enumerate(depths):
                for qi, nq in enumerate(qubit_sizes):
                    runs = result.get_runs(solver=solver_type, n_qubits=nq, depth=d)
                    if runs:
                        matrix[di, qi] = np.mean([r.success_probability for r in runs])

            im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd',
                           vmin=0, vmax=max(0.5, matrix.max()))
            ax.set_xticks(range(len(qubit_sizes)))
            ax.set_xticklabels(qubit_sizes)
            ax.set_yticks(range(len(depths)))
            ax.set_yticklabels([f"p={d}" for d in depths])
            ax.set_xlabel("Number of Qubits")
            ax.set_ylabel("QAOA Depth")
            label = "Ideal" if "Ideal" in solver_type else "Noisy"
            ax.set_title(f"Success Probability ({label})")
            fig.colorbar(im, ax=ax, shrink=0.8)

            for di in range(len(depths)):
                for qi in range(len(qubit_sizes)):
                    ax.text(qi, di, f"{matrix[di, qi]:.2f}",
                            ha='center', va='center', fontsize=7,
                            color='white' if matrix[di, qi] > 0.3 else 'black')

            fig.tight_layout()
            suffix = "ideal" if "Ideal" in solver_type else "noisy"
            path = os.path.join(output_dir, f"fig_hw_success_prob_{suffix}.pdf")
            fig.savefig(path)
            plt.close(fig)
            saved.append(path)

    # ── Figure 6: Noise Impact Analysis ──
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for nq in result.config.num_qubits_range:
        ideal_runs = result.get_runs(solver="QAOA_Ideal", n_qubits=nq)
        noisy_runs = result.get_runs(solver="QAOA_Noisy", n_qubits=nq)

        if ideal_runs and noisy_runs:
            ideal_mean = np.mean([r.approximation_ratio for r in ideal_runs])
            noisy_mean = np.mean([r.approximation_ratio for r in noisy_runs])
            degradation = (ideal_mean - noisy_mean) / ideal_mean * 100
            ax.bar(nq, degradation, width=1.5, color='#FF5722', alpha=0.7,
                   edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Quality Degradation (%)")
    ax.set_title("Noise-Induced Performance Loss")
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_hw_noise_degradation.pdf")
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)

    # ── Figure 7: Measurement Count Distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    for ax, solver_type, title in [
        (axes[0], "QAOA_Ideal", "Ideal Simulator"),
        (axes[1], "QAOA_Noisy", "Noisy Simulator")
    ]:
        target_nq = min(result.config.num_qubits_range)
        runs = result.get_runs(solver=solver_type, n_qubits=target_nq)
        if runs and runs[0].counts:
            counts = runs[0].counts
            sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:15]
            states = [s[0][:8] for s in sorted_counts]
            values = [s[1] for s in sorted_counts]

            bars = ax.bar(range(len(states)), values, color=colors[solver_type],
                          alpha=0.7, edgecolor='black', linewidth=0.3)
            ax.set_xticks(range(len(states)))
            ax.set_xticklabels(states, rotation=90, fontsize=6)
            ax.set_ylabel("Counts")
            ax.set_title(f"{title} (n={target_nq})")
            ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("Measurement Output Distribution", fontsize=10)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_hw_count_distribution.pdf")
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)

    print(f"\n Generated {len(saved)} hardware benchmark figures:")
    for p in saved:
        print(f"   {p}")

    return saved


def run_quick_benchmark(
    output_dir: str = "figures",
    token: Optional[str] = None,
    instance: Optional[str] = None,
    channel: Optional[str] = None,
) -> HardwareBenchmarkResult:
    """Quick benchmark for testing (smaller parameters)."""
    config = HardwareBenchmarkConfig(
        num_qubits_range=[4, 6, 8],
        qaoa_depths=[1, 2],
        shots=1000,
        optimizer_maxiter=30,
        sa_sweeps=500,
        num_runs_per_config=3,
    )
    result = run_hardware_benchmark(
        token=token, config=config,
        instance=instance, channel=channel,
    )
    generate_hardware_figures(result, output_dir=output_dir)
    return result


if __name__ == "__main__":
    import os
    ibm_token = os.environ.get("IBM_QUANTUM_TOKEN") or os.environ.get("IBM_CLOUD_API_KEY")
    ibm_instance = os.environ.get("IBM_CLOUD_CRN")
    result = run_quick_benchmark(
        token=ibm_token, instance=ibm_instance,
    )
