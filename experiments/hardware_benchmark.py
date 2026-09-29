"""SA vs QAOA (ideal Aer, noisy Aer, optionally IBM hardware) on the toy execution QUBO.

All solvers see the same toy_execution_qubo(n). Quality is the approximation ratio
(E_max - E)/(E_max - E_opt) and the relative optimality gap, with exact bounds from
exhaustive enumeration. Hardware jobs go through qexec.hardware.ibm, which appends every
job's ID and raw counts to --counts-log as soon as the job returns.

Usage:
    python experiments/hardware_benchmark.py --simulator-only --output-dir /tmp/hw_figs
    IBM_QUANTUM_TOKEN=... python experiments/hardware_benchmark.py
    IBM_CLOUD_API_KEY=... IBM_CLOUD_CRN=... python experiments/hardware_benchmark.py
"""

import argparse
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from qexec.hardware.ibm import backend_properties, connect_service, run_qaoa_on_hardware
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
from qexec.optimization.solvers.metrics import (
    EnergyBounds,
    approximation_ratio,
    energy_bounds,
    optimality_gap,
)
from qexec.optimization.solvers.qaoa import QAOAResult, aer_sampler, run_qaoa

logger = logging.getLogger(__name__)

DEFAULT_COUNTS_LOG = Path("results/hw_jobs.jsonl")
SOLVERS = ("SA", "QAOA_Ideal", "QAOA_Noisy", "QAOA_Hardware")
COLORS = {
    "SA": "#2196F3",
    "QAOA_Ideal": "#4CAF50",
    "QAOA_Noisy": "#FF9800",
    "QAOA_Hardware": "#E91E63",
}
MARKERS = {"SA": "s", "QAOA_Ideal": "o", "QAOA_Noisy": "^", "QAOA_Hardware": "D"}
LABELS = {
    "SA": "Sim. Annealing",
    "QAOA_Ideal": "QAOA (Ideal)",
    "QAOA_Noisy": "QAOA (Noisy)",
    "QAOA_Hardware": "QAOA (IBM HW)",
}
OPTIMUM_TOL = 1e-6
MAX_HARDWARE_QUBITS = 12
MAX_HARDWARE_MAXITER = 30
PLOT_RC = {
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
}


@dataclass(frozen=True)
class HardwareBenchmarkConfig:
    num_qubits_range: list[int] = field(default_factory=lambda: [4, 6, 8, 10, 12])
    qaoa_depths: list[int] = field(default_factory=lambda: [1, 2, 3])
    shots: int = 4000
    optimizer_maxiter: int = 50
    sa_sweeps: int = 1000
    num_runs_per_config: int = 5
    seed: int = 42
    noise_level: float = 0.02
    backend_name: str = "ibm_brisbane"


@dataclass
class SingleRunResult:
    solver: str
    num_qubits: int
    qaoa_depth: int
    energy: float
    optimal_energy: float
    approximation_ratio: float
    optimality_gap: float
    solve_time_s: float
    success_probability: float
    counts: dict[str, int] = field(default_factory=dict)
    run_id: int = 0


@dataclass
class HardwareBenchmarkResult:
    config: HardwareBenchmarkConfig
    runs: list[SingleRunResult] = field(default_factory=list)
    backend_properties: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def get_runs(
        self, solver: str | None = None, n_qubits: int | None = None, depth: int | None = None
    ) -> list[SingleRunResult]:
        return [
            r
            for r in self.runs
            if (solver is None or r.solver == solver)
            and (n_qubits is None or r.num_qubits == n_qubits)
            and (depth is None or r.qaoa_depth == depth)
        ]

    def summary_table(self) -> dict[str, dict[str, float]]:
        table = {}
        for solver in SOLVERS:
            runs = self.get_runs(solver=solver)
            if runs:
                energies = [r.energy for r in runs]
                table[solver] = {
                    "mean_energy": float(np.mean(energies)),
                    "best_energy": float(np.min(energies)),
                    "mean_approx_ratio": float(np.mean([r.approximation_ratio for r in runs])),
                    "mean_gap": float(np.mean([r.optimality_gap for r in runs])),
                    "mean_time_s": float(np.mean([r.solve_time_s for r in runs])),
                    "num_runs": len(runs),
                }
        return table


def toy_execution_qubo(n_qubits: int) -> NDArray[np.float64]:
    """Toy QUBO of the ibm_fez runs in results/bench_hw_*.json (not the six-term HFT QUBO).

    n/2 slices x 2 levels q in {1, 2}: impact 0.1 q^2 + timing 0.05 (t+1) q on the diagonal
    and the equality penalty 10 (sum q_i x_i - n/2)^2.
    """
    Q = np.zeros((n_qubits, n_qubits))
    num_levels = 2
    num_slices = n_qubits // num_levels
    impact_coeff = 0.1
    timing_coeff = 0.05
    penalty = 10.0

    for t in range(num_slices):
        for k in range(num_levels):
            i = t * num_levels + k
            q = k + 1
            Q[i, i] += impact_coeff * q * q
            Q[i, i] += timing_coeff * (t + 1) * q

    for i in range(n_qubits):
        q_i = i % num_levels + 1
        Q[i, i] += penalty * q_i * q_i - 2 * penalty * num_slices * q_i
        for j in range(i + 1, n_qubits):
            q_j = j % num_levels + 1
            Q[i, j] += 2 * penalty * q_i * q_j

    return (Q + Q.T) / 2


def _run_result(
    solver: str,
    Q: NDArray[np.float64],
    bounds: EnergyBounds,
    energy: float,
    elapsed: float,
    *,
    depth: int = 0,
    success_probability: float,
    run_id: int,
    counts: dict[str, int] | None = None,
) -> SingleRunResult:
    return SingleRunResult(
        solver=solver,
        num_qubits=Q.shape[0],
        qaoa_depth=depth,
        energy=energy,
        optimal_energy=bounds.min_energy,
        approximation_ratio=approximation_ratio(energy, bounds),
        optimality_gap=optimality_gap(energy, bounds),
        solve_time_s=elapsed,
        success_probability=success_probability,
        counts=counts or {},
        run_id=run_id,
    )


def solve_with_sa(
    Q: NDArray[np.float64], bounds: EnergyBounds, num_sweeps: int, seed: int, run_id: int
) -> SingleRunResult:
    start = time.perf_counter()
    result = SimulatedAnnealingSolver(num_sweeps=num_sweeps, seed=seed).solve(Q)
    hit = abs(result.energy - bounds.min_energy) < OPTIMUM_TOL
    return _run_result(
        "SA",
        Q,
        bounds,
        result.energy,
        time.perf_counter() - start,
        success_probability=1.0 if hit else 0.0,
        run_id=run_id,
    )


def noisy_aer_backend(noise_level: float = 0.02) -> AerSimulator:
    """Depolarizing gate noise (p on 1q gates, 5p on 2q gates) and readout errors
    p(0|1) = p, p(1|0) = 0.8p."""
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(noise_level, 1), ["rx", "rz", "h"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(noise_level * 5, 2), ["rzz", "cx"])
    p_0_given_1 = noise_level
    p_1_given_0 = noise_level * 0.8
    noise_model.add_all_qubit_readout_error(
        ReadoutError([[1 - p_1_given_0, p_1_given_0], [p_0_given_1, 1 - p_0_given_1]])
    )
    return AerSimulator(noise_model=noise_model)


def _qaoa_run_result(
    solver: str,
    Q: NDArray[np.float64],
    bounds: EnergyBounds,
    result: QAOAResult,
    *,
    elapsed: float,
    depth: int,
    run_id: int,
) -> SingleRunResult:
    return _run_result(
        solver,
        Q,
        bounds,
        result.energy,
        elapsed,
        depth=depth,
        success_probability=result.success_probability,
        run_id=run_id,
        counts=dict(Counter(result.counts).most_common(20)),
    )


def solve_with_qaoa_simulator(
    Q: NDArray[np.float64],
    bounds: EnergyBounds,
    config: HardwareBenchmarkConfig,
    *,
    p: int,
    noisy: bool,
    seed: int,
    run_id: int,
) -> SingleRunResult:
    start = time.perf_counter()
    backend = noisy_aer_backend(config.noise_level) if noisy else AerSimulator()
    result = run_qaoa(
        Q,
        p=p,
        sample=aer_sampler(backend, seed=seed),
        shots=config.shots,
        maxiter=config.optimizer_maxiter,
        final_shots=config.shots * 5,
        rng=np.random.default_rng(seed),
    )
    solver = "QAOA_Noisy" if noisy else "QAOA_Ideal"
    elapsed = time.perf_counter() - start
    return _qaoa_run_result(solver, Q, bounds, result, elapsed=elapsed, depth=p, run_id=run_id)


def solve_with_ibm_hardware(
    Q: NDArray[np.float64],
    bounds: EnergyBounds,
    config: HardwareBenchmarkConfig,
    *,
    p: int,
    token: str,
    instance: str | None,
    counts_log: Path,
    seed: int,
    run_id: int,
) -> tuple[SingleRunResult, dict[str, Any]]:
    """QAOA on IBM hardware; every job's ID and raw counts are appended to `counts_log`."""
    start = time.perf_counter()
    service = connect_service(token=token, instance=instance)
    backend = service.backend(config.backend_name)
    props = backend_properties(backend)
    result, job_ids = run_qaoa_on_hardware(
        Q,
        backend,
        p=p,
        shots=config.shots,
        maxiter=min(config.optimizer_maxiter, MAX_HARDWARE_MAXITER),
        seed=seed,
        log_path=counts_log,
        metadata={
            "experiment": "hardware_benchmark",
            "run_id": run_id,
            "seed": seed,
            "optimal_energy": bounds.min_energy,
        },
    )
    props["job_ids"] = job_ids
    elapsed = time.perf_counter() - start
    hw = _qaoa_run_result(
        "QAOA_Hardware", Q, bounds, result, elapsed=elapsed, depth=p, run_id=run_id
    )
    return hw, props


def run_hardware_benchmark(
    config: HardwareBenchmarkConfig,
    token: str | None = None,
    instance: str | None = None,
    counts_log: Path = DEFAULT_COUNTS_LOG,
) -> HardwareBenchmarkResult:
    """Simulator runs always; IBM hardware runs too when `token` is given."""
    result = HardwareBenchmarkResult(config=config, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    mode = f"backend {config.backend_name}" if token else "simulator only"
    print(
        f"Hardware benchmark ({mode}): qubits {config.num_qubits_range}, "
        f"depths {config.qaoa_depths}, {config.num_runs_per_config} runs per configuration"
    )

    for n_qubits in config.num_qubits_range:
        Q = toy_execution_qubo(n_qubits)
        bounds = energy_bounds(Q)
        print(f"\nn = {n_qubits}: optimum {bounds.min_energy:.4f}, worst {bounds.max_energy:.4f}")
        for run_id in range(config.num_runs_per_config):
            run_seed = config.seed + run_id * 100
            result.runs.append(solve_with_sa(Q, bounds, config.sa_sweeps, run_seed, run_id))
            for p in config.qaoa_depths:
                for noisy in (False, True):
                    result.runs.append(
                        solve_with_qaoa_simulator(
                            Q, bounds, config, p=p, noisy=noisy, seed=run_seed, run_id=run_id
                        )
                    )
                if token and n_qubits <= MAX_HARDWARE_QUBITS:
                    # A hardware job can fail for many external reasons (queue, quota,
                    # network); record it and continue with the remaining runs.
                    try:
                        hw, props = solve_with_ibm_hardware(
                            Q,
                            bounds,
                            config,
                            p=p,
                            token=token,
                            instance=instance,
                            counts_log=counts_log,
                            seed=run_seed,
                            run_id=run_id,
                        )
                    except Exception:
                        logger.exception("Hardware run failed (n=%d, p=%d)", n_qubits, p)
                        continue
                    result.runs.append(hw)
                    result.backend_properties = props
        for run in result.get_runs(n_qubits=n_qubits):
            if run.run_id == 0:
                print(
                    f"  {run.solver:<14} p={run.qaoa_depth}  E={run.energy:10.4f}  "
                    f"ratio={run.approximation_ratio:.4f}  gap={run.optimality_gap:.2%}  "
                    f"t={run.solve_time_s:.3f}s"
                )

    columns = ("Mean E", "Best E", "Ratio", "Gap", "Time(s)", "Runs")
    widths = (10, 10, 9, 9, 10, 6)
    print(
        "\n"
        + f"{'Solver':<16}"
        + "".join(f"{c:>{w}}" for c, w in zip(columns, widths, strict=True))
    )
    for solver, s in result.summary_table().items():
        print(
            f"{solver:<16}{s['mean_energy']:>10.2f}{s['best_energy']:>10.2f}"
            f"{s['mean_approx_ratio']:>9.3f}{s['mean_gap']:>9.2%}"
            f"{s['mean_time_s']:>10.3f}{s['num_runs']:>6}"
        )
    return result


def _solvers_present(result: HardwareBenchmarkResult) -> list[str]:
    return [s for s in SOLVERS if result.get_runs(solver=s)]


def _fig_ratio_vs_size(result: HardwareBenchmarkResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for solver in _solvers_present(result):
        sizes = sorted({r.num_qubits for r in result.get_runs(solver=solver)})
        ratios = [[r.approximation_ratio for r in result.get_runs(solver, n)] for n in sizes]
        ax.errorbar(
            sizes,
            [np.mean(r) for r in ratios],
            yerr=[np.std(r) for r in ratios],
            marker=MARKERS[solver],
            color=COLORS[solver],
            label=LABELS[solver],
            capsize=3,
            linewidth=1.2,
            markersize=4,
        )
    ax.set(xlabel="Number of Qubits", ylabel="Approximation Ratio", ylim=(0, 1.1))
    ax.set_title("Solution Quality vs Problem Size")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.savefig(path)
    plt.close(fig)


def _fig_time_scaling(result: HardwareBenchmarkResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for solver in _solvers_present(result):
        sizes = sorted({r.num_qubits for r in result.get_runs(solver=solver)})
        times = [np.mean([r.solve_time_s for r in result.get_runs(solver, n)]) for n in sizes]
        ax.semilogy(
            sizes,
            times,
            marker=MARKERS[solver],
            color=COLORS[solver],
            label=LABELS[solver],
            linewidth=1.2,
            markersize=4,
        )
    ax.set(xlabel="Number of Qubits", ylabel="Solve Time (s)", title="Computational Cost Scaling")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    fig.savefig(path)
    plt.close(fig)


def _fig_energy_distribution(result: HardwareBenchmarkResult, path: Path) -> None:
    sizes = result.config.num_qubits_range
    fig, axes = plt.subplots(1, len(sizes), figsize=(7, 2.5), squeeze=False)
    for ax, n in zip(axes[0], sizes, strict=True):
        solvers = [s for s in SOLVERS if result.get_runs(s, n)]
        if solvers:
            bp = ax.boxplot(
                [[r.energy for r in result.get_runs(s, n)] for s in solvers],
                patch_artist=True,
                widths=0.6,
            )
            ax.set_xticks(range(1, len(solvers) + 1))
            ax.set_xticklabels([s.replace("QAOA_", "") for s in solvers])
            for patch, solver in zip(bp["boxes"], solvers, strict=True):
                patch.set_facecolor(COLORS[solver])
                patch.set_alpha(0.6)
            ax.axhline(
                result.get_runs(n_qubits=n)[0].optimal_energy,
                color="red",
                linestyle="--",
                linewidth=0.8,
            )
        ax.set_title(f"n={n}", fontsize=9)
        ax.tick_params(axis="x", rotation=45)
    axes[0][0].set_ylabel("Energy")
    fig.suptitle("Energy Distribution by Solver and Problem Size", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_depth_effect(result: HardwareBenchmarkResult, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for ax, solver, title in [
        (axes[0], "QAOA_Ideal", "Ideal Simulator"),
        (axes[1], "QAOA_Noisy", "Noisy Simulator"),
    ]:
        for n in result.config.num_qubits_range:
            depths = sorted({r.qaoa_depth for r in result.get_runs(solver, n)})
            if depths:
                ratios = [
                    np.mean([r.approximation_ratio for r in result.get_runs(solver, n, d)])
                    for d in depths
                ]
                ax.plot(depths, ratios, marker="o", label=f"n={n}", linewidth=1.2, markersize=4)
        ax.set(xlabel="QAOA Depth (p)", ylabel="Approximation Ratio", title=title, ylim=(0, 1.1))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Effect of Circuit Depth on Solution Quality", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_success_heatmap(result: HardwareBenchmarkResult, solver: str, path: Path) -> None:
    depths = sorted({r.qaoa_depth for r in result.runs if r.qaoa_depth > 0})
    sizes = sorted({r.num_qubits for r in result.runs})
    matrix = np.zeros((len(depths), len(sizes)))
    for di, d in enumerate(depths):
        for qi, n in enumerate(sizes):
            runs = result.get_runs(solver, n, d)
            if runs:
                matrix[di, qi] = np.mean([r.success_probability for r in runs])
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(0.5, matrix.max()))
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([f"p={d}" for d in depths])
    ax.set(xlabel="Number of Qubits", ylabel="QAOA Depth")
    ax.set_title(f"Success Probability ({solver.removeprefix('QAOA_')})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    for di in range(len(depths)):
        for qi in range(len(sizes)):
            value = matrix[di, qi]
            color = "white" if value > 0.3 else "black"
            ax.text(qi, di, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_noise_degradation(result: HardwareBenchmarkResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for n in result.config.num_qubits_range:
        ideal = result.get_runs("QAOA_Ideal", n)
        noisy = result.get_runs("QAOA_Noisy", n)
        if ideal and noisy:
            ideal_mean = np.mean([r.approximation_ratio for r in ideal])
            noisy_mean = np.mean([r.approximation_ratio for r in noisy])
            degradation = (ideal_mean - noisy_mean) / ideal_mean * 100
            ax.bar(n, degradation, width=1.5, color="#FF5722", alpha=0.7, edgecolor="black")
    ax.set(xlabel="Number of Qubits", ylabel="Quality Degradation (%)")
    ax.set_title("Noise-Induced Performance Loss")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_count_distribution(result: HardwareBenchmarkResult, path: Path) -> None:
    n = min(result.config.num_qubits_range)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for ax, solver, title in [
        (axes[0], "QAOA_Ideal", "Ideal Simulator"),
        (axes[1], "QAOA_Noisy", "Noisy Simulator"),
    ]:
        runs = result.get_runs(solver, n)
        if runs and runs[0].counts:
            top = sorted(runs[0].counts.items(), key=lambda kv: -kv[1])[:15]
            ax.bar(
                range(len(top)),
                [count for _, count in top],
                color=COLORS[solver],
                alpha=0.7,
                edgecolor="black",
                linewidth=0.3,
            )
            ax.set_xticks(range(len(top)))
            ax.set_xticklabels([bs[:8] for bs, _ in top], rotation=90, fontsize=6)
            ax.set(ylabel="Counts", title=f"{title} (n={n})")
            ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Measurement Output Distribution", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def generate_hardware_figures(result: HardwareBenchmarkResult, output_dir: Path) -> list[Path]:
    """Write the fig_hw_*.pdf figures to `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLOT_RC)
    figures = {
        "fig_hw_approx_ratio_vs_size.pdf": _fig_ratio_vs_size,
        "fig_hw_solve_time_scaling.pdf": _fig_time_scaling,
        "fig_hw_energy_distribution.pdf": _fig_energy_distribution,
        "fig_hw_depth_effect.pdf": _fig_depth_effect,
        "fig_hw_noise_degradation.pdf": _fig_noise_degradation,
        "fig_hw_count_distribution.pdf": _fig_count_distribution,
    }
    saved = []
    for name, draw in figures.items():
        draw(result, output_dir / name)
        saved.append(output_dir / name)
    for solver, suffix in [("QAOA_Ideal", "ideal"), ("QAOA_Noisy", "noisy")]:
        path = output_dir / f"fig_hw_success_prob_{suffix}.pdf"
        _fig_success_heatmap(result, solver, path)
        saved.append(path)
    return saved


QUICK_CONFIG = HardwareBenchmarkConfig(
    num_qubits_range=[4, 6, 8],
    qaoa_depths=[1, 2],
    shots=1000,
    optimizer_maxiter=30,
    sa_sweeps=500,
    num_runs_per_config=3,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--simulator-only",
        action="store_true",
        help="skip IBM hardware even if credentials are set",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--counts-log", type=Path, default=DEFAULT_COUNTS_LOG)
    parser.add_argument("--seed", type=int, default=QUICK_CONFIG.seed)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    token = None
    if not args.simulator_only:
        token = os.environ.get("IBM_QUANTUM_TOKEN") or os.environ.get("IBM_CLOUD_API_KEY")
    config = HardwareBenchmarkConfig(**{**vars(QUICK_CONFIG), "seed": args.seed})
    result = run_hardware_benchmark(
        config,
        token=token,
        instance=os.environ.get("IBM_CLOUD_CRN"),
        counts_log=args.counts_log,
    )
    for path in generate_hardware_figures(result, args.output_dir):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
