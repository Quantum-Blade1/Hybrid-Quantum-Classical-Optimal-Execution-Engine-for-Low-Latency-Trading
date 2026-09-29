"""Figure registry: every PDF in paper/figures/ -> plot function -> results inputs -> experiment.

`kind` is "empirical" (numbers read from results/) or "illustrative" (a diagram computed
directly from qexec that makes no empirical claim). `optional` figures read inputs that can
legitimately be absent (real data before `make data`, recovered IBM counts); they are
skipped with a message instead of failing when an input is missing. `REMOVED` lists
figures the paper still references that were deleted because they plotted fabricated
numbers (docs/CLAIMS_AUDIT.md, section 1); they have no producer and must not be recreated.
"""

from collections.abc import Callable
from dataclasses import dataclass

from matplotlib.figure import Figure

from figures import diagrams, execution, formulation, market, realdata, solvers
from figures.common import Results


@dataclass(frozen=True)
class FigureSpec:
    file: str
    plot: Callable[[Results], Figure]
    inputs: tuple[str, ...]
    kind: str = "empirical"
    note: str = ""
    optional: bool = False

    @property
    def experiments(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(i.split("/")[0] for i in self.inputs))


def _spec(
    file: str,
    plot: Callable[[Results], Figure],
    *inputs: str,
    kind: str = "empirical",
    note: str = "",
    optional: bool = False,
) -> FigureSpec:
    return FigureSpec(file, plot, tuple(inputs), kind, note, optional)


QB = "qaoa_benchmark/runs.csv"
FIGURES: tuple[FigureSpec, ...] = (
    _spec(
        "fig01_system_architecture.pdf",
        diagrams.fig01_system_architecture,
        kind="illustrative",
        note="Block diagram; no numbers.",
    ),
    _spec(
        "fig02_qubo_matrix_structure.pdf",
        diagrams.fig02_qubo_matrix_structure,
        kind="illustrative",
        note="Builds a 20-variable QUBO matrix to show its structure.",
    ),
    _spec(
        "fig03_hft_qubo_cost_decomposition.pdf",
        formulation.fig03_hft_qubo_cost_decomposition,
        "formulation/cost_breakdown_summary.csv",
    ),
    _spec(
        "fig04_qubo_ising_conversion.pdf",
        formulation.fig04_qubo_ising_conversion,
        "formulation/ising_check.csv",
    ),
    _spec(
        "fig05_sa_convergence.pdf",
        solvers.fig05_sa_convergence,
        "solver_benchmark/sa_convergence_history.csv",
        "solver_benchmark/sa_convergence_final.csv",
    ),
    _spec(
        "fig06_solver_comparison.pdf",
        solvers.fig06_solver_comparison,
        "solver_benchmark/summary.csv",
    ),
    _spec("fig07_qaoa_vs_sa.pdf", solvers.fig07_qaoa_vs_sa, QB),
    _spec(
        "fig08_qaoa_landscape.pdf",
        solvers.fig08_qaoa_landscape,
        "qaoa_landscape/landscape.csv",
        "qaoa_landscape/summary.json",
    ),
    _spec(
        "fig09_solution_quality_scaling.pdf",
        solvers.fig09_solution_quality_scaling,
        "solver_benchmark/summary.csv",
    ),
    _spec(
        "fig10_kyle_lambda.pdf",
        market.fig10_kyle_lambda,
        "microstructure/series.csv",
        "microstructure/summary.csv",
    ),
    _spec("fig11_vpin_estimation.pdf", market.fig11_vpin_estimation, "microstructure/series.csv"),
    _spec(
        "fig12_microstructure_dashboard.pdf",
        market.fig12_microstructure_dashboard,
        "microstructure/series.csv",
    ),
    _spec("fig15_lambda_adaptation.pdf", market.fig15_lambda_adaptation, "regime/series.csv"),
    _spec(
        "fig16_volatility_regime_detection.pdf",
        market.fig16_volatility_regime_detection,
        "regime/series.csv",
    ),
    _spec(
        "fig17_regime_distribution.pdf",
        market.fig17_regime_distribution,
        "regime/summary.csv",
        "regime/series_1000.csv",
    ),
    _spec(
        "fig18_qubo_param_sensitivity.pdf",
        formulation.fig18_qubo_param_sensitivity,
        "formulation/sensitivity.csv",
    ),
    _spec(
        "fig19_execution_schedule_comparison.pdf",
        execution.fig19_execution_schedule_comparison,
        "ac_frontier/schedules.csv",
    ),
    _spec(
        "fig20_almgren_chriss_frontier.pdf",
        execution.fig20_almgren_chriss_frontier,
        "ac_frontier/frontier.csv",
    ),
    _spec(
        "fig21_walk_forward_shortfall.pdf",
        execution.fig21_walk_forward_shortfall,
        "walk_forward/summary.csv",
        "walk_forward/paired.csv",
    ),
    _spec(
        "fig22_is_decomposition.pdf",
        execution.fig22_is_decomposition,
        "is_comparison/summary.csv",
        "is_comparison/paired.csv",
        note="Replaces the removed hand-typed fig22 (audit R10) with measured data.",
    ),
    _spec(
        "fig23_stress_test_results.pdf",
        execution.fig23_stress_test_results,
        "stress_test/summary.csv",
        "stress_test/paired.csv",
        note="Replaces the removed hand-typed fig23 (audit R11) with measured data.",
    ),
    _spec(
        "fig24_latency_distribution.pdf",
        execution.fig24_latency_distribution,
        "latency/samples.csv",
        "latency/manifest.json",
        note="Replaces the removed lognormal-draw fig24 (audit R12) with measured latency.",
    ),
    _spec(
        "fig27_venue_routing.pdf", formulation.fig27_venue_routing, "formulation/venue_routing.csv"
    ),
    _spec(
        "fig28_qaoa_circuit_depth.pdf",
        formulation.fig28_qaoa_circuit_depth,
        "formulation/circuit_scaling.csv",
    ),
    _spec(
        "fig_hw_approx_ratio_vs_size.pdf",
        solvers.fig_hw_approx_ratio_vs_size,
        QB,
        note="Simulators only; IBM records unverifiable (audit F10).",
    ),
    _spec(
        "fig_hw_solve_time_scaling.pdf",
        solvers.fig_hw_solve_time_scaling,
        QB,
        note="Simulators only.",
    ),
    _spec(
        "fig_hw_energy_distribution.pdf",
        solvers.fig_hw_energy_distribution,
        QB,
        note="Simulators only.",
    ),
    _spec("fig_hw_depth_effect.pdf", solvers.fig_hw_depth_effect, QB),
    _spec("fig_hw_success_prob_ideal.pdf", solvers.fig_hw_success_prob_ideal, QB),
    _spec("fig_hw_success_prob_noisy.pdf", solvers.fig_hw_success_prob_noisy, QB),
    _spec("fig_hw_noise_degradation.pdf", solvers.fig_hw_noise_degradation, QB),
    _spec(
        "fig_hw_count_distribution.pdf",
        solvers.fig_hw_count_distribution,
        "qaoa_benchmark/top_counts.csv",
    ),
    _spec(
        "fig_real_primary_comparisons.pdf",
        realdata.fig_real_primary_comparisons,
        "real_data_test/comparisons.csv",
        optional=True,
        note="Held-out test days; the 12 pre-registered primary comparisons.",
    ),
    _spec(
        "fig_real_dev_comparisons.pdf",
        realdata.fig_real_dev_comparisons,
        "real_data_dev/comparisons.csv",
        optional=True,
        note="Development days, in-sample.",
    ),
    _spec(
        "fig_real_sensitivity.pdf",
        realdata.fig_real_sensitivity,
        "real_data_test/comparisons.csv",
        optional=True,
    ),
    _spec(
        "fig_real_cost_components.pdf",
        realdata.fig_real_cost_components,
        "real_data_test/strategy_summary.csv",
        optional=True,
    ),
    _spec(
        "fig_real_impact_calibration.pdf",
        realdata.fig_real_impact_calibration,
        "real_data_dev/impact_bins.csv",
        optional=True,
    ),
    _spec(
        "fig_real_qubo_ac_gap.pdf",
        realdata.fig_real_qubo_ac_gap,
        "real_data_test/gaps.csv",
        optional=True,
    ),
    _spec(
        "fig_hw_ibm_success_prob.pdf",
        realdata.fig_hw_ibm_success_prob,
        "hardware/summary.csv",
        "hardware/manifest.json",
        optional=True,
        note="Recovered ibm_fez counts; absent until recovered.",
    ),
    _spec(
        "fig_hw_ibm_approx_ratio.pdf",
        realdata.fig_hw_ibm_approx_ratio,
        "hardware/jobs.csv",
        optional=True,
        note="Recovered ibm_fez counts; absent until recovered.",
    ),
)

# Referenced by paper/main.tex, deleted in Phase 1 as fabricated; no producer by design.
REMOVED = {
    "fig13_adverse_selection_venue.pdf": "R8: hand-typed venue scores",
    "fig14_order_flow_imbalance.pdf": "R9: random walk; OFI not implemented",
    "fig25_policy_staleness.pdf": "R13: synthetic sawtooth",
    "fig26_pipeline_execution_trace.pdf": "R14: random prices, no pipeline code",
    "fig29_error_mitigation.pdf": "R15: hand-built counts",
    "fig30_quantum_advantage_projection.pdf": "R16: hand-typed projection",
}


def by_file() -> dict[str, FigureSpec]:
    return {spec.file: spec for spec in FIGURES}
