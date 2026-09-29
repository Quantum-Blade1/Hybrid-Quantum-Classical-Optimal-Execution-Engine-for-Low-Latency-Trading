# Claims Audit

Phase 1 of the refactor: remove fabricated results so that every reported number traces to a real computation.
Baseline: tag `pre-refactor`. This ledger covers `src/`, `examples/`, `README.md`, the docs, the figures in `figures/` and `assets/`, and every quantitative claim in `paper/main.tex`. `paper/main.tex` was not edited in this phase.

"Fabricated" here means a number shown as a result that comes from a hand-typed literal, a random draw, or a hardcoded multiplier instead of from running the project's code.

> **Paths updated in Phase 3.** The entries below keep the file names that were current when the audit was written. The code now lives in the `qexec` package:
> `src/ibm_hardware_benchmark.py` → `experiments/hardware_benchmark.py` (`build_execution_qubo` → `toy_execution_qubo`, unchanged, now in `src/qexec/optimization/toy.py` and pinned by a golden test; IBM service/sampler code → `src/qexec/hardware/ibm.py`) ·
> `src/generate_journal_figures.py` → `experiments/journal_figures.py` ·
> `src/benchmark.py` → `experiments/solver_benchmark.py` ·
> `src/is_comparison.py`, `src/load_test.py` → `experiments/` ·
> `src/qaoa_solver.py` → `src/qexec/optimization/solvers/qaoa.py` (compare/plot part → `experiments/qaoa_vs_sa.py`) ·
> `src/hybrid_async.py` → `src/qexec/runtime/{policy,optimizer,engine,controller}.py` ·
> `src/hft_microstructure.py` → `src/qexec/microstructure/{kyle,vpin,adverse_selection,queue,analyzer}.py` ·
> `src/adaptive_risk.py` → `src/qexec/microstructure/regime.py` ·
> `src/hft_qubo.py`, `src/qubo_execution.py`, `src/qubo_to_ising.py` → `src/qexec/optimization/{hft_qubo,qubo,ising}.py` ·
> `src/qubo_solvers.py` → `src/qexec/optimization/solvers/{exact,annealing,greedy,compare,result}.py` ·
> `src/decision_layer.py`, `src/latency_monitor.py`, `src/hft_pipeline.py` → `src/qexec/runtime/{decision,latency,hft_pipeline}.py` ·
> `src/error_mitigation.py` → `src/qexec/hardware/mitigation.py` ·
> `src/walk_forward.py`, `src/implementation_shortfall.py`, `src/stress_test.py` → `src/qexec/analysis/{walk_forward,shortfall,stress}.py` ·
> `src/dashboard.py` → `apps/dashboard.py` ·
> `figures/*.pdf` → `paper/figures/`, `figures/*.json` → `results/` (byte-identical).
> The `run_*_demo` functions cited in F5, F7, F8 and R2 were deleted. `src/generate_academic_plots.py` (R17, R18) was deleted. F4 is resolved: the runners `stress_test.py` needed now live in `src/qexec/analysis/runners.py`, and `experiments/stress_test.py` runs the suite. F6 now refers to `qexec.analysis.runners`.

---

## 1. Removed

| # | Location | What it did | Why removed |
|---|----------|-------------|-------------|
| R1 | `src/dqc_client.py` (whole module) | A "Distributed Quantum Computing" backend. It ran ordinary SA (≤50 sweeps), multiplied the energy by 1.02 ("Assume DQC finds 2% better solution"), and reported a made-up execution time (`0.001·n²/partitions + 0.05`) and speedup. | Fake backend with a fabricated energy gain and fabricated timing. |
| R2 | `src/benchmark.py:run_benchmark_demo` | Listed `SimulatedDQCSolver(workers=4/16)`. That class was never defined, so the demo raised `NameError`. | Reference to the fabricated DQC backend. |
| R3 | `src/benchmark.py` module docstring | Said "Real Quantum Hardware: IBM (placeholder)". | The module has no hardware solver, so the line claimed a capability that does not exist. |
| R4 | `README.md` Key Features | Advertised a "DQC backend for scaling >100 variables". | Described the fabricated backend (R1). |
| R5 | `README.md` Benchmarks table, "Avg Shortfall" row | Showed VWAP +15 bps, TWAP +8 bps, Hybrid +5 bps. | No code or data produces these numbers. |
| R6 | `docs/MATHEMATICAL_MODEL.md`, `docs/FAQ_TOP_30.md` (Q22, Q26, Q27) | Described DQC decomposition and capability as if implemented, and pointed to `dqc_client.py`'s `network_latency` (a parameter that never existed). | Described the fabricated backend. Q27 now says the backend was removed. |
| R7 | `src/hybrid_async.py:AsyncOptimizer._optimize_qaoa` and its `'qaoa'` option | Picking `optimizer_type='qaoa'` silently ran SA, and the result was labelled `optimizer_name='qaoa'`. | Results were mislabelled as QAOA. The async runtime now accepts only `'sa'` or `'uniform'` and raises `ValueError` for anything else. QAOA stays available offline in `src/qaoa_solver.py`. |
| R8 | `src/generate_journal_figures.py:fig13_adverse_selection_by_venue` and `figures/fig13_adverse_selection_venue.pdf` | Plotted hand-typed venue scores (AS `[0.45,0.12,0.28,0.08]`, fill `[0.95,0.55,0.82,0.40]`, leakage `[0.8,0.15,0.5,0.05]`). | Hardcoded numbers shown as results. These literals even contradict the paper's caption: they give dark pools a lower AS than lit venues. |
| R9 | `...:fig14_order_flow_imbalance` and its PDF | Plotted a normalised random walk as "imbalance", plus "QUBO weight multipliers" `1+0.5\|OFI\|` and `1+0.3·OFI²`. | Random data. The multipliers are not implemented anywhere in `src/`, and `hft_microstructure` hardwires `order_imbalance=0.0`. |
| R10 | `...:fig22_implementation_shortfall_decomposition` and its PDF | IS decomposition (delay/impact/timing/opportunity) for 5 strategies, all hand-typed. | Hardcoded results. |
| R11 | `...:fig23_stress_test_results` and its PDF | Classical vs hybrid slippage `[45,28,35,80]` vs `[32,18,22,55]` and fill rates, all hand-typed. | Hardcoded results. `src/stress_test.py` was never called, and it cannot run (see F4). |
| R12 | `...:fig24_latency_distribution` and its PDF | Showed "fast path" latency as `np.random.lognormal(3.5,0.5)` (median ≈ 33 µs, P99 ≈ 80 µs). The slow path and policy propagation were also lognormal draws. | Random numbers presented as measured latency. This is the only source of the paper's "median 33 µs, P99 = 80 µs". |
| R13 | `...:fig25_policy_staleness` and its PDF | Showed "staleness" as a synthetic sawtooth `(i%20)·5 + Exp(3)` ms. The right panel plotted the assumed formula `0.002·√T`. | Synthetic series presented as a measurement. |
| R14 | `...:fig26_pipeline_execution_trace` and its PDF | Showed a "pipeline trace" of random prices and shares with λ hand-set to 0.5/2.0/0.8. It called no pipeline code. | Random or hardcoded data. The paper's caption describes content that is not in the figure at all (interleaved fast/slow cycles). |
| R15 | `...:fig29_error_mitigation` and its PDF | Built "ideal" counts from uniform random bitstrings weighted by `exp(-E/2)`, then injected 40% single-bit flips by hand, then applied MEM (with an assumed 2% confusion matrix), majority voting and ZNE. | The counts were not produced by any circuit or noise model. ZNE got no noise-scaled data, so it returns the raw value unchanged. The chart does not measure error mitigation. |
| R16 | `...:fig30_quantum_advantage_projection` and its PDF | Showed quantum solve time `[100,50,20,8,3,1,0.3,0.05]` s per year, SA fixed at 0.01 s, and "max QUBO variables" `[20…2000]`, all hand-typed. | Hand-made projection presented as analysis. |
| R17 | `src/generate_academic_plots.py:generate_figure_2_performance` and `assets/figure2_performance.png` | Panel B "Cumulative savings" was `cumsum(N(50,20))` random. Panel C "optimizer events" had random ticks and `uniform(-100,-150)` energies. | Random numbers presented as results. |
| R18 | `src/generate_academic_plots.py:generate_figure_3_scaling`, the "Quantum (Projected)" series | Plotted `0.05·exp(0.05n)` ("Mock: Very flat scaling"). | Made-up curve. The real BF and SA timings are still in the figure. |
| R19 | `src/is_comparison.py`, the "Hybrid" branch, and `assets/is_comparison.png` | Copied the TWAP fills, added `N(0,100)` noise to the share counts, and re-priced them with a hand-picked lower impact coefficient (`0.005·√q`). | The hybrid IS was fabricated. The script now compares VWAP and TWAP only. |
| R20 | `assets/walk_forward_results.png` | Generated before commit `a12fe39`, when `walk_forward.py` computed hybrid IS as `adaptive IS − 0.20·impact` (a hardcoded 20% saving). | Output of a fabricated computation. The current `walk_forward.py` runs real SA and writes a new PNG when run. |
| R21 | `src/dashboard.py`, "Latest Solution Energy" metric | Displayed `-150 + std(schedule)` as the solution energy ("Mock energy based on variance"). | Fabricated metric. |

`src/generate_journal_figures.py` keeps its original figure numbering, so the removed numbers (13, 14, 22–26, 29, 30) are simply missing. It now generates 21 figures.

### Tests
No test covered any removed item. `tests/` is unchanged, and `.venv/bin/python -m pytest tests -q` gives 91 passed.

---

## 2. Flagged, not changed in this phase

These items are not fabricated results, but they matter for whether the results can be trusted. They are left for later phases.

| # | Location | Issue |
|---|----------|-------|
| F1 | `src/ibm_hardware_benchmark.py:solve_with_sa / solve_with_qaoa_simulator / solve_with_ibm_hardware`; `generate_journal_figures.py:fig06` | **The approximation-ratio metric is wrong for negative energies.** It computes `ratio = min(E_opt/E, 1.0)`. When both energies are negative and the solution is worse (|E| < |E_opt|), `E_opt/E > 1`, so the ratio is clipped to **1.000**. This already happens in the committed data: SA n=10 reaches −248.8, −248.7 and −248.65 against an optimum of −248.9, and all three are recorded with ratio 1.0. The same happens for SA at n=6 and n=8. Every "ratio = 1.000" claim is therefore uninformative. |
| F2 | `src/ibm_hardware_benchmark.py` QAOA solvers | **The quality metric saturates.** "Energy" is the best bitstring among `5×shots` final samples (≥ 10,000 on hardware). There are at most 2^10 = 1,024 bitstrings, so a uniform random sampler misses a given optimum with probability ≈ e^(−10000/1024) ≈ 6×10⁻⁵. A ratio of 1.000 is expected even from pure noise. |
| F3 | `src/ibm_hardware_benchmark.py:build_execution_qubo` | **The hardware and benchmark QUBO is not the paper's six-term HFT QUBO (`src/hft_qubo.py`).** It is a small toy: n/2 slices × 2 levels, a diagonal impact term `0.1·q²` plus a linear timing term `0.05·(t+1)·q`, and an equality penalty of weight 10 that dominates. It has no adverse selection, inventory, leakage or venue terms, and no microstructure inputs. All of `bench_*.json`, `bench_hw_*.json` and the `fig_hw_*` figures use this toy QUBO. |
| F4 | `src/stress_test.py` | It cannot run: it imports `hybrid_demo` from the repo root, but the file lives at `examples/hybrid_demo.py`. Its scenario definitions also differ from the paper (see P14). |
| F5 | `src/decision_layer.py:ImprovementTracker.expected_improvement` | With no history it assumes a 1% improvement. `analyze_lambda_sensitivity`, `run_decision_demo` and `examples/hybrid_demo.py:run_hybrid_execution` seed the tracker with synthetic ~5% improvements (baseline 100 → 95), so the decision layer's "expected improvement" is an assumption, not a measurement. This is a heuristic prior and is kept, but no invocation-rate or improvement numbers from it should be reported. |
| F6 | `examples/hybrid_demo.py` | The comparison is not like-for-like. VWAP runs through `ExecutionEngine`, which has an impact model. The SA and hybrid modes fill at `price + spread/2` with no impact, so their "cost" advantage is built in. |
| F7 | `src/latency_monitor.py:run_latency_demo` | It times `time.sleep(Exp(...))`, so the reported latencies are random. It demonstrates the API only, and its numbers must not be reported. |
| F8 | `src/error_mitigation.py:run_mitigation_demo` | Uses hand-built noisy counts. Demo only. `ZeroNoiseExtrapolation.mitigate` returns the raw value when it gets no noise-scaled counts, and there is no gate folding anywhere in `src/`. |
| F9 | `src/walk_forward.py` | `MarketDataSimulator` is unseeded, so results change on every run. The default in `fig21` is 8 days / 3 train / 1 test, which gives **5 windows**. |
| F10 | `figures/bench_hw_*.json` | Eight records (n=4: 3, n=6: 1, n=8: 1, n=10: 3). All have `counts: {}` and `success_probability: 0.0`, yet energy equals the optimum. Under the code, finding the optimal bitstring means `success_probability ≥ 1/shots`, so these records cannot be raw output of `solve_with_ibm_hardware`. They have no job IDs, shots, transpiled depth or timestamps. The same 8 records are also merged into `figures/bench_n*.json`, which hold 68 records = 60 simulator + 8 hardware. Left untouched until the raw counts are recovered. |
| F11 | `fig_hw_*.pdf` provenance | `generate_hardware_figures` takes an in-memory `HardwareBenchmarkResult`. No code in the repo reads `bench_*.json` back, so the script that produced the committed `fig_hw_*` PDFs from the JSON is not in the repo. |
| F12 | `results_paper.md`, `EXTREME_CAPABILITIES.md` (linked from the README) | Contain unsupported numbers: "~15–20% IS reduction", "12 bps", "8% timing-variance reduction", "95% capital preservation", "18 bps alpha", "25% impact reduction", "100% uptime". No code produces them. **Resolved in Phase 2: both files deleted.** |

---

## 3. Paper claims without a valid source (`paper/main.tex`)

Status key:
- **removed**: the figure or data behind the claim was deleted in this phase.
- **needs rerun**: a real experiment exists or can be written, but the current numbers are missing, inconsistent or broken.
- **needs rewording**: the claim is not a measurement of this system (priority claims, implementation details that do not match the code, captions that do not match their figures).

| # | Claim | Line(s) | Current source | Status | Experiment that would substantiate it |
|---|-------|---------|----------------|--------|---------------------------------------|
| P1 | "First empirical validation of execution-QUBO on real quantum hardware" / "no prior work…" | 79, 129, 154, 984, 1207, 1259 | None (priority claim). The hardware problem is the toy QUBO (F3). | needs rewording | Literature search. Run the actual six-term `HFTExecutionQUBO` (or a faithful reduction) on hardware. |
| P2 | All IBM `ibm_fez` runs reach approximation ratio 1.000 (Table `tab:hw_results`, `fig_hw_approx_ratio_vs_size`) | 79, 129, 988–999, 1007, 1009, 1019, 1207, 1259 | `bench_hw_*.json`: empty counts and success probability 0.0 (F10). The ratio metric is clipped (F1) and best-of-shots saturates (F2). | needs rerun | Recover the raw counts and job IDs from IBM. Report the probability of the optimal state and ⟨H⟩-based ratios against a uniform-random baseline (2⁻ⁿ) and against random-parameter circuits, after fixing F1. |
| P3 | Benchmark instances: T=5, V=1, K∈{4,6,8,10} → n∈{4,6,8,10}. Hardware problem is the "execution-QUBO". | 386, 994–997 | `build_execution_qubo` (toy: n/2 slices × 2 levels). The stated arithmetic is also wrong: T·V·K = 20–50, not 4–10. | needs rewording | Describe the real instance, or benchmark the six-term QUBO at these sizes. |
| P4 | "68 configurations: 60 simulator + 8 hardware", "all achieve 1.000" | 79, 950, 977, 1025, 1029, 1259 | `bench_n*.json` (68 records incl. 8 HW). The "all 1.000" part rests on the F1 clipping: SA n=6/8/10 include suboptimal runs. | needs rerun | Re-run after fixing the metric, and store the config with the results. |
| P5 | Table `tab:sim_results` values | 1035–1056 | `bench_n*.json`. Several cells differ from the JSON: SA n=6 mean 4.7 vs 5.0 ms; n=8 5.5 vs 5.9 ms; n=10 5.9 vs 5.5 ms; SA n=10 success "<0.001" vs 0.000 (never optimal, yet ratio 1.000); Ideal n=8 p=2 success 0.007 vs 0.006. | needs rerun | Generate the table programmatically from the JSON. |
| P6 | Config table and text: 10,000 sim shots, maxiter 40, SA 1,000 sweeps with cooling 0.995 and T₀ = max\|Qᵢᵢ\|, readout error 5% | 839, 865, 908, 962–975 | Code defaults: `HardwareBenchmarkConfig` has shots 4000, maxiter 50, 5 runs, depths 1–3, sizes 4–12. `SimulatedAnnealingSolver` has T₀=10, cooling 0.95. The noise model's readout is p(0\|1)=0.02, p(1\|0)=0.016. No record of the config actually used. | needs rerun | Log the full config in every results file, then re-run. |
| P7 | Hardware time breakdown: queue ~60–120 s, transpile ~10–30 s, QPU ~100 µs per circuit | 1011 | Not recorded anywhere. | needs rerun | Record per-stage timestamps and IBM job metrics (`usage`, `quantum_seconds`). |
| P8 | "24 planned, 8 completed; quota exhausted" | 1013, 1241 | Not in the repo (commit `5ddc264` says 10 runs; the JSON has 8). | needs rerun | Verify against IBM job history and report job IDs. |
| P9 | "Noise reduces success probability 30–50% vs ideal" | 1091 | JSON contradicts it: n=4 p=1 noisy 0.079 vs ideal 0.044 (higher); n=6 p=1 0.052 vs 0.271 (−81%). | needs rewording | Report the per-size ratios from the JSON with CIs across more runs. |
| P10 | `fig_hw_noise_degradation`: "noisy/ideal ratios remain above 0.5" | 1100 | The figure plots % degradation of the (clipped) approximation ratio, which is 0 everywhere. | needs rewording | Plot the success-probability or ⟨H⟩ ratio after fixing F1. |
| P11 | "15–40% implementation-shortfall reduction vs static strategies" | 79, 1257 | No source. In a `fig21_walk_forward_shortfall` run during this audit, hybrid won 2 of 5 windows, and its cumulative IS was −13,486 against −34,468 for static VWAP (worse). The runs are unseeded (F9). | needs rerun | Seeded, multi-seed walk-forward with CIs and a like-for-like fill model. |
| P12 | fig21: "20 non-overlapping windows, 32% reduction, consistent outperformance" | 1152 | `walk_forward.py` produces 5 windows and gives no consistent result. | needs rerun | Same as P11. |
| P13 | IS decomposition (fig22): impact −25%, timing +5%, net −20% | 1161 | Hand-typed (R10). | removed | Run `ISAnalyzer` on each strategy's fills over many seeds. |
| P14 | Stress testing (fig23): slippage −20–30%, fill +10–25% vs TWAP; scenario list | 131, 1176–1189 | Hand-typed (R11). The scenarios in the text (5% drop in 10 ticks, 80% liquidity cut, 5× vol, 50-tick outage) differ from `stress_test.py` (50% drop, 10× spread with 90% volume cut, $5 jumps, 10-minute outage), and that script cannot run (F4). | removed | Fix and run `stress_test.py` with the stated scenarios over many seeds. |
| P15 | Fast path median 33 µs, P₉₉ = 80 µs; "sub-millisecond" fast path; slow path SA ~5 ms, QAOA ~5 s | 79, 105, 127, 712, 720, 727, 1196, 1257 | The µs figures came from `np.random.lognormal` (R12). The SA and QAOA times are consistent with `bench_n*.json`. | removed (µs claims) | Instrument `AsyncExecutionEngine` and `HFTQuantumPipeline` ticks with `LatencyMonitor` over a tick replay. Report hardware, OS and Python version. |
| P16 | Kernel bypass (DPDK/Solarflare), ring buffers, zero allocation, atomic CAS, "lock-free policy queue" | 709, 712, 736 | The code is Python threads. `PolicyQueue` uses `threading.Lock`. None of these features exist. | needs rewording | Describe what is actually implemented. |
| P17 | Staleness regret ≤ 4.5 bps; fig25 sawtooth | 786, 792 | fig25 was synthetic (R13). The corollary follows from assumed σ_S=0.002 and L=1 with t in ms, and the "cost units → bps" step is not justified. | removed (fig), needs rewording (corollary) | Measure the cost difference when policy application is delayed by Δ in simulation. |
| P18 | fig26 pipeline execution trace | 799 | Random data (R14). The caption describes content that is not in the figure. | removed | Log a real `HFTQuantumPipeline` run (tick, policy version, solve events). |
| P19 | Dark pools have higher AS than lit (fig13) | 586, 591 | Hand-typed (R8), and it contradicts its own literals. | removed | Measure AS per venue on real or simulated multi-venue fills. |
| P20 | OFI signal (fig14) and OFI in the state vector and fast path | 597–612, 733 | Random (R9). `order_imbalance` is hardwired to 0.0 in `hft_microstructure.py`. | removed (fig), needs rewording (text) | Implement OFI, then plot it on the replayed data. |
| P21 | Error mitigation: MEM 8–15%, ZNE 5–10% (fig29); ZNE via gate folding c∈{1,2,3} + Richardson | 928–941 | Synthetic counts (R15). There is no gate folding. The code's ZNE uses polyfit on factors [1,1.5,2]. | removed | Run calibrated MEM and folded-circuit ZNE on the noisy simulator and hardware. |
| P22 | Quantum-advantage projection: crossover 2028–29 at n≈50–100; HFT scale feasible 2027–28 (fig30); "crossover at n~50–100" | 1218, 1228 | Hand-typed (R16). | removed | Drop, or cite external roadmaps explicitly as projections. |
| P23 | Two-qubit error below 1e-4 by 2026–27 | 1220 | No source. | needs rewording | Cite IBM roadmap or remove. |
| P24 | C₄ shifts 10–15% of volume away from toxic periods; C₆ cuts leakage 20–30% | 1234 | No experiment. | needs rerun | Ablation: solve `HFTExecutionQUBO` with and without C₄/C₆ under time-varying VPIN. |
| P25 | fig03 caption: impact ~43%, timing ~28%, AS ~11%, leakage ~6% | 510 | A rerun of `fig03` code gives leakage 83.7%, impact 14.2%, AS 0.9%, transaction 0.8%, inventory 0.3%, timing 0.1% (n=120). | needs rewording | Report the computed shares. |
| P26 | fig17: NORMAL ~60%, HIGH ~20%, EXTREME ~5%, "transition matrix" | 698 | A rerun gives NORMAL 71.8%, LOW 19.8%, HIGH 8.3%, EXTREME 0.1%. The figure has no transition matrix. | needs rewording | Report the computed distribution. |
| P27 | λ scaled "up to 4×", "4× amplification during EXTREME" (fig15) | 125, 691, 1257 | The 4.0 multiplier exists in `adaptive_risk.py`, but on the fig15/17 data λ ranges only 0.25–1.20 × base. | needs rewording | Show a scenario that reaches EXTREME, or state the observed range. |
| P28 | EXTREME regime liquidates 80–90% in the first two slices | 685 | No computation. | needs rerun | Compute the AC trajectory with κ×6. |
| P29 | fig10: Kyle λ rises 2–3× in stress | 550 | A rerun gives about 5.8× (mean 7.4e-5 before, 4.3e-4 in stress). Trade side is set to sign(return) by the synthetic generator, so the rise is built in. | needs rewording | Report the computed ratio and note the construction. |
| P30 | Kyle λ via exponentially weighted OLS with Lee-Ready; VPIN via bulk-volume classification (Φ) | 540–565 | The code uses equal-weight rolling OLS and the tick rule, with no BVC. | needs rewording | Describe the implemented estimators, or implement the stated ones. |
| P31 | VPIN > τ=0.7 amplifies w₄ via `1 + max(0,VPIN−τ)/(1−τ)`; C₄ = λ_K·VPIN·q² | 435, 567–570 | `hft_qubo.py` uses `adverse_selection_cost · venue_as · (1 + 2·VPIN)`. There is no threshold and no Kyle-λ factor in that term. | needs rewording | Describe the implemented term, or implement the stated one. |
| P32 | QUBO weights "calibrated from historical execution data"; η from the square-root model | 395, 410 | Hardcoded defaults in `HFTQUBOConfig`. No calibration code or data. | needs rewording | Calibrate on real TAQ/LOBSTER data, or state that the values are assumed. |
| P33 | fig02 caption "n=4" | 503 | The figure is a 20×20 matrix (T=5, K=4). | needs rewording | Fix the caption. |
| P34 | QUBO↔Ising verified for n = 4…20, relative error < 1e-10 | 370, 517 | `fig04` uses n = 4…16 and plots absolute error. | needs rewording | Match the text to the code, or extend the sizes. |
| P35 | fig06: "all solvers achieve ratio 1.0 at n ≤ 10"; fig09: "noisy QAOA shows increasing variance" | 879, 893 | fig06 plots `min(E_opt/E,1)` (F1). fig09 contains no QAOA. | needs rerun | Fix the metric and regenerate. Fix the fig09 caption. |
| P36 | fig07: SA > 1000× faster (< 6 ms vs > 4 s) | 886 | `fig07` is computed (n=12), but the caption numbers were not checked against its output. | needs rerun | Print the timings from the figure run and cite them. |
| P37 | Fidelity estimate "COBYLA reliably identifies the optimal bitstring" at F ≈ 2.6e-5 | 916 | Rests on best-of-shots (F2). Also assumes 5% readout error, while the code uses 2%. | needs rewording | See P2. |
| P38 | fig19: QUBO schedule "reflects microstructure-venue interactions" | 1134 | `fig19` uses the single-venue `ExecutionQUBO` with no microstructure inputs. VWAP is a hand-coded U-shape. The QUBO schedule is rescaled to the total. | needs rewording | Use `HFTExecutionQUBO` and `VWAPStrategy`, and show the unscaled fill. |
| P39 | fig27: dark-pool share rises in high-toxicity periods | 1170 | `fig27` uses a constant VPIN=0.4, so toxicity never varies. | needs rerun | Solve with a time-varying VPIN and compare venue shares. |
| P40 | fig18: "cost vs risk aversion λ" | 524 | The x-axis scales `impact_weight = 0.25·λ`, not timing-risk λ. | needs rewording | Relabel, or sweep the actual λ. |
| P41 | "38 publication-quality figures" | 131, 1259 | 9 of the 38 were removed. 29 remain (21 journal + 8 `fig_hw_*`). `main.tex` still `\includegraphics` the 9 deleted PDFs and will not compile until Phase 8. | needs rewording | Update the count and remove the references. |
| P42 | Other "first" claims: first to encode microstructure as QUBO terms; first end-to-end integration | 79, 160, 1255, 1257 | None. | needs rewording | Literature search, or soften the wording. |
| P43 | Data availability: "all benchmark data (JSON with per-run metrics)" | 1279 | The hardware JSON has no counts or job IDs (F10). The JSON-to-figure script is missing (F11). | needs rerun | Commit raw counts, job IDs and the plotting script. |
| P44 | "Calibrated synthetic data" | 1239 | `MarketDataSimulator` parameters are not calibrated to any dataset. | needs rewording | Calibrate to a named dataset, or call the data "synthetic". |

The DQC backend (R1) is not mentioned in the paper.

**Totals (44 claims, one primary status each):** removed **9** (P13–P15, P17–P22) · needs rerun **14** (P2, P4–P8, P11, P12, P24, P28, P35, P36, P39, P43) · needs rewording **21** (P1, P3, P9, P10, P16, P23, P25–P27, P29–P34, P37, P38, P40–P42, P44). The text parts of P17 and P20 also need rewording.

---

## 4. Kept

All kept figures run on simulated market data or synthetic QUBOs. None uses real market data.

### `src/generate_journal_figures.py`

| Figure | Source | Class | Caveats |
|--------|--------|-------|---------|
| fig01_system_architecture | Diagram | (b) illustrative | The "<1ms" and "100ms–5s" labels are design targets, not measurements. |
| fig02_qubo_matrix_structure | `ExecutionQUBO.build_qubo_matrix` | (a) | Caption must say n=20 (P33). |
| fig03_hft_qubo_cost_decomposition | `HFTExecutionQUBO` + `SimulatedAnnealingSolver` + `calculate_cost_breakdown` | (a) | Caption percentages wrong (P25). |
| fig04_qubo_ising_conversion | `qubo_to_ising`, `binary_to_spins` | (a) | n=4…16; absolute error (P34). |
| fig05_sa_convergence | `SimulatedAnnealingSolver` | (a) | — |
| fig06_solver_comparison | `BruteForceSolver`, `SimulatedAnnealingSolver`, `GreedySolver` | (a) | Ratio clipped (F1). |
| fig07_qaoa_vs_sa | `QAOASolver`, `SimulatedAnnealingSolver` | (a) | n=12 `ExecutionQUBO`. |
| fig08_qaoa_landscape | `build_qaoa_circuit_from_ising` on `AerSimulator` | (a) | n=4. |
| fig09_solution_quality_vs_size | `SimulatedAnnealingSolver`, `BruteForceSolver`, `interpret_solution` | (a) | Caption mentions QAOA, which is not in the figure (P35). |
| fig10_kyle_lambda | `KyleLambdaEstimator` on `_generate_micro_data` | (a) | Side = sign(return) by construction (P29). |
| fig11_vpin_estimation | `VPINEstimator` | (a) | Tick rule, not BVC (P30). |
| fig12_microstructure_dashboard | `MicrostructureAnalyzer` | (a) | Shows λ, VPIN, AS and spread. OFI is absent. |
| fig15_lambda_adaptation | `AdaptiveRiskManager` | (a) | λ range 0.25–1.2× (P27). |
| fig16_volatility_regime_detection | `VolatilityEstimator` | (a) | — |
| fig17_regime_distribution | `AdaptiveRiskManager` | (a) | Caption numbers wrong (P26). |
| fig18_qubo_param_sensitivity | `HFTExecutionQUBO` + SA | (a) | The x-axis is really an impact-weight scale (P40). |
| fig19_execution_schedule_comparison | `AlmgrenChrissSolver`, `ExecutionQUBO` + SA | (a)/(b) | TWAP and VWAP are formula shapes; the QUBO schedule is rescaled (P38). |
| fig20_almgren_chriss_frontier | `AlmgrenChrissSolver` | (a) | — |
| fig21_walk_forward_shortfall | `WalkForwardAnalyzer` (real SA + `ExecutionEngine`) | (a) | Unseeded, 5 windows (F9, P11–P12). |
| fig27_venue_routing | `HFTExecutionQUBO` + SA | (a) | Constant VPIN (P39). |
| fig28_qaoa_circuit_depth | `build_qaoa_circuit_from_ising` (logical circuit) | (a) | Untranspiled depth. |

### `src/ibm_hardware_benchmark.py:generate_hardware_figures` (committed `fig_hw_*.pdf`)

These are computed from `HardwareBenchmarkResult` runs of the real SA, ideal Aer and noisy Aer solvers, plus the 8 hardware records. All of them inherit F1, F2, F3, F10 and F11.

| Figure | Content | Caveats |
|--------|---------|---------|
| fig_hw_approx_ratio_vs_size | Mean ratio per solver and size | Clipped ratio; hardware records unverifiable. |
| fig_hw_solve_time_scaling | Mean wall time | Hardware times include the queue. |
| fig_hw_energy_distribution | Best-of-shots energies | Saturated metric (F2). |
| fig_hw_depth_effect | Ratio vs p (simulators) | Clipped ratio. |
| fig_hw_success_prob_ideal / _noisy | Frequency of the best-found bitstring | This is the frequency of the best bitstring, not of the optimum. They coincide here only because the best equals the optimum. |
| fig_hw_noise_degradation | % drop in clipped ratio | Shows 0 everywhere; caption wrong (P10). |
| fig_hw_count_distribution | Top-15 counts, ideal and noisy, n=4 | No hardware counts exist. |

### Other computed outputs kept
- `src/generate_academic_plots.py`: figure 1 (diagram) and figure 3 (real BF/SA timings).
- `src/load_test.py`: real wall-clock measurements of `HybridController`. Numbers depend on the machine.
- `src/ac_benchmark.py`, `src/qubo_integration.py`, `src/benchmark.py` (without DQC), `src/is_comparison.py` (VWAP and TWAP only): all computed.
- `assets/` PNGs other than R17, R19 and R20 were produced by the unchanged current code.
- `figures/bench_n*.json`: simulator records are real runs (subject to F1 and F2). `figures/bench_hw_*.json` are untouched, pending recovery of the raw counts.

---

## 5. Correctness fixes found by the Phase 5 test suite

Each fix has a test under `tests/` that failed before it. Outputs computed with the old code (named below) were not regenerated in Phase 5.

| # | Location | Bug | Effect on outputs |
|---|----------|-----|-------------------|
| T1 | `qexec.execution.strategies.almgren_chriss.calculate_expected_cost` | Used η instead of η̃ = η − γτ/2 (Almgren & Chriss 2000, eq. 20), overstating E[C] by γ Σ n_k² / 2. Checked against a direct simulation of the discrete model. | `fig20_almgren_chriss_frontier` (E[C] axis shifts slightly). |
| T2 | `qexec.analysis.shortfall.ISAnalyzer` | Opportunity cost used (P_T − P_d) U while delay already charged (P_0 − P_d) to all N shares, double counting (P_0 − P_d) U. Now (P_T − P_0) U, so the total equals Perold's shortfall. | `experiments/is_comparison.py` when orders are not fully filled. |
| T3 | `qexec.hardware.mitigation.MeasurementErrorMitigator` | Inverted A instead of Aᵀ for A[prepared, measured]; correct only for symmetric readout error. | None committed (only the default symmetric matrix was ever used). |
| T4 | `qexec.execution.engine.ExecutionEngine` | Child orders were not capped at the parent size, so a schedule summing to more than the order (e.g. a QUBO solution off the equality constraint) overfilled it. | `experiments/strategy_comparison.py`, `examples/qubo_vs_baselines.py`. |
| T5 | `qexec.runtime.engine.AsyncExecutionEngine` | After switching policies mid-order the tick loop followed the new whole-order plan and could trade more than the order. Now capped at the policy total. Underfill after a switch remains possible (the runtime does not re-plan the remainder). | `experiments/load_test.py`, `examples/hybrid_runtime.py`, `apps/dashboard.py` fill counts. |
| T6 | `qexec.execution.strategies.twap` | The remainder share could push a slice one share above `max_slice_pct`. | Negligible. |
| T7 | `qexec.execution.strategies.vwap` | Leftover-share allocation ignored `max_slice_pct`. | VWAP schedules on short horizons. |
| T8 | `qexec.execution.strategies.fixed` | `astype(int)` truncation made schedules sum to less than the order. Now largest-remainder rounding. | Walk-forward hybrid leg (`fig21_walk_forward_shortfall`, `experiments/walk_forward.py`). |

---

## 6. Phase 6 status of every paper claim

Phase 6 re-ran everything from `experiments/` with fixed code (section 5, plus the fair execution model, schedule repair and async re-planning described in `docs/MATHEMATICAL_MODEL.md`, "Evaluation Model"). Numbers below are from the committed `results/`; `docs/RESULTS.md` has the CIs and p-values. `paper/main.tex` is still unedited (Phase 8).

Status key: **supported** (a committed result backs the claim as stated) · **partly supported** · **contradicted** (a committed result disagrees) · **still unsupported** (no experiment produces it) · **rewording** (not a measurement; the text must change to match the code).

| # | Claim (short) | Phase 6 evidence | Status |
|---|---|---|---|
| P1 | First hardware validation of execution-QUBO | No verifiable hardware data; priority claim | still unsupported |
| P2 | All IBM runs ratio 1.000 | No verifiable hardware data (F10). On simulators the best-of-shots metric does not discriminate: uniform sampling with the same budget finds the optimum in 100% of runs (`qaoa_benchmark`) | still unsupported |
| P3 | Instances T=5, V=1, K=4-10 | The benchmark problem is `toy_execution_qubo` (`experiments/problems.py`) | rewording |
| P4 | 68 configurations, all 1.000 | Ideal QAOA best sample optimal in 95% (toy) / 88% (random) of runs; SA as configured optimal in 10% of seeds at n = 8-12 (toy) | contradicted |
| P5 | tab:sim_results values | Not reproducible; replace with `results/qaoa_benchmark/summary.csv` | contradicted |
| P6 | Config: 10k shots, maxiter 40, SA 1000 sweeps / 0.995 / T0 = max Qii, readout 5% | Actual: 1,000 shots x <= 50 COBYLA evaluations + 5,000 final shots; SA T0 = 10, rate 0.95 (stops at 135 sweeps); readout 2% / 1.6% (`qaoa_benchmark/manifest.json`) | contradicted |
| P7 | Hardware time breakdown | Not recorded | still unsupported |
| P8 | 24 planned, 8 completed | Not verifiable | still unsupported |
| P9 | Noise reduces success 30-50% | Noisy/ideal P_opt ranges 0.37-1.6 across (n, p) | contradicted |
| P10 | Noisy/ideal ratios > 0.5 | P_opt ratio 0.37 at n = 4, p = 2 (fig_hw_noise_degradation) | contradicted |
| P11 | 15-40% IS reduction | Hybrid - TWAP: +0.2 [-3.2, 3.5] bps (1 h), +0.7 [-0.2, 1.7] (1% ADV day), +0.3 [-2.1, 2.8] (walk-forward); no CI excludes zero in the hybrid's favour | contradicted |
| P12 | fig21: 20 windows, 32% reduction | 10 windows x 30 seeds: hybrid mean shortfall not lower than TWAP, static or adaptive VWAP | contradicted |
| P13 | fig22: impact -25%, net -20% | QUBO schedules pay *more* impact than TWAP (+0.15 [0.14, 0.17] bps at 1% ADV); total shortfall indistinguishable (fig22) | contradicted |
| P14 | fig23: slippage -20-30%, fill +10-25% vs TWAP | No scenario shows a significant hybrid improvement vs TWAP; every strategy fills 100%; hybrid worse than VWAP in the outage (+9.5 [1.2, 17.8] bps). Scenario definitions still differ from the text | contradicted |
| P15 | Fast path median 33 us, p99 80 us; sub-ms; SA ~5 ms | Measured tick work 16 / 28 us (runtime) and 64 / 103 us (pipeline): sub-millisecond work holds; but ticks start a median 35 ms late under GIL contention, and the runtime SA solve takes 64 ms (3-4 ms only for n <= 20) | partly supported |
| P16 | Kernel bypass, lock-free queue, zero allocation | Python threads and a `threading.Lock` | rewording |
| P17 | Staleness regret <= 4.5 bps | No experiment measures the cost of a delayed policy; propagation delay measured at 7.5 ms median | still unsupported |
| P18 | fig26 pipeline trace | Removed (R14) | still unsupported |
| P19 | Dark pools have higher AS | Removed (R8); no multi-venue fill model | still unsupported |
| P20 | OFI signal | Not implemented | still unsupported |
| P21 | MEM 8-15%, ZNE 5-10% | Removed (R15); no gate folding | still unsupported |
| P22 | Quantum-advantage projection | Removed (R16) | still unsupported |
| P23 | Two-qubit error < 1e-4 by 2026-27 | External roadmap claim | rewording |
| P24 | C4/C6 shift volume / cut leakage | No ablation | still unsupported |
| P25 | fig03: impact 43%, timing 28%, AS 11%, leakage 6% | Leakage 70% [52, 83], impact 26% [15, 41], AS 2.1%, timing 0.1% | contradicted |
| P26 | fig17: NORMAL 60%, HIGH 20%, EXTREME 5% | NORMAL 70.4%, LOW 20.5%, HIGH 9.0%, EXTREME 0.16% | contradicted |
| P27 | lambda up to 4x | Max 1.28x [1.24, 1.32] of base | contradicted |
| P28 | EXTREME liquidates 80-90% in two slices | Not computed | still unsupported |
| P29 | Kyle lambda rises 2-3x | 5.3x [5.1, 5.6], partly by construction of the data | contradicted |
| P30 | EW-OLS + Lee-Ready; BVC | Code: equal-weight rolling OLS, tick rule | rewording |
| P31 | VPIN threshold amplifier | Code: (1 + 2 VPIN) factor; VPIN > 0.7 on 98% of ticks (saturated) | rewording |
| P32 | Weights calibrated on historical data | Hardcoded defaults | rewording |
| P33 | fig02 is n = 4 | fig02 shows 20 variables | rewording |
| P34 | QUBO-Ising verified n = 4-20, rel. error < 1e-10 | Max relative error 2.7e-13, n = 4-20 (fig04) | supported |
| P35 | fig06 all ratio 1.0; fig09 noisy QAOA | SA/greedy miss the optimum often (toy, execution families); fig09 has no QAOA | contradicted |
| P36 | fig07: SA > 1000x faster, both optimal | SA 3.7 ms vs ideal QAOA 1.5-2.9 s (400-800x); SA optimal in 1 of 5 seeds on that instance | contradicted |
| P37 | COBYLA reliably identifies the optimum | Uniform sampling with the same shots identifies it at least as reliably (100% vs 88-100%) | contradicted |
| P38 | fig19 reflects microstructure-venue interactions | Single-venue `ExecutionQUBO`; fig19 now shows raw and repaired QUBO schedules | rewording |
| P39 | Dark-pool share rises with toxicity | VPIN still constant in fig27 | still unsupported |
| P40 | fig18 x-axis is risk aversion | Axis is an impact-weight multiplier (now labelled so) | rewording |
| P41 | 38 figures | 32 registered producers (2 illustrative); 6 referenced PDFs removed (`figures/registry.py`, `REMOVED`) | rewording |
| P42 | Other "first" claims | Literature claim | rewording |
| P43 | All benchmark data available | Every simulator/market result and its plotting code is committed with provenance; hardware raw counts are still missing | partly supported |
| P44 | Calibrated synthetic data | Uncalibrated simulator | rewording |

**Totals (44):** supported **1** (P34) · partly supported **2** (P15, P43) · contradicted **16** (P4-P6, P9-P14, P25-P27, P29, P35-P37) · still unsupported **13** (P1, P2, P7, P8, P17-P22, P24, P28, P39) · rewording **12** (P3, P16, P23, P30-P33, P38, P40-P42, P44).

### Flagged items resolved in Phase 6
F2 (saturating best-of-shots): probability on the optimal set, <H> ratio and a same-budget uniform baseline are now reported. F6 (unfair comparison): one engine, one book, no fills without volume, opportunity cost for unfilled shares. F9: walk-forward seeded and repeated over 30 seeds. F11: every figure is drawn from `results/` by a registered function. T5 (underfill after a policy switch): the fast path re-plans the remaining shares. `run_integrated_comparison` ranks by shortfall including opportunity cost. F5 (assumed improvement prior) remains and is documented; F10 remains until IBM raw counts are recovered.
