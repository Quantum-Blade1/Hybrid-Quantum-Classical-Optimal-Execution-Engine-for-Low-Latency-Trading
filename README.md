# Hybrid Quantum-Classical Trading Execution System

![Status](https://img.shields.io/badge/Status-Project%20Complete-success)
![Quantum](https://img.shields.io/badge/Quantum-Ready-blueviolet)

A research-grade hybrid architecture for optimal trade execution, combining low-latency classical execution with quantum-inspired optimization (QUBO/QAOA) to minimize implementation shortfall.

![Architecture](assets/figure1_architecture.png)

##  Key Features

*   **Hybrid Architecture**: Decoupled "Fast Path" (tick loop) and "Slow Path" (optimizer thread) exchanging policies through a latest-value queue; measured Python latencies are in `docs/RESULTS.md`.
*   **Asynchronous Optimization**: Trading never blocks; execution policy is updated in real-time via a thread-safe queue.
*   **Quantum Solvers**: Supports Simulated Annealing (SA) and QAOA (via Qiskit).
*   **Real Data Ready**: Includes `qexec.market.loader.DataLoader` for NSE/Binance/NYSE tick data ingestion and backtesting.
*   **Real-Time Dashboard**: Streamlit interface for live monitoring, strategy comparison, and quantum visualization.
*   **Robustness**: Built-in resilience against solver timeouts and crashes with exponential backoff and classical fallback.
*   **Advanced Analytics**: Implementation Shortfall decomposition (Delay, Impact, Timing Risk).

##  System Requirements

*   **OS**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS (M1/Intel).
*   **Python**: Version 3.10 or higher.
*   **RAM**: Minimum 8GB (16GB recommended for heavy simulations).
*   **CPU**: Multi-core processor (Hybrid engine is multi-threaded).
*   **Optional**:
    *   **GPU**: For Qiskit Aer GPU support (CUDA 11+).
    *   **IBMQ Account**: For running on real Quantum Hardware.

##  Installation

```bash
# Clone repository
git clone https://github.com/Quantum-Blade1/Hybrid-Quantum-Classical-Optimal-Execution-Engine-for-Low-Latency-Trading.git
cd Hybrid-Quantum-Classical-Optimal-Execution-Engine-for-Low-Latency-Trading

# Install dependencies (requires Python 3.10+)
pip install -e ".[dev]"          # core + test/lint tools
pip install -e ".[app]"          # Streamlit dashboard (streamlit, plotly)
pip install -e ".[hardware]"     # IBM Quantum hardware runs (qiskit-ibm-runtime)
```

Dependencies are declared in `pyproject.toml`. The library is the `qexec` package under `src/`; experiment and example scripts import it, so install it first and run the scripts from the repository root (figure and result paths are relative to it).

##  Quick Start

### 1. Interactive Dashboard (Recommended)
Launch the real-time control center:
```bash
streamlit run apps/dashboard.py
```
*Features: Live P&L, Execution Trajectory, Quantum Heatmap, TWAP Benchmark.*

### 2. Examples
Short scripts showing the library API:
```bash
python examples/vwap_twap_execution.py   # classical baselines through the ExecutionEngine
python examples/qubo_solvers.py          # build the execution QUBO; exact vs SA vs greedy
python examples/qubo_vs_baselines.py     # QUBO-optimised schedule vs VWAP/TWAP
python examples/hybrid_runtime.py        # fast/slow-path runtime and the HFT pipeline
```

### 3. Reproducing the paper's numbers and figures
Every number and figure comes from an experiment script that writes machine-readable results with provenance; figures are drawn only from those results.

```bash
make all                  # every experiment (results/), then every figure (paper/figures/)
make quick                # tiny sizes -> build/quick/{results,figures}; what CI runs (~20 s)
make experiments          # python -m experiments.run_all
make figures              # python -m figures.make_figures
make check-figures        # every PDF in paper/figures/ must have a registered producer
python -m experiments.walk_forward --quick   # one experiment; --results-dir, --seed
```
Use `make PYTHON=.venv/bin/python ...` if `python` is not your environment's interpreter. The full suite takes about 15 minutes on an Apple M5 (10 cores); per-experiment wall times are in each `manifest.json`.

| Layer | Location | Contents |
|---|---|---|
| Experiments | `experiments/<name>.py` | One per result group; a `Config` dataclass with `FULL` and `QUICK` sizes; writes `results/<name>/` through `qexec.experiment.ExperimentRecorder` |
| Results | `results/<name>/` | CSV tables, JSON summaries and `manifest.json` (config, seeds, git commit + dirty flag, package versions, platform, UTC time, wall time, SHA-256 of every file) |
| Figures | `figures/` | Plot functions that read `results/` only (enforced by `tests/test_figures_registry.py`); `figures/registry.py` maps each PDF to its function, inputs and experiment |
| Summary | `docs/RESULTS.md` | Headline numbers with 95% CIs, including where the QUBO/hybrid approach does not beat the baselines |

Strategy comparisons (`is_comparison`, `strategy_comparison`, `stress_test`, `walk_forward`) run 30 seeds, execute every strategy through the same engine and book (no fills in zero-volume bars, carry-forward, opportunity cost of unfilled shares) and report bootstrap CIs and paired Wilcoxon tests against TWAP/VWAP; see `docs/MATHEMATICAL_MODEL.md`, "Evaluation Model". `experiments/hardware_benchmark.py` runs QAOA on IBM hardware when credentials are set (`IBM_QUANTUM_TOKEN`, or `IBM_CLOUD_API_KEY` + `IBM_CLOUD_CRN`); it is not part of `run_all`, and it appends every job's ID and raw counts to `results/hw_jobs.jsonl`. `results/bench_*.json` are historical records (claims audit F10) and are not read by any figure.

##  System Architecture

The system operates on two timescales:
1.  **Fast Loop (100ms)**: The `ExecutionEngine` consumes tick data and executes orders based on the current *Execution Policy*.
2.  **Slow Loop (1s+)**: The `HybridController` aggregates market state, formulates a QUBO problem, solves it (SA/Quantum), and pushes a new *Execution Policy*.

##  Benchmarks & Results

See `docs/RESULTS.md`. In short: on the synthetic market used here, the SA-QUBO and hybrid schedules do **not** reduce implementation shortfall relative to TWAP or VWAP (paired differences within a few bps, confidence intervals include zero), and QAOA on simulators does not beat uniform random sampling at finding the optimum of the benchmark QUBOs. `docs/CLAIMS_AUDIT.md` gives the status of every quantitative claim in the paper.

##  Project Structure

```
Hybrid-Quantum-Classical-Optimal-Execution-Engine-for-Low-Latency-Trading/
├── src/qexec/             # Library (import-only; no scripts)
│   ├── market/            # simulator, order_book, loader
│   ├── execution/         # engine + strategies/ (base, twap, vwap, almgren_chriss, qubo)
│   ├── optimization/      # qubo, hft_qubo, ising, schedule, toy, solvers/ (exact, annealing, greedy, qaoa)
│   ├── microstructure/    # kyle, vpin, adverse_selection, analyzer, regime
│   ├── runtime/           # policy, optimizer, engine, controller, hft_pipeline, decision, resilience, latency
│   ├── hardware/          # ibm (IBM Quantum runtime access), mitigation
│   └── analysis/          # shortfall, walk_forward, stress, runners
├── experiments/           # One script per result group -> results/<name>/ (run_all runs them all)
├── figures/               # Plot functions reading results/ only; registry.py maps PDFs to producers
├── examples/              # Short scripts demonstrating the library
├── apps/dashboard.py      # Streamlit dashboard
├── results/               # Experiment outputs with manifest.json provenance (+ historical bench_*.json)
├── paper/                 # Manuscript (main.tex) and figures/
├── docs/                  # MATHEMATICAL_MODEL.md, CLAIMS_AUDIT.md, RESULTS.md
├── Makefile               # make all | experiments | figures | quick
├── assets/                # README images
├── tests/                 # Property and regression tests, mirroring src/qexec/
└── pyproject.toml     # Project metadata and dependencies
```

## Tests

```bash
pip install -e ".[dev]"
pytest                                   # fast suite (< 60 s), excludes @pytest.mark.slow
pytest -m slow                           # QAOA runs on the Aer simulator
pytest --cov=qexec --cov-report=term-missing
```

##  License
Apache License 2.0. See `LICENSE`.
