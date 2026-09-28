# Hybrid Quantum-Classical Trading Execution System

![Status](https://img.shields.io/badge/Status-Project%20Complete-success)
![Quantum](https://img.shields.io/badge/Quantum-Ready-blueviolet)

A research-grade hybrid architecture for optimal trade execution, combining low-latency classical execution with quantum-inspired optimization (QUBO/QAOA) to minimize implementation shortfall.

![Architecture](assets/figure1_architecture.png)

##  Key Features

*   **Hybrid Architecture**: Decoupled "Fast Path" (Execution Engine, <10ms delay) and "Slow Path" (Quantum/Classical Optimizer, 1-5s update).
*   **Asynchronous Optimization**: Trading never blocks; execution policy is updated in real-time via a thread-safe queue.
*   **Quantum Solvers**: Supports Simulated Annealing (SA) and QAOA (via Qiskit).
*   **Real Data Ready**: Includes `DataLoader` for NSE/Binance/NYSE tick data ingestion and backtesting.
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

Dependencies are declared in `pyproject.toml`. Run the scripts below from the repository root.

##  Quick Start

### 1. Interactive Dashboard (Recommended)
Launch the real-time control center:
```bash
streamlit run src/dashboard.py
```
*Features: Live P&L, Execution Trajectory, Quantum Heatmap, TWAP Benchmark.*

### 2. Headless Demo
Run a standard execution simulation:
```bash
python examples/hybrid_demo.py
```

### 3. Walk-Forward Analysis
Run the rolling window backtest:
```bash
python -m src.walk_forward
```

##  System Architecture

The system operates on two timescales:
1.  **Fast Loop (100ms)**: The `ExecutionEngine` consumes tick data and executes orders based on the current *Execution Policy*.
2.  **Slow Loop (1s+)**: The `HybridController` aggregates market state, formulates a QUBO problem, solves it (SA/Quantum), and pushes a new *Execution Policy*.

##  Benchmarks & Results

| Metric | VWAP | TWAP | Hybrid (Quantum) |
| :--- | :--- | :--- | :--- |
| **Market Impact** | High | Medium | **Low** |
| **Timing Risk** | Low | Low | **Low** |
| **Robustness** | Low | High | **High** |

*Qualitative comparison only. See `docs/CLAIMS_AUDIT.md` for the status of every quantitative claim.*

##  Project Structure

```
Hybrid-Quantum-Classical-Optimal-Execution-Engine-for-Low-Latency-Trading/
├── src/               # Core Source Code
│   ├── hybrid_async.py
│   ├── qubo_execution.py
│   └── ...
├── examples/          # Demo Scripts
│   ├── hybrid_demo.py
│   ├── solver_demo.py
│   └── ...
├── docs/              # Documentation
│   ├── MATHEMATICAL_MODEL.md
│   └── CLAIMS_AUDIT.md
├── assets/            # README images
├── figures/           # Paper figures and benchmark JSON
├── paper/             # Manuscript source (main.tex)
├── tests/             # Unit Tests
└── pyproject.toml     # Project metadata and dependencies
```

##  License
Apache License 2.0. See `LICENSE`.
