# Hybrid Quantum-Classical Trading Execution System

![Status](https://img.shields.io/badge/Status-Project%20Complete-success)
![Quantum](https://img.shields.io/badge/Quantum-Ready-blueviolet)

> **🚀 [READ THIS FIRST: 10 Extreme Performance Use Cases](EXTREME_CAPABILITIES.md)**  
> *Discover how this system outperforms traditional algorithms in Flash Crashes, HFT Evasion, and Massive Block Trades.*

> **🧠 [Creator's Insights & Deep-Dive FAQ](docs/)**
> *Read the Founder's candid thoughts on the architecture and the Top 30 technical questions (with proofs).*

A research-grade hybrid architecture for optimal trade execution, combining low-latency classical execution with quantum-inspired optimization (QUBO/QAOA) to minimize implementation shortfall.

![Architecture](assets/figure1_architecture.png)

## 🚀 Key Features

*   **Hybrid Architecture**: Decoupled "Fast Path" (Execution Engine, <10ms delay) and "Slow Path" (Quantum/Classical Optimizer, 1-5s update).
*   **Asynchronous Optimization**: Trading never blocks; execution policy is updated in real-time via a thread-safe queue.
*   **Quantum Solvers**: Supports Simulated Annealing (SA), QAOA (via Qiskit), and **Distributed Quantum Computing (DQC)** backend for scaling >100 variables.
*   **Real Data Ready**: Includes `DataLoader` for NSE/Binance/NYSE tick data ingestion and backtesting.
*   **Real-Time Dashboard**: Streamlit interface for live monitoring, strategy comparison, and quantum visualization.
*   **Robustness**: Built-in resilience against solver timeouts and crashes with exponential backoff and classical fallback.
*   **Advanced Analytics**: Implementation Shortfall decomposition (Delay, Impact, Timing Risk).

## 💻 System Requirements

*   **OS**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS (M1/Intel).
*   **Python**: Version 3.9 or higher (3.10+ recommended).
*   **RAM**: Minimum 8GB (16GB recommended for heavy simulations).
*   **CPU**: Multi-core processor (Hybrid engine is multi-threaded).
*   **Optional**:
    *   **GPU**: For Qiskit Aer GPU support (CUDA 11+).
    *   **IBMQ Account**: For running on real Quantum Hardware.

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/quantum-trading-engine.git
cd quantum-trading-engine

# Install dependencies (requires Python 3.9+)
pip install -r requirements.txt
```

## ⚡ Quick Start

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

## 🏗️ System Architecture

The system operates on two timescales:
1.  **Fast Loop (100ms)**: The `ExecutionEngine` consumes tick data and executes orders based on the current *Execution Policy*.
2.  **Slow Loop (1s+)**: The `HybridController` aggregates market state, formulates a QUBO problem, solves it (SA/Quantum), and pushes a new *Execution Policy*.

## 📊 Benchmarks & Results

| Metric | VWAP | TWAP | Hybrid (Quantum) |
| :--- | :--- | :--- | :--- |
| **Market Impact** | High | Medium | **Low** |
| **Timing Risk** | Low | Low | **Low** |
| **Robustness** | Low | High | **High** |
| **Avg Shortfall** | +15 bps | +8 bps | **+5 bps** |

*See `results_paper.md` for detailed research findings.*

## 📂 Project Structure

```
c:\Engine\
├── src/               # Core Source Code
│   ├── hybrid_async.py
│   ├── qubo_execution.py
│   └── ...
├── examples/          # Demo Scripts
│   ├── hybrid_demo.py
│   ├── solver_demo.py
│   └── ...
├── docs/              # Documentation & Founder's Notes
│   ├── CREATOR_THOUGHTS.md
│   └── MATHEMATICAL_MODEL.md
├── assets/            # Images & Figures
├── tests/             # Unit Tests
├── results_paper.md   # Research Summary
└── requirements.txt
```

## 📜 License
Research Use Only.
