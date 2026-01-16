# User Guide: Hybrid Quantum-Classical Trading System

This guide walks you through setting up and using the trading engine.

## 1. Prerequisites
- Python 3.9 or higher
- Git (for cloning)
- A modern web browser (for Dashboard)

## 2. Installation
1.  **Navigate to project folder**:
    ```bash
    cd c:\Engine
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 3. Using the Interactive Dashboard
The dashboard is the best way to visualize the system.

1.  **Run the App**:
    ```bash
    streamlit run src/dashboard.py
    ```
2.  **Interface Overview**:
    - **Sidebar**: Configure symbol (AAPL), Quantity, and choose Strategy (Hybrid vs Adaptive VWAP).
    - **Control**: Click "▶ START SIMULATION" to begin real-time trading.
    - **Tab 1 (Monitor)**: Watch price updates and fill progress.
    - **Tab 3 (Quantum)**: See the latest "Heatmap" of the optimized schedule.

## 4. Running Experiments

### Quick Demo
To see a text-based output of a single execution 50k share order:
```bash
python hybrid_demo.py
```

### Walk-Forward Analysis (Research Mode)
To reproduce the research paper results comparing Adaptive vs Hybrid:
```bash
python -m src.walk_forward
```
*Output: `walk_forward_results.png` & console table.*

### Hardware Benchmarking
To compare CPU vs Simulated Quantum performance:
```bash
python -m src.benchmark
```

## 5. Troubleshooting
- **Optimization Timeout**: If the "Quantum Panel" freezes, the solver might be timing out. Check console logs. The system is designed to auto-fallback to classical TWAP.
- **Missing Dependencies**: Ensure `streamlit` and `plotly` are installed (`pip install streamlit plotly`).

---
*For development details, see `README.md`.*
