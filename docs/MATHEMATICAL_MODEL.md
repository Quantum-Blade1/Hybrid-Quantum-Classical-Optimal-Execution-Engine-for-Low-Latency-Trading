# 🧠 The Mathematics of Quantum Trading
**A Deep Dive from Intuition to Rigorous Proofs**

This document explains the "engine under the hood" of the Hybrid Quantum-Classical Trading System. It is structured in three levels of difficulty.

---

## 🟢 Level 1: The Intuition (Easy)

### 1. The "Pizza Slicing" Problem
Imagine you need to buy 50,000 shares of Apple (AAPL).
*   **The Problem:** If you buy it all at once, the price "explodes" upwards because you consume all available sellers (Liquidity). This is called **Market Impact**.
*   **The Solution:** You slice the order into smaller pieces over time (e.g., 1,000 shares every minute).
*   **The Risk:** If you wait too long to buy the last slice, the price might move against you naturally. This is **Timing Risk**.

**The Goal:** Find the "Sweet Spot" schedule that balances *Impact* (trading too fast) vs. *Risk* (trading too slow).

### 2. The "Energy Landscape"
Think of every possible trading schedule as a spot on a map.
*   **Bad Schedules** (High Cost) are "Mountains".
*   **Good Schedules** (Low Cost) are "Valleys".
*   **The Quantum advantage:** A classical computer (like a hiker) has to walk down the hill and might get stuck in a small valley (local minimum). A Quantum computer can "tunnel" through the mountain to find the deepest valley (Global Minimum) instantly.

---

## 🟡 Level 2: The Model (Medium)

### 1. The Objective Function
We describe the "Cost" of a unified schedule $\mathbf{x}$ as a sum of three parts:

$$ C(\mathbf{x}) = \text{Transaction Cost} + \text{Market Impact} + \text{Timing Risk} $$

### 2. QUBO Formulation
Quantum computers don't solve algebra; they solve **QUBOs** (Quadratic Unconstrained Binary Optimization). We translate our problem into a matrix equation:

$$ \min_{\mathbf{x}} \left( \mathbf{x}^T Q \mathbf{x} + \mathbf{c}^T \mathbf{x} \right) $$

*   $\mathbf{x}$: A vector of 0s and 1s representing decisions (e.g., $x_{i}$ = "Buy 100 shares at time $i$").
*   $Q$ (The Matrix): Describes the relationship between decisions.
    *   **Diagonals** ($Q_{ii}$): The cost of doing action $i$ alone.
    *   **Off-Diagonals** ($Q_{ij}$): The "friction" or "synergy" of doing action $i$ AND action $j$ together.

### 3. Constraints as Penalties
We accept that Quantum computers "don't like rules" (constraints).
*   **Rule:** "You must buy exactly 50,000 shares."
*   **Quantum Translation:** "I will fine you \$1,000,000 for every share you miss."
This converts a *Hard Constraint* into a *Soft Penalty* in the energy function.

---

## 🔴 Level 3: The Theory (Hard)

### 1. Almgren-Chriss Market Impact Model (1999)
We derive our $Q$ matrix terms from the foundational Almgren-Chriss framework.
The expected cost of execution $E[x]$ is:

$$ E[x] = \sum_{t=1}^T \tau \left( \epsilon \text{sgn}(n_t) + \eta \frac{n_t}{\tau} \right) $$

*   $\epsilon$: Fixed bid-ask spread cost.
*   $\eta$: Market impact coefficient (Permanent vs Temporary).
*   $n_t$: Shares executed at time $t$.

**Adaptation for QUBO:**
Since QUBO variables $x$ are binary, we discretize $n_t$ into levels $q_k$.
The quadratic variance term (Risk) becomes fundamental:
$$ V[x] = \sigma^2 \sum_{t=1}^T \tau \left( \sum_{j=1}^t n_j \right)^2 $$
This squared term $\left( \sum n_j \right)^2$ creates the dense **Off-Diagonal** elements in our $Q$ matrix, representing the covariance of holding inventory over time.

### 2. Deriving the Squared Penalty Term
We transform the equality constraint $\sum_{i} q_i x_i = S$ into the objective function.
Let the penalty function be $P(\mathbf{x}) = \lambda (\sum q_i x_i - S)^2$.
Expanding this:
$$ (\sum q_i x_i - S)^2 = (\sum q_i x_i)^2 - 2S(\sum q_i x_i) + S^2 $$
$$ = \sum_{i} q_i^2 x_i^2 + \sum_{i \neq j} q_i q_j x_i x_j - 2S \sum q_i x_i + \text{const} $$

Since $x_i$ is binary ($x_i^2 = x_i$):
*   **Linear Terms** (Diagonal update): $Q_{ii} \leftarrow Q_{ii} + \lambda(q_i^2 - 2Sq_i)$
*   **Quadratic Terms** (Off-Diagonal update): $Q_{ij} \leftarrow Q_{ij} + 2\lambda q_i q_j$

This proves why "Global Constraints" result in "All-to-All Connectivity" in the qubit graph ($\sum_{i \neq j} x_i x_j$).

### 3. QAOA: The Quantum Approximate Optimization Algorithm
We solve this Ising Hamiltonian $H_C$ by applying a time-dependent evolution:
$$ |\psi(\gamma, \beta)\rangle = e^{-i\beta H_B} e^{-i\gamma H_C} \dots e^{-i\beta H_B} e^{-i\gamma H_C} |+\rangle^{\otimes n} $$
*   $H_C$: The Cost Hamiltonian (Our $Q$ matrix encoded as Pauli-Z operators).
*   $H_B$: The Mixer Hamiltonian (Transverse field $\sum \sigma_x$).
*   **Adiabatic Theorem:** As $p \to \infty$, if we evolve slowly enough from the ground state of $H_B$ to $H_C$, we are guaranteed to find the optimum.
*   **NISQ Reality:** We use finite $p$ (depth) and optimize angles $(\gamma, \beta)$ classically (COBYLA) to approximate this evolution.

### 4. Complexity Analysis
*   **Classical Brute Force:** $O(2^N)$. Impossible for $N > 50$.
*   **Simulated Annealing:** Heuristic. $O(e^{k})$. Can get stuck in local minima.
*   **Quantum Annealing:** $O(e^{k/\sqrt{width}})$. Theoretically tunnels through barriers that are "tall but thin".

---

## 🧪 Evaluation Model (what the experiments actually compute)

This section documents the modelling decisions behind every number in `results/` (Phase 6). Where it disagrees with the intuition sections above, this section describes the code.

### 1. Execution simulation (one fill model for every strategy)
All strategy comparisons (`experiments/{is_comparison,strategy_comparison,stress_test,walk_forward}.py`) execute through `qexec.execution.engine.ExecutionEngine`:

* **Book.** For minute $k$ with mid $m_k$, spread $s_k$ and bar volume $V_k$, a synthetic book is generated with 10 levels per side, one tick apart, level sizes $\max(10,\ \lfloor b_k\,0.7^i\,\xi_{k,i}\rfloor)$ with $b_k=\max(100,\ 1000\,V_k/10^5)$ and $\xi_{k,i}\sim\mathrm{LogNormal}(0,0.5)$. A marketable child order walks this book. This is the only impact model; it is temporary (each bar's book is fresh, there is no permanent impact and no decay kernel).
* **No liquidity, no fill.** A bar with $V_k=0$ (e.g. the stress "market outage") has an empty book and fills nothing. Before Phase 6 the hybrid runner filled at $m_k+s_k/2$ in such bars, which with the outage's \$1000 spread produced a 6,185 bps "slippage" (audit F6).
* **Carry forward.** Shares a child order does not fill are added to the next minute's target. Shares still unfilled after the last bar are *not* dropped: they are charged as opportunity cost (below).
* **Common random numbers.** Level sizes for minute $k$ are drawn from a generator seeded by $(\text{seed}, k)$. Every strategy run on the same market path with the same seed faces the identical book in every minute, so paired differences between strategies measure the schedule, not the book noise.
* **Participation.** VWAP keeps its own cap (10% of forecast volume per minute); the other schedules are limited only by the book depth. No strategy sees future bars: VWAP uses the simulator's *expected* volume curve (or a training-window profile in the walk-forward), and the hybrid's re-planning receives only bars $0..k$.

### 2. Implementation shortfall (Perold 1988)
For a buy of $N$ shares with arrival mid $P_0$ (decision price = arrival price), fills $n_i$ at $p_i$ and $U$ unfilled shares:

$$\mathrm{IS} = \underbrace{\sum_i n_i (p_i - P_0)}_{\text{execution cost}} + \underbrace{U\,(P_c - P_0)}_{\text{opportunity cost}}, \qquad \mathrm{IS}_{\text{bps}} = 10^4\,\frac{\mathrm{IS}}{N P_0}.$$

$P_c$ is the average price of a clean-up market order for the $U$ shares against the *final* bar's book (shares beyond the book's depth at its deepest level; the far touch $P_T+s_T/2$ if the final bar has no volume). A strategy that underfills is therefore charged the spread and the impact of completing its remainder at once and cannot look cheaper by not trading. The execution cost splits exactly into half-spread $\sum n_i(a_i-m_i)$, impact $\sum n_i(p_i-a_i)$ (walking past the touch $a_i$) and timing $\sum n_i(m_i-P_0)$. `StrategyComparison.best_strategy` ranks by $\mathrm{IS}$ (it used to rank by spread + impact on filled shares only).

### 3. Schedules
* **Repair.** QUBO solutions select discrete quantity levels ($\{0, \lfloor N/2T\rfloor, \lfloor N/T\rfloor\}$), so decoded schedules can miss the order (4,997 of 5,000) or overshoot it off the equality constraint. `repair_schedule` rescales proportionally and applies largest-remainder rounding so the integer schedule sums to $N$ exactly; an all-zero schedule becomes uniform. The pre-repair total is reported where relevant.
* **Async re-planning.** A policy published while an order is executing plans the whole order. At tick $t$ the fast path replaces its plan from $t$ on with the policy's schedule tail rescaled (repaired) to the shares still unexecuted, so the order completes after any number of policy switches and never overfills.
* **Hybrid.** Starts uniform; at 5 evenly spaced checkpoints the decision layer may re-solve the remaining shares with SA over the remaining minutes. Its improvement tracker is seeded with five synthetic 5% improvements (audit F5): an assumed prior, not a measurement.

### 4. Statistics
Each strategy experiment is repeated over independent seeds (default 30; 3 in `--quick` mode). A seed fixes the market path and the books, so strategies are compared *paired by seed*. Reported: mean, sample std, 95% percentile-bootstrap CI of the mean (10,000 resamples, seeded), and for each strategy vs TWAP and vs VWAP the mean paired difference with its bootstrap CI and the two-sided Wilcoxon signed-rank p-value. Walk-forward windows within one seed share a price history, so the unit of observation there is the per-seed mean over windows. No multiple-comparison correction is applied; p-values are reported as computed.

### 5. Solver quality
For a QUBO with exact bounds $E_{\min}, E_{\max}$ (exhaustive enumeration, $n\le 20$):
* approximation ratio $r(E)=(E_{\max}-E)/(E_{\max}-E_{\min})$ (valid for signed energies; $1$ = optimal, $0$ = worst);
* optimality gap $(E-E_{\min})/|E_{\min}|$;
* **success probability** $P_{\text{opt}}$: probability mass of the final QAOA distribution on the optimal set $\{x: E(x)\le E_{\min}+10^{-6}\}$, estimated from the final shots (not the frequency of the best sampled bitstring);
* $\langle H\rangle$ ratio: $r(\langle E\rangle)$ of the mean energy of the final distribution;
* best-of-shots energy, which saturates once the shot budget is comparable to $2^n$ (audit F2);
* **uniform-random baseline** with the same total shot budget as QAOA (all optimisation shots plus the final shots): its exact $P_{\text{opt}} = |\text{opt set}|/2^n$, its exact $\langle E\rangle$ (mean over all $2^n$ energies) and its best-of-shots energy from a seeded sample of that size.

### 6. Latency
Fast-path latency is the wall time of one `AsyncExecutionEngine` tick (policy poll, re-plan, execute; the sleep between ticks is excluded) and of one `HFTQuantumPipeline` tick (estimator updates + policy application), measured with `time.monotonic_ns` in CPython on the machine named in the manifest. These are Python-thread latencies, not kernel-bypass numbers.

---

## 📐 Appendix: Key Mathematical Concepts Used

### 1. Linear Algebra (The Core)
*   **Matrix Multiplication**: Used to calculate the cost energy ($x^T Q x$).
*   **Symmetric Matrices**: The $Q$ matrix must be symmetric ($Q_{ij} = Q_{ji}$) for QUBO solvers.
*   **Eigenvalues & Eigenvectors**: In Quantum Mechanics, the "optimal solution" is the eigenvector with the lowest eigenvalue (Ground State) of the Hamiltonian matrix.
*   **Hilbert Space**: The complex vector space where quantum states $|\psi\rangle$ live ($2^N$ dimensions).

### 2. Optimization Theory
*   **Combinatorial Optimization**: Solving problems where variables are discrete (0 or 1).
*   **Lagrangian Multipliers (Penalty Method)**: Converting "hard constraints" ($\sum x = S$) into "soft penalties" in the objective function.
*   **Heuristics**: Algorithms like **Simulated Annealing** that find "good enough" solutions when the perfect one takes too long.
*   **Gradient Descent**: Used in QAOA to find the optimal angles $(\beta, \gamma)$ for the quantum circuit.

### 3. Statistics & Stochastic Calculus
*   **Geometric Brownian Motion (GBM)**: The math used to simulate stock price paths ($dS_t = \mu S_t dt + \sigma S_t dW_t$).
*   **Variance & Covariance**: Used to model **Timing Risk**. The "Risk" term in the objective function basically minimizes the variance of the execution schedule.
*   **Expected Value**: We optimize for the *Expected* Implementation Shortfall.

### 4. Quantum Physics / Mechanics
*   **Ising Model**: A physics model of magnetism used to map the problem to qubits.
*   **Hamiltonians**: The total energy operator of the system.
    *   **Cost Hamiltonian ($H_C$)**: Encodes the problem.
    *   **Mixer Hamiltonian ($H_M$)**: Helps explore the solution space.
*   **Adiabatic Theorem**: The proof that if you evolve a quantum system slowly enough, it stays in its ground state (finds the answer).

### 5. Financial Mathematics
*   **Market Impact Models**: Specifically the **Square-Root Law** (Impact $\propto \sqrt{\text{Volume}}$).
*   **Almgren-Chriss Framework**: The fundamental differential equations for optimal execution strategies.

---

**Summary:**
This project is not just "coding." It is translating **Financial Theory** (Almgren-Chriss) into **Statistical Physics** (Ising Model), solving it with **Quantum Mechanics** (QAOA/Annealing), and executing it on **Software Engineering** (AsyncIO).
