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
*   **DQC (Distributed):** We decompose the graph into sub-QUBOs. Solving large $Q$ becomes equivalent to solving $k$ sub-problems of size $N/k$ plus a recombination step (Lagrangian Relaxation).

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
