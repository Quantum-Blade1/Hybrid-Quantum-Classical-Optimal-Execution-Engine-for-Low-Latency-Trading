"""
QUBO to Ising Converter

============================================================================
CONVERSION THEORY
============================================================================

QUBO (Quadratic Unconstrained Binary Optimization):
    min x^T Q x    where x_i ∈ {0, 1}

Ising Model (for quantum computers):
    H = Σ_i h_i Z_i + Σ_{i<j} J_{ij} Z_i Z_j + offset
    
    where Z_i eigenvalues are ∈ {-1, +1}

TRANSFORMATION:
    x_i = (1 - z_i) / 2
    
    When z_i = +1 → x_i = 0
    When z_i = -1 → x_i = 1

DERIVATION:
    x_i = (1 - z_i) / 2
    x_i² = x_i (for binary)
    x_i x_j = (1 - z_i)(1 - z_j) / 4
            = (1 - z_i - z_j + z_i z_j) / 4

Substituting into x^T Q x and collecting terms:
    
    h_i = -Q_{ii}/2 - Σ_{j≠i} Q_{ij}/4
    J_{ij} = Q_{ij}/4
    offset = Σ_i Q_{ii}/2 + Σ_{i<j} Q_{ij}/4

============================================================================
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


# =============================================================================
# Ising Hamiltonian Representation
# =============================================================================

@dataclass
class IsingHamiltonian:
    """
    Ising model Hamiltonian representation.
    
    H = Σ_i h_i Z_i + Σ_{i<j} J_{ij} Z_i Z_j + offset
    
    Attributes:
        h: Linear coefficients (on-site fields), shape (n,)
        J: Quadratic coefficients (interactions), shape (n, n), upper triangular
        offset: Constant energy offset
        num_qubits: Number of qubits (spins)
    """
    h: np.ndarray
    J: np.ndarray
    offset: float
    num_qubits: int
    
    def evaluate(self, spins: np.ndarray) -> float:
        """
        Evaluate Hamiltonian energy for a given spin configuration.
        
        H = Σ_i h_i z_i + Σ_{i<j} J_{ij} z_i z_j + offset
        
        Args:
            spins: Array of +1/-1 values, shape (n,)
            
        Returns:
            Energy value
        """
        linear_term = np.dot(self.h, spins)
        
        # J is upper triangular, so we compute Σ_{i<j} J_ij * z_i * z_j
        quadratic_term = 0.0
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                quadratic_term += self.J[i, j] * spins[i] * spins[j]
        
        return linear_term + quadratic_term + self.offset
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format for some solvers."""
        result = {'offset': self.offset}
        
        # Linear terms
        for i in range(self.num_qubits):
            if abs(self.h[i]) > 1e-10:
                result[f'Z{i}'] = float(self.h[i])
        
        # Quadratic terms
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                if abs(self.J[i, j]) > 1e-10:
                    result[f'Z{i}Z{j}'] = float(self.J[i, j])
        
        return result
    
    def __repr__(self) -> str:
        num_linear = np.sum(np.abs(self.h) > 1e-10)
        num_quadratic = np.sum(np.abs(self.J) > 1e-10) // 2
        return (f"IsingHamiltonian(n={self.num_qubits}, "
                f"linear_terms={num_linear}, quadratic_terms={num_quadratic}, "
                f"offset={self.offset:.4f})")


# =============================================================================
# QUBO to Ising Conversion
# =============================================================================

def qubo_to_ising(Q: np.ndarray) -> IsingHamiltonian:
    """
    Convert QUBO matrix to Ising Hamiltonian.
    
    QUBO: min x^T Q x,  x_i ∈ {0, 1}
    Ising: H = Σ h_i Z_i + Σ_{i<j} J_{ij} Z_i Z_j + offset
    
    Transformation: x_i = (1 - z_i) / 2
    
    Derivation for symmetric Q:
        QUBO = Σ_i Q_ii * x_i + Σ_{i<j} 2*Q_ij * x_i * x_j
        
        Substituting x_i = (1-z_i)/2:
        - Q_ii * x_i = Q_ii * (1-z_i)/2 = Q_ii/2 - Q_ii/2 * z_i
        - 2*Q_ij * x_i * x_j = 2*Q_ij * (1-z_i)/2 * (1-z_j)/2
                             = Q_ij/2 * (1 - z_i - z_j + z_i*z_j)
        
    Collecting terms:
        offset = Σ_i Q_ii/2 + Σ_{i<j} Q_ij/2
        h_i = -Q_ii/2 - Σ_{j<i} Q_ji/2 - Σ_{j>i} Q_ij/2 = -Q_ii/2 - Σ_{j≠i} Q_ij/2
        J_ij = Q_ij/2 (for symmetric Q where Q_ij = Q_ji)
    
    Args:
        Q: QUBO matrix, shape (n, n), assumed symmetric
        
    Returns:
        IsingHamiltonian with h, J, offset
    """
    n = Q.shape[0]
    
    # Make Q symmetric (average with transpose)
    Q_sym = (Q + Q.T) / 2
    
    # Initialize Ising coefficients
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0
    
    # =========================================================================
    # Calculate offset
    # offset = Σ_i Q_ii/2 + Σ_{i<j} Q_ij/2
    # =========================================================================
    diag_sum = np.trace(Q_sym)
    # Upper triangular off-diagonal sum (for symmetric Q)
    upper_sum = np.sum(np.triu(Q_sym, k=1))
    offset = diag_sum / 2 + upper_sum / 2
    
    # =========================================================================
    # Calculate linear coefficients h_i
    # h_i = -Q_ii/2 - Σ_{j≠i} Q_ij/2
    #     = -Q_ii/2 - (row_sum - Q_ii)/2
    #     = -row_sum/2
    # =========================================================================
    for i in range(n):
        row_sum = np.sum(Q_sym[i, :])
        h[i] = -row_sum / 2
    
    # =========================================================================
    # Calculate quadratic coefficients J_ij
    # J_ij = Q_ij / 2 (for i < j only, upper triangular)
    # =========================================================================
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = Q_sym[i, j] / 2
    
    return IsingHamiltonian(
        h=h,
        J=J,
        offset=offset,
        num_qubits=n
    )


def ising_to_qubo(ising: IsingHamiltonian) -> Tuple[np.ndarray, float]:
    """
    Convert Ising Hamiltonian back to QUBO.
    
    Inverse of qubo_to_ising.
    
    Args:
        ising: Ising Hamiltonian
        
    Returns:
        Tuple of (Q matrix, constant offset)
    """
    n = ising.num_qubits
    Q = np.zeros((n, n))
    
    # Reconstruct from J_ij = Q_ij / 4
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = 4 * ising.J[i, j]
            Q[j, i] = Q[i, j]  # Symmetric
    
    # Reconstruct diagonal from h_i = -Σ_j Q_ij / 4
    # This is underdetermined - we need to solve a system
    # For now, assume diagonal dominates
    for i in range(n):
        off_diag_sum = np.sum(Q[i, :]) - Q[i, i]
        Q[i, i] = -4 * ising.h[i] - off_diag_sum
    
    return Q, ising.offset


# =============================================================================
# Qiskit Operator Construction
# =============================================================================

def ising_to_qiskit_operator(ising: IsingHamiltonian):
    """
    Convert Ising Hamiltonian to Qiskit SparsePauliOp.
    
    H = Σ_i h_i Z_i + Σ_{i<j} J_{ij} Z_i Z_j + offset * I
    
    Args:
        ising: Ising Hamiltonian
        
    Returns:
        qiskit.quantum_info.SparsePauliOp
    """
    from qiskit.quantum_info import SparsePauliOp
    
    n = ising.num_qubits
    pauli_list = []
    coeffs = []
    
    # Identity term (offset)
    if abs(ising.offset) > 1e-10:
        pauli_list.append('I' * n)
        coeffs.append(ising.offset)
    
    # Linear terms: h_i Z_i
    for i in range(n):
        if abs(ising.h[i]) > 1e-10:
            # Build Pauli string: Z on qubit i, I elsewhere
            # Qiskit uses little-endian: rightmost = qubit 0
            pauli = ['I'] * n
            pauli[n - 1 - i] = 'Z'  # Adjust for Qiskit ordering
            pauli_list.append(''.join(pauli))
            coeffs.append(ising.h[i])
    
    # Quadratic terms: J_{ij} Z_i Z_j
    for i in range(n):
        for j in range(i + 1, n):
            if abs(ising.J[i, j]) > 1e-10:
                pauli = ['I'] * n
                pauli[n - 1 - i] = 'Z'
                pauli[n - 1 - j] = 'Z'
                pauli_list.append(''.join(pauli))
                coeffs.append(ising.J[i, j])
    
    if not pauli_list:
        # Empty Hamiltonian - return identity
        return SparsePauliOp(['I' * n], [0.0])
    
    return SparsePauliOp(pauli_list, coeffs)


def build_qaoa_circuit_from_ising(
    ising: IsingHamiltonian,
    gamma: float,
    beta: float,
    p: int = 1
):
    """
    Build QAOA circuit from Ising Hamiltonian.
    
    Circuit structure:
    1. Initial |+⟩^n state
    2. For each layer:
       a) Cost unitary: e^{-iγ H_C} using RZ and RZZ gates
       b) Mixer unitary: e^{-iβ Σ X_i} using RX gates
    3. Measurement
    
    Args:
        ising: Ising Hamiltonian (cost function)
        gamma: Cost layer parameter
        beta: Mixer layer parameter  
        p: Number of QAOA layers
        
    Returns:
        Qiskit QuantumCircuit
    """
    from qiskit import QuantumCircuit
    
    n = ising.num_qubits
    qc = QuantumCircuit(n, n)
    
    # Initial superposition
    for i in range(n):
        qc.h(i)
    
    qc.barrier()
    
    # QAOA layers
    for layer in range(p):
        g = gamma[layer] if isinstance(gamma, (list, np.ndarray)) else gamma
        b = beta[layer] if isinstance(beta, (list, np.ndarray)) else beta
        
        # -----------------------------------------------------------------
        # Cost layer: e^{-iγ H_C}
        # 
        # For H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j
        # 
        # e^{-iγ h_i Z_i} = RZ(2 * γ * h_i)
        # e^{-iγ J_ij Z_i Z_j} = RZZ(2 * γ * J_ij)
        # -----------------------------------------------------------------
        
        # Linear terms: RZ gates
        for i in range(n):
            if abs(ising.h[i]) > 1e-10:
                qc.rz(2 * g * ising.h[i], i)
        
        # Quadratic terms: RZZ gates
        for i in range(n):
            for j in range(i + 1, n):
                if abs(ising.J[i, j]) > 1e-10:
                    qc.rzz(2 * g * ising.J[i, j], i, j)
        
        qc.barrier()
        
        # -----------------------------------------------------------------
        # Mixer layer: e^{-iβ Σ X_i}
        # e^{-iβ X_i} = RX(2β)
        # -----------------------------------------------------------------
        for i in range(n):
            qc.rx(2 * b, i)
        
        qc.barrier()
    
    # Measurement
    qc.measure(range(n), range(n))
    
    return qc


# =============================================================================
# Solution Conversion
# =============================================================================

def spins_to_binary(spins: np.ndarray) -> np.ndarray:
    """
    Convert spin configuration to binary.
    
    x_i = (1 - z_i) / 2
    z = +1 → x = 0
    z = -1 → x = 1
    """
    return ((1 - spins) / 2).astype(int)


def binary_to_spins(binary: np.ndarray) -> np.ndarray:
    """
    Convert binary configuration to spins.
    
    z_i = 1 - 2*x_i
    x = 0 → z = +1
    x = 1 → z = -1
    """
    return (1 - 2 * binary).astype(int)


def bitstring_to_spins(bitstring: str) -> np.ndarray:
    """
    Convert measurement bitstring to spin array.
    
    Note: Qiskit returns bitstrings in little-endian order.
    """
    # Reverse to get qubit 0 first
    binary = np.array([int(b) for b in bitstring[::-1]])
    return binary_to_spins(binary)


# =============================================================================
# Verification Utilities
# =============================================================================

def verify_conversion(Q: np.ndarray, num_samples: int = 100) -> bool:
    """
    Verify QUBO to Ising conversion is correct.
    
    Checks that QUBO(x) = Ising(z) where z = 1 - 2x
    
    Args:
        Q: QUBO matrix
        num_samples: Number of random solutions to test
        
    Returns:
        True if conversion is verified
    """
    n = Q.shape[0]
    ising = qubo_to_ising(Q)
    
    for _ in range(num_samples):
        # Random binary solution
        x = np.random.randint(0, 2, n)
        
        # QUBO value
        qubo_value = x @ Q @ x
        
        # Ising value
        z = binary_to_spins(x)
        ising_value = ising.evaluate(z)
        
        # Should be equal
        if abs(qubo_value - ising_value) > 1e-6:
            print(f"Mismatch: x={x}, QUBO={qubo_value}, Ising={ising_value}")
            return False
    
    return True


def print_conversion_summary(Q: np.ndarray) -> None:
    """Print summary of QUBO to Ising conversion."""
    ising = qubo_to_ising(Q)
    
    print("\n" + "=" * 60)
    print(" QUBO to Ising Conversion Summary")
    print("=" * 60)
    print(f"\n QUBO Matrix ({Q.shape[0]} x {Q.shape[0]}):")
    print(f"   Non-zero entries: {np.count_nonzero(Q)}")
    print(f"   Diagonal sum: {np.trace(Q):.4f}")
    print(f"   Total sum: {np.sum(Q):.4f}")
    
    print(f"\n Ising Hamiltonian:")
    print(f"   Num qubits: {ising.num_qubits}")
    print(f"   Linear terms (h): {np.sum(np.abs(ising.h) > 1e-10)}")
    print(f"   Quadratic terms (J): {np.sum(np.abs(ising.J) > 1e-10)}")
    print(f"   Offset: {ising.offset:.4f}")
    
    # Verify conversion
    print(f"\n Verification: ", end="")
    if verify_conversion(Q, 100):
        print("✓ PASSED")
    else:
        print("✗ FAILED")
