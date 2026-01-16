"""
Quantum Error Mitigation Module

Production-ready error mitigation techniques for quantum execution:

1. Zero-Noise Extrapolation (ZNE)
2. Measurement Error Mitigation
3. Dynamic Decoupling
4. Majority Voting

All techniques include proper error handling and logging.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# =============================================================================
# Error Mitigation Results
# =============================================================================

@dataclass
class MitigationResult:
    """Result from error-mitigated execution."""
    raw_counts: Dict[str, int]
    mitigated_counts: Dict[str, int]
    raw_solution: np.ndarray
    mitigated_solution: np.ndarray
    raw_energy: float
    mitigated_energy: float
    improvement: float  # Energy improvement from mitigation
    technique: str
    metadata: Dict = field(default_factory=dict)
    
    @property
    def improvement_percent(self) -> float:
        if abs(self.raw_energy) < 1e-10:
            return 0.0
        return (self.raw_energy - self.mitigated_energy) / abs(self.raw_energy) * 100


# =============================================================================
# Base Error Mitigator
# =============================================================================

class ErrorMitigator(ABC):
    """Abstract base class for error mitigation techniques."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def mitigate(
        self,
        counts: Dict[str, int],
        Q: np.ndarray,
        **kwargs
    ) -> MitigationResult:
        """Apply error mitigation to measurement counts."""
        pass


# =============================================================================
# 1. Zero-Noise Extrapolation (ZNE)
# =============================================================================

class ZeroNoiseExtrapolation(ErrorMitigator):
    """
    Zero-Noise Extrapolation for error mitigation.
    
    Runs circuits at multiple noise levels and extrapolates to zero noise.
    
    In practice:
    - Scale noise by inserting identity gates (pulse stretching)
    - Run at noise factors [1.0, 1.5, 2.0, ...]
    - Fit polynomial/exponential and extrapolate to factor=0
    """
    
    def __init__(
        self,
        noise_factors: List[float] = [1.0, 1.5, 2.0],
        extrapolation: str = 'linear'  # 'linear', 'polynomial', 'exponential'
    ):
        self.noise_factors = noise_factors
        self.extrapolation = extrapolation
    
    @property
    def name(self) -> str:
        return f"ZNE-{self.extrapolation}"
    
    def mitigate(
        self,
        counts: Dict[str, int],
        Q: np.ndarray,
        noise_scaled_counts: Optional[List[Dict[str, int]]] = None,
        **kwargs
    ) -> MitigationResult:
        """
        Apply ZNE to measurement counts.
        
        Args:
            counts: Baseline counts (noise_factor=1.0)
            Q: QUBO matrix for energy calculation
            noise_scaled_counts: Counts from scaled noise circuits
        """
        n = Q.shape[0]
        
        # Calculate raw solution
        raw_solution, raw_energy = self._best_from_counts(counts, Q)
        
        if noise_scaled_counts is None or len(noise_scaled_counts) < 2:
            # No scaled data - return raw with warning
            logger.warning("ZNE requires multiple noise levels. Returning raw.")
            return MitigationResult(
                raw_counts=counts,
                mitigated_counts=counts,
                raw_solution=raw_solution,
                mitigated_solution=raw_solution,
                raw_energy=raw_energy,
                mitigated_energy=raw_energy,
                improvement=0.0,
                technique=self.name,
                metadata={'warning': 'No noise-scaled data provided'}
            )
        
        # Calculate expected value at each noise level
        all_counts = [counts] + noise_scaled_counts
        expected_values = []
        
        for c in all_counts:
            total = sum(c.values())
            exp_val = 0.0
            for bitstring, count in c.items():
                x = np.array([int(b) for b in bitstring[::-1]])
                if len(x) == n:
                    exp_val += (x @ Q @ x) * count / total
            expected_values.append(exp_val)
        
        # Extrapolate to zero noise
        factors = self.noise_factors[:len(expected_values)]
        mitigated_energy = self._extrapolate(factors, expected_values)
        
        # Mitigated counts: reweight based on extrapolation
        mitigated_counts = self._reweight_counts(counts, Q, mitigated_energy)
        mit_solution, _ = self._best_from_counts(mitigated_counts, Q)
        
        return MitigationResult(
            raw_counts=counts,
            mitigated_counts=mitigated_counts,
            raw_solution=raw_solution,
            mitigated_solution=mit_solution,
            raw_energy=raw_energy,
            mitigated_energy=mitigated_energy,
            improvement=raw_energy - mitigated_energy,
            technique=self.name,
            metadata={
                'noise_factors': factors,
                'expected_values': expected_values
            }
        )
    
    def _best_from_counts(
        self,
        counts: Dict[str, int],
        Q: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Find best solution from counts."""
        n = Q.shape[0]
        best_x = None
        best_e = float('inf')
        
        for bitstring in counts:
            x = np.array([int(b) for b in bitstring[::-1]])
            if len(x) == n:
                e = x @ Q @ x
                if e < best_e:
                    best_e = e
                    best_x = x
        
        return best_x if best_x is not None else np.zeros(n), best_e
    
    def _extrapolate(
        self,
        factors: List[float],
        values: List[float]
    ) -> float:
        """Extrapolate to zero noise."""
        if self.extrapolation == 'linear':
            # Linear fit: y = a*x + b
            coeffs = np.polyfit(factors, values, 1)
            return coeffs[1]  # y-intercept (x=0)
        
        elif self.extrapolation == 'polynomial':
            # Quadratic fit
            degree = min(2, len(factors) - 1)
            coeffs = np.polyfit(factors, values, degree)
            return np.polyval(coeffs, 0)
        
        else:  # exponential
            # Exponential fit: y = a * exp(b*x)
            log_vals = np.log(np.abs(np.array(values)) + 1e-10)
            coeffs = np.polyfit(factors, log_vals, 1)
            return np.exp(coeffs[1])
    
    def _reweight_counts(
        self,
        counts: Dict[str, int],
        Q: np.ndarray,
        target_energy: float
    ) -> Dict[str, int]:
        """Reweight counts to match target energy."""
        n = Q.shape[0]
        
        # Score each bitstring by energy
        scored = []
        for bitstring, count in counts.items():
            x = np.array([int(b) for b in bitstring[::-1]])
            if len(x) == n:
                e = x @ Q @ x
                scored.append((bitstring, count, e))
        
        # Sort by energy (lower is better)
        scored.sort(key=lambda x: x[2])
        
        # Reweight: boost low-energy states
        total = sum(c for _, c, _ in scored)
        mitigated = {}
        
        for bitstring, count, energy in scored:
            # Weight inversely proportional to energy gap from target
            gap = abs(energy - target_energy)
            weight = 1.0 / (gap + 1.0)
            mitigated[bitstring] = int(count * weight * 2)
        
        return mitigated


# =============================================================================
# 2. Measurement Error Mitigation
# =============================================================================

class MeasurementErrorMitigator(ErrorMitigator):
    """
    Measurement error mitigation using confusion matrix.
    
    Learns measurement errors from calibration circuits and applies
    inverse transformation to results.
    """
    
    def __init__(self, confusion_matrix: Optional[np.ndarray] = None):
        self._confusion_matrix = confusion_matrix
        self._calibrated = confusion_matrix is not None
    
    @property
    def name(self) -> str:
        return "MeasurementError"
    
    def calibrate(self, calibration_counts: List[Dict[str, int]], n_qubits: int) -> None:
        """
        Calibrate from preparation circuits.
        
        Args:
            calibration_counts: Counts from preparing |00...0⟩, |00...1⟩, etc.
            n_qubits: Number of qubits
        """
        n_states = 2 ** n_qubits
        self._confusion_matrix = np.zeros((n_states, n_states))
        
        for prepared_state, counts in enumerate(calibration_counts):
            total = sum(counts.values())
            for measured, count in counts.items():
                measured_idx = int(measured[::-1], 2) if len(measured) == n_qubits else 0
                if measured_idx < n_states:
                    self._confusion_matrix[prepared_state, measured_idx] = count / total
        
        self._calibrated = True
        logger.info(f"MEM calibrated for {n_qubits} qubits")
    
    def mitigate(
        self,
        counts: Dict[str, int],
        Q: np.ndarray,
        **kwargs
    ) -> MitigationResult:
        """Apply measurement error mitigation."""
        n = Q.shape[0]
        
        raw_solution, raw_energy = self._best_from_counts(counts, Q)
        
        if not self._calibrated:
            # Generate approximate confusion matrix from noise model
            logger.warning("Using approximate confusion matrix")
            self._approximate_calibration(n)
        
        # Convert counts to probability vector
        n_states = 2 ** n
        prob_vector = np.zeros(n_states)
        total = sum(counts.values())
        
        for bitstring, count in counts.items():
            if len(bitstring) == n:
                idx = int(bitstring[::-1], 2)
                if idx < n_states:
                    prob_vector[idx] = count / total
        
        # Apply inverse confusion matrix (pseudo-inverse for stability)
        try:
            inv_matrix = np.linalg.pinv(self._confusion_matrix)
            mitigated_probs = inv_matrix @ prob_vector
            
            # Ensure non-negative and normalized
            mitigated_probs = np.maximum(mitigated_probs, 0)
            mitigated_probs /= mitigated_probs.sum() + 1e-10
        except Exception as e:
            logger.error(f"MEM inversion failed: {e}")
            mitigated_probs = prob_vector
        
        # Convert back to counts
        mitigated_counts = {}
        for idx, prob in enumerate(mitigated_probs):
            if prob > 0.001:
                bitstring = format(idx, f'0{n}b')[::-1]
                mitigated_counts[bitstring] = int(prob * total)
        
        mit_solution, mit_energy = self._best_from_counts(mitigated_counts, Q)
        
        return MitigationResult(
            raw_counts=counts,
            mitigated_counts=mitigated_counts,
            raw_solution=raw_solution,
            mitigated_solution=mit_solution,
            raw_energy=raw_energy,
            mitigated_energy=mit_energy,
            improvement=raw_energy - mit_energy,
            technique=self.name
        )
    
    def _best_from_counts(self, counts: Dict[str, int], Q: np.ndarray):
        n = Q.shape[0]
        best_x, best_e = np.zeros(n), float('inf')
        for bs in counts:
            x = np.array([int(b) for b in bs[::-1]])
            if len(x) == n:
                e = x @ Q @ x
                if e < best_e:
                    best_e, best_x = e, x
        return best_x, best_e
    
    def _approximate_calibration(self, n_qubits: int):
        """Generate approximate confusion matrix assuming independent errors."""
        # Typical readout error ~1-3%
        error_rate = 0.02
        n_states = 2 ** n_qubits
        
        self._confusion_matrix = np.zeros((n_states, n_states))
        
        for i in range(n_states):
            for j in range(n_states):
                # Count bit flips
                diff = bin(i ^ j).count('1')
                prob = ((1 - error_rate) ** (n_qubits - diff)) * (error_rate ** diff)
                self._confusion_matrix[i, j] = prob
        
        # Normalize rows
        for i in range(n_states):
            self._confusion_matrix[i] /= self._confusion_matrix[i].sum()
        
        self._calibrated = True


# =============================================================================
# 3. Dynamic Decoupling
# =============================================================================

class DynamicDecoupling:
    """
    Dynamic decoupling pulse insertion.
    
    Adds DD sequences during idle periods to suppress decoherence.
    Common sequences: X-X, Y-Y, XY4, CPMG
    """
    
    SEQUENCES = {
        'XX': ['x', 'x'],
        'YY': ['y', 'y'],
        'XY4': ['x', 'y', 'x', 'y'],
        'CPMG': ['y', 'y'],
    }
    
    def __init__(self, sequence: str = 'XY4'):
        if sequence not in self.SEQUENCES:
            raise ValueError(f"Unknown DD sequence: {sequence}")
        self.sequence = sequence
        self.pulses = self.SEQUENCES[sequence]
    
    def add_to_circuit(self, circuit, idle_qubits: List[int] = None):
        """
        Add DD sequence to circuit during idle periods.
        
        Args:
            circuit: Qiskit QuantumCircuit
            idle_qubits: Qubits to apply DD to (None = all)
            
        Returns:
            Modified circuit with DD
        """
        from qiskit import QuantumCircuit
        
        if idle_qubits is None:
            idle_qubits = list(range(circuit.num_qubits))
        
        # Note: Full DD requires timing analysis
        # This is a simplified version
        dd_circuit = circuit.copy()
        
        # For demonstration, add DD at barriers
        for qubit in idle_qubits:
            for pulse in self.pulses:
                if pulse == 'x':
                    dd_circuit.x(qubit)
                    dd_circuit.x(qubit)  # X-X = I
                elif pulse == 'y':
                    dd_circuit.y(qubit)
                    dd_circuit.y(qubit)  # Y-Y = I
        
        return dd_circuit


# =============================================================================
# 4. Majority Voting
# =============================================================================

class MajorityVoting(ErrorMitigator):
    """
    Majority voting across multiple execution runs.
    
    Runs circuit multiple times and aggregates results to reduce
    statistical fluctuations and occasional errors.
    """
    
    def __init__(self, num_runs: int = 5, voting_method: str = 'weighted'):
        """
        Args:
            num_runs: Number of independent runs to aggregate
            voting_method: 'simple' (count votes) or 'weighted' (by energy)
        """
        self.num_runs = num_runs
        self.voting_method = voting_method
    
    @property
    def name(self) -> str:
        return f"MajorityVote-{self.voting_method}"
    
    def mitigate(
        self,
        counts: Dict[str, int],
        Q: np.ndarray,
        additional_counts: Optional[List[Dict[str, int]]] = None,
        **kwargs
    ) -> MitigationResult:
        """
        Apply majority voting to aggregate counts.
        
        Args:
            counts: Primary run counts
            Q: QUBO matrix
            additional_counts: Counts from additional runs
        """
        n = Q.shape[0]
        
        # Combine all counts
        all_counts = [counts] + (additional_counts or [])
        
        raw_solution, raw_energy = self._best_from_counts(counts, Q)
        
        if self.voting_method == 'simple':
            mitigated_counts = self._simple_vote(all_counts)
        else:
            mitigated_counts = self._weighted_vote(all_counts, Q)
        
        mit_solution, mit_energy = self._best_from_counts(mitigated_counts, Q)
        
        return MitigationResult(
            raw_counts=counts,
            mitigated_counts=mitigated_counts,
            raw_solution=raw_solution,
            mitigated_solution=mit_solution,
            raw_energy=raw_energy,
            mitigated_energy=mit_energy,
            improvement=raw_energy - mit_energy,
            technique=self.name,
            metadata={'num_runs': len(all_counts)}
        )
    
    def _simple_vote(self, all_counts: List[Dict[str, int]]) -> Dict[str, int]:
        """Simple aggregation: sum all counts."""
        aggregated = Counter()
        for counts in all_counts:
            for bitstring, count in counts.items():
                aggregated[bitstring] += count
        return dict(aggregated)
    
    def _weighted_vote(
        self,
        all_counts: List[Dict[str, int]],
        Q: np.ndarray
    ) -> Dict[str, int]:
        """Energy-weighted aggregation."""
        n = Q.shape[0]
        aggregated = Counter()
        
        for counts in all_counts:
            for bitstring, count in counts.items():
                x = np.array([int(b) for b in bitstring[::-1]])
                if len(x) == n:
                    energy = x @ Q @ x
                    # Weight by inverse energy (lower = better)
                    weight = 1.0 / (abs(energy) + 1.0)
                    aggregated[bitstring] += int(count * weight)
        
        return dict(aggregated)
    
    def _best_from_counts(self, counts: Dict[str, int], Q: np.ndarray):
        n = Q.shape[0]
        best_x, best_e = np.zeros(n), float('inf')
        for bs in counts:
            x = np.array([int(b) for b in bs[::-1]])
            if len(x) == n:
                e = x @ Q @ x
                if e < best_e:
                    best_e, best_x = e, x
        return best_x, best_e


# =============================================================================
# Combined Error Mitigation Pipeline
# =============================================================================

class ErrorMitigationPipeline:
    """
    Pipeline combining multiple error mitigation techniques.
    
    Applies techniques in sequence and tracks improvement at each stage.
    """
    
    def __init__(self, techniques: Optional[List[ErrorMitigator]] = None):
        self.techniques = techniques or [
            MeasurementErrorMitigator(),
            MajorityVoting(num_runs=3, voting_method='weighted'),
        ]
    
    def run(
        self,
        counts: Dict[str, int],
        Q: np.ndarray,
        additional_data: Optional[Dict] = None
    ) -> List[MitigationResult]:
        """
        Run full mitigation pipeline.
        
        Args:
            counts: Raw measurement counts
            Q: QUBO matrix
            additional_data: Extra data for specific techniques
            
        Returns:
            List of results from each stage
        """
        results = []
        current_counts = counts
        
        for technique in self.techniques:
            try:
                result = technique.mitigate(
                    current_counts,
                    Q,
                    **(additional_data or {})
                )
                results.append(result)
                current_counts = result.mitigated_counts
                
                logger.info(
                    f"{technique.name}: improvement={result.improvement:.4f}"
                )
            except Exception as e:
                logger.error(f"{technique.name} failed: {e}")
        
        return results
    
    def summary(self, results: List[MitigationResult]) -> Dict:
        """Generate summary of mitigation pipeline."""
        if not results:
            return {}
        
        return {
            'initial_energy': results[0].raw_energy,
            'final_energy': results[-1].mitigated_energy,
            'total_improvement': results[0].raw_energy - results[-1].mitigated_energy,
            'techniques_applied': [r.technique for r in results],
            'stage_improvements': [r.improvement for r in results]
        }


# =============================================================================
# Demo
# =============================================================================

def run_mitigation_demo():
    """Demonstrate error mitigation techniques."""
    from .qaoa_tutorial import Graph
    
    print("\n" + "="*70)
    print(" Quantum Error Mitigation Demo")
    print("="*70)
    
    # Create test problem
    graph = Graph.create_square()
    n = graph.num_vertices
    
    Q = np.zeros((n, n))
    for i, j in graph.edges:
        Q[i, i] -= 1
        Q[j, j] -= 1
        Q[i, j] += 1
        Q[j, i] += 1
    
    print(f"\n Problem: Max-Cut on {n}-node graph")
    
    # Simulate noisy counts
    np.random.seed(42)
    ideal_counts = {'0101': 300, '1010': 300, '0110': 50, '1001': 50}
    
    # Add noise
    noisy_counts = {}
    for bitstring, count in ideal_counts.items():
        noisy_counts[bitstring] = int(count * 0.7)
        for flip in [1, 2]:
            noisy_bs = bitstring[:flip-1] + str(1-int(bitstring[flip-1])) + bitstring[flip:]
            noisy_counts[noisy_bs] = noisy_counts.get(noisy_bs, 0) + int(count * 0.1)
    
    print(f"\n Simulated noisy counts: {len(noisy_counts)} unique states")
    
    # Test each technique
    techniques = [
        MeasurementErrorMitigator(),
        MajorityVoting(num_runs=3),
        ZeroNoiseExtrapolation(),
    ]
    
    print(f"\n Testing mitigation techniques:")
    print("-" * 50)
    
    for tech in techniques:
        result = tech.mitigate(noisy_counts, Q)
        print(f" {tech.name:25} Raw={result.raw_energy:8.2f}  "
              f"Mit={result.mitigated_energy:8.2f}  "
              f"Δ={result.improvement:+.2f}")
    
    # Run pipeline
    print(f"\n Running full pipeline:")
    pipeline = ErrorMitigationPipeline()
    results = pipeline.run(noisy_counts, Q)
    summary = pipeline.summary(results)
    
    print(f" Initial energy: {summary['initial_energy']:.2f}")
    print(f" Final energy: {summary['final_energy']:.2f}")
    print(f" Total improvement: {summary['total_improvement']:.2f}")
    
    return results


if __name__ == "__main__":
    results = run_mitigation_demo()
