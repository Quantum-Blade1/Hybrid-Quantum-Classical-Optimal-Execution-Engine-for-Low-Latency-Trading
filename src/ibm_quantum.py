"""
IBM Quantum Hardware Integration

This module provides integration with IBM Quantum hardware for running
QAOA on real quantum devices. Includes:

1. IBM Quantum account setup
2. Backend selection and transpilation
3. Error mitigation techniques
4. Job submission and monitoring
5. Simulator vs hardware comparison

SETUP:
1. Create IBM Quantum account at https://quantum.ibm.com
2. Get API token from Account settings
3. Save token: IBMQuantumService.save_account(token="YOUR_TOKEN")
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from time import time, sleep
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class IBMQuantumConfig:
    """Configuration for IBM Quantum access."""
    channel: str = "ibm_quantum"  # or "ibm_cloud"
    instance: str = "ibm-q/open/main"  # For free tier
    max_qubits: int = 127
    default_shots: int = 4000
    optimization_level: int = 3
    resilience_level: int = 1  # Error mitigation level


# =============================================================================
# IBM Quantum Service Wrapper
# =============================================================================

class IBMQuantumService:
    """
    Wrapper for IBM Quantum services.
    
    Provides simplified interface for:
    - Account management
    - Backend selection
    - Job submission
    - Result retrieval
    """
    
    _instance = None
    _service = None
    
    def __init__(self, config: Optional[IBMQuantumConfig] = None):
        self.config = config or IBMQuantumConfig()
        self._backends_cache = None
    
    @classmethod
    def save_account(cls, token: str, overwrite: bool = True) -> None:
        """
        Save IBM Quantum account credentials.
        
        Args:
            token: IBM Quantum API token
            overwrite: Whether to overwrite existing credentials
        """
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            QiskitRuntimeService.save_account(
                channel="ibm_quantum",
                token=token,
                overwrite=overwrite
            )
            print("✓ Account saved successfully")
        except Exception as e:
            print(f"✗ Failed to save account: {e}")
    
    def connect(self) -> bool:
        """
        Connect to IBM Quantum service.
        
        Returns:
            True if connection successful
        """
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            self._service = QiskitRuntimeService(channel=self.config.channel)
            logger.info("Connected to IBM Quantum")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    @property
    def is_connected(self) -> bool:
        return self._service is not None
    
    def list_backends(self, min_qubits: int = 0, operational: bool = True) -> List[Dict]:
        """
        List available quantum backends.
        
        Args:
            min_qubits: Minimum number of qubits required
            operational: Only show operational backends
            
        Returns:
            List of backend info dicts
        """
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        
        backends = []
        for backend in self._service.backends():
            config = backend.configuration()
            status = backend.status()
            
            if config.n_qubits < min_qubits:
                continue
            if operational and not status.operational:
                continue
            
            backends.append({
                'name': backend.name,
                'qubits': config.n_qubits,
                'operational': status.operational,
                'pending_jobs': status.pending_jobs,
                'simulator': config.simulator
            })
        
        return sorted(backends, key=lambda x: x['qubits'], reverse=True)
    
    def get_backend(self, name: Optional[str] = None, min_qubits: int = 4):
        """
        Get a quantum backend.
        
        Args:
            name: Specific backend name, or None for least busy
            min_qubits: Minimum qubits if selecting automatically
            
        Returns:
            Backend instance
        """
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        
        if name:
            return self._service.backend(name)
        
        # Get least busy backend with enough qubits
        from qiskit_ibm_runtime import least_busy
        backends = self._service.backends(
            filters=lambda x: x.configuration().n_qubits >= min_qubits 
                             and x.status().operational
        )
        return least_busy(backends)


# =============================================================================
# Hardware QAOA Runner
# =============================================================================

@dataclass
class HardwareResult:
    """Result from hardware execution."""
    solution: np.ndarray
    energy: float
    counts: Dict[str, int]
    success_probability: float
    execution_time: float
    backend_name: str
    is_simulator: bool
    job_id: str
    transpiled_depth: int


class HardwareQAOA:
    """
    QAOA solver for IBM Quantum hardware.
    
    Includes:
    - Device-aware transpilation
    - Error mitigation
    - Job monitoring
    """
    
    def __init__(
        self,
        service: IBMQuantumService,
        backend_name: Optional[str] = None,
        shots: int = 4000,
        optimization_level: int = 3,
        resilience_level: int = 1
    ):
        self.service = service
        self.backend_name = backend_name
        self.shots = shots
        self.optimization_level = optimization_level
        self.resilience_level = resilience_level
        
        self._backend = None
    
    def run(
        self,
        Q: np.ndarray,
        gamma: float,
        beta: float,
        p: int = 1,
        verbose: bool = True
    ) -> HardwareResult:
        """
        Run QAOA on hardware.
        
        Args:
            Q: QUBO matrix
            gamma: Cost layer parameter
            beta: Mixer layer parameter
            p: Number of QAOA layers
            verbose: Print progress
            
        Returns:
            HardwareResult with solution and metrics
        """
        from qiskit import transpile
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        from qiskit_ibm_runtime import Session
        
        # Import local converters
        from .qubo_to_ising import qubo_to_ising, build_qaoa_circuit_from_ising
        
        n = Q.shape[0]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f" Hardware QAOA (n={n}, p={p})")
            print(f"{'='*60}")
        
        # Get backend
        self._backend = self.service.get_backend(
            name=self.backend_name,
            min_qubits=n
        )
        
        if verbose:
            print(f" Backend: {self._backend.name}")
            print(f" Qubits: {self._backend.configuration().n_qubits}")
        
        # Convert QUBO to Ising and build circuit
        ising = qubo_to_ising(Q)
        qc = build_qaoa_circuit_from_ising(ising, gamma, beta, p)
        
        # Remove measurements for Sampler
        qc.remove_final_measurements()
        
        # Transpile for device
        if verbose:
            print(f" Transpiling for device topology...")
        
        start_time = time()
        
        transpiled = transpile(
            qc,
            backend=self._backend,
            optimization_level=self.optimization_level
        )
        
        transpiled_depth = transpiled.depth()
        if verbose:
            print(f" Transpiled depth: {transpiled_depth}")
        
        # Run with Sampler
        if verbose:
            print(f" Submitting job ({self.shots} shots)...")
        
        with Session(backend=self._backend) as session:
            sampler = Sampler(session=session)
            
            # Set options for error mitigation
            sampler.options.resilience_level = self.resilience_level
            
            job = sampler.run([transpiled], shots=self.shots)
            job_id = job.job_id()
            
            if verbose:
                print(f" Job ID: {job_id}")
                print(f" Waiting for results...")
            
            result = job.result()
        
        execution_time = time() - start_time
        
        # Process results
        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()
        
        # Find best solution
        best_bitstring = None
        best_energy = float('inf')
        
        for bitstring, count in counts.items():
            binary = np.array([int(b) for b in bitstring[::-1]])
            energy = binary @ Q @ binary
            if energy < best_energy:
                best_energy = energy
                best_bitstring = bitstring
        
        best_solution = np.array([int(b) for b in best_bitstring[::-1]])
        
        # Success probability
        total_counts = sum(counts.values())
        success_count = counts.get(best_bitstring, 0)
        success_prob = success_count / total_counts
        
        if verbose:
            print(f"\n Results:")
            print(f"   Best energy: {best_energy:.4f}")
            print(f"   Success prob: {success_prob:.1%}")
            print(f"   Time: {execution_time:.1f}s")
        
        return HardwareResult(
            solution=best_solution,
            energy=best_energy,
            counts=counts,
            success_probability=success_prob,
            execution_time=execution_time,
            backend_name=self._backend.name,
            is_simulator=self._backend.configuration().simulator,
            job_id=job_id,
            transpiled_depth=transpiled_depth
        )


# =============================================================================
# Simulator vs Hardware Comparison
# =============================================================================

@dataclass
class ComparisonResult:
    """Result comparing simulator vs hardware."""
    simulator_result: HardwareResult
    hardware_result: Optional[HardwareResult]
    qubo_size: int
    optimal_energy: Optional[float]
    
    @property
    def simulator_gap(self) -> float:
        """Gap from optimal for simulator."""
        if self.optimal_energy is None:
            return 0.0
        return self.simulator_result.energy - self.optimal_energy
    
    @property
    def hardware_gap(self) -> float:
        """Gap from optimal for hardware."""
        if self.hardware_result is None or self.optimal_energy is None:
            return float('inf')
        return self.hardware_result.energy - self.optimal_energy


def compare_simulator_vs_hardware(
    Q: np.ndarray,
    gamma: float,
    beta: float,
    service: Optional[IBMQuantumService] = None,
    shots: int = 4000,
    verbose: bool = True
) -> ComparisonResult:
    """
    Compare QAOA results on simulator vs real hardware.
    
    Args:
        Q: QUBO matrix
        gamma: Cost parameter
        beta: Mixer parameter
        service: IBM Quantum service (optional, for hardware)
        shots: Number of shots
        verbose: Print progress
        
    Returns:
        ComparisonResult with both results
    """
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    
    from .qubo_to_ising import qubo_to_ising, build_qaoa_circuit_from_ising
    
    n = Q.shape[0]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f" Simulator vs Hardware Comparison")
        print(f"{'='*60}")
        print(f" QUBO size: {n} qubits")
    
    # Build circuit
    ising = qubo_to_ising(Q)
    qc = build_qaoa_circuit_from_ising(ising, gamma, beta, p=1)
    
    # =========================================================================
    # Run on simulator
    # =========================================================================
    if verbose:
        print(f"\n Running on simulator...")
    
    sim_start = time()
    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    sim_result = simulator.run(compiled, shots=shots).result()
    sim_counts = sim_result.get_counts()
    sim_time = time() - sim_start
    
    # Find best simulator solution
    sim_best_bitstring = None
    sim_best_energy = float('inf')
    
    for bitstring, count in sim_counts.items():
        binary = np.array([int(b) for b in bitstring[::-1]])
        energy = binary @ Q @ binary
        if energy < sim_best_energy:
            sim_best_energy = energy
            sim_best_bitstring = bitstring
    
    sim_solution = np.array([int(b) for b in sim_best_bitstring[::-1]])
    sim_success_prob = sim_counts.get(sim_best_bitstring, 0) / sum(sim_counts.values())
    
    simulator_result = HardwareResult(
        solution=sim_solution,
        energy=sim_best_energy,
        counts=sim_counts,
        success_probability=sim_success_prob,
        execution_time=sim_time,
        backend_name="aer_simulator",
        is_simulator=True,
        job_id="local",
        transpiled_depth=compiled.depth()
    )
    
    if verbose:
        print(f"   Energy: {sim_best_energy:.4f}")
        print(f"   Success prob: {sim_success_prob:.1%}")
        print(f"   Time: {sim_time:.2f}s")
    
    # =========================================================================
    # Run on hardware (if service provided)
    # =========================================================================
    hardware_result = None
    
    if service is not None and service.is_connected:
        if verbose:
            print(f"\n Running on hardware...")
        
        try:
            hw_qaoa = HardwareQAOA(
                service=service,
                shots=shots,
                optimization_level=3,
                resilience_level=1
            )
            hardware_result = hw_qaoa.run(Q, gamma, beta, p=1, verbose=verbose)
        except Exception as e:
            logger.warning(f"Hardware execution failed: {e}")
            if verbose:
                print(f"   ⚠ Hardware execution failed: {e}")
    else:
        if verbose:
            print(f"\n Hardware: Skipped (no service connected)")
    
    # =========================================================================
    # Find optimal (brute force for small problems)
    # =========================================================================
    optimal_energy = None
    if n <= 20:
        if verbose:
            print(f"\n Finding optimal solution (brute force)...")
        
        optimal_energy = float('inf')
        for i in range(2**n):
            x = np.array([(i >> j) & 1 for j in range(n)])
            e = x @ Q @ x
            optimal_energy = min(optimal_energy, e)
        
        if verbose:
            print(f"   Optimal energy: {optimal_energy:.4f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        print(f"\n{'='*60}")
        print(f" Summary")
        print(f"{'='*60}")
        print(f" {'Metric':<25} {'Simulator':>15} {'Hardware':>15}")
        print(f" {'-'*55}")
        
        hw_energy = hardware_result.energy if hardware_result else "N/A"
        hw_prob = f"{hardware_result.success_probability:.1%}" if hardware_result else "N/A"
        hw_time = f"{hardware_result.execution_time:.1f}s" if hardware_result else "N/A"
        
        print(f" {'Best Energy':<25} {sim_best_energy:>15.4f} {str(hw_energy):>15}")
        print(f" {'Success Probability':<25} {sim_success_prob:>14.1%} {str(hw_prob):>15}")
        print(f" {'Execution Time':<25} {sim_time:>14.2f}s {str(hw_time):>15}")
        
        if optimal_energy is not None:
            sim_gap = sim_best_energy - optimal_energy
            hw_gap = (hardware_result.energy - optimal_energy) if hardware_result else float('inf')
            print(f" {'Gap from Optimal':<25} {sim_gap:>15.4f} {hw_gap:>15.4f}")
    
    return ComparisonResult(
        simulator_result=simulator_result,
        hardware_result=hardware_result,
        qubo_size=n,
        optimal_energy=optimal_energy
    )


# =============================================================================
# Demo
# =============================================================================

def run_ibm_demo(use_hardware: bool = False):
    """
    Demo IBM Quantum integration.
    
    Args:
        use_hardware: If True, attempt to run on real hardware
    """
    from .qaoa_tutorial import Graph
    
    print("\n" + "="*70)
    print(" IBM Quantum Hardware Integration Demo")
    print("="*70)
    
    # Create small test problem (Max-Cut on 4-node graph)
    graph = Graph.create_square()
    n = graph.num_vertices
    
    # Build QUBO manually for Max-Cut
    Q = np.zeros((n, n))
    for i, j in graph.edges:
        Q[i, i] -= 1
        Q[j, j] -= 1
        Q[i, j] += 1
        Q[j, i] += 1
    
    print(f"\n Problem: Max-Cut on 4-node graph")
    print(f" QUBO size: {n} qubits")
    
    # Parameters (pre-optimized)
    gamma = 0.5
    beta = 0.3
    
    # Connect to IBM Quantum (if attempting hardware)
    service = None
    if use_hardware:
        print("\n Connecting to IBM Quantum...")
        service = IBMQuantumService()
        if service.connect():
            backends = service.list_backends(min_qubits=n)
            print(f" Available backends: {len(backends)}")
            for b in backends[:3]:
                print(f"   - {b['name']} ({b['qubits']} qubits)")
        else:
            print(" ⚠ Could not connect. Running simulator only.")
            service = None
    
    # Run comparison
    result = compare_simulator_vs_hardware(
        Q=Q,
        gamma=gamma,
        beta=beta,
        service=service,
        shots=4000,
        verbose=True
    )
    
    # Document limitations
    print("\n" + "="*70)
    print(" Limitations & Observations")
    print("="*70)
    print("""
 NOISE IMPACT:
   - Real hardware introduces decoherence and gate errors
   - Success probability typically 5-20% lower than simulator
   - Deep circuits (high p) suffer more from noise
   
 CONNECTIVITY:
   - Hardware has limited qubit connectivity (not all-to-all)
   - Transpilation adds SWAP gates, increasing depth
   - Can 2-3x circuit depth depending on topology
   
 QUEUE TIMES:
   - Free tier: 10+ minute wait typical
   - Premium access: <1 minute
   - Job monitoring essential for production
   
 ERROR MITIGATION:
   - Resilience level 1: Zero-noise extrapolation
   - Resilience level 2: More aggressive mitigation
   - Higher levels increase classical post-processing cost
   
 RECOMMENDATIONS:
   1. Optimize parameters on simulator first
   2. Use smallest possible circuit (low p)
   3. Enable error mitigation for real hardware
   4. Consider hybrid approaches for larger problems
""")
    
    return result


if __name__ == "__main__":
    # Run demo (simulator only by default)
    result = run_ibm_demo(use_hardware=False)
