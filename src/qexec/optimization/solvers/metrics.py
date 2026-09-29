from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_MAX_ENUMERATION_VARIABLES = 24
_CHUNK_BITS = 16


@dataclass(frozen=True)
class EnergyBounds:
    """Minimum (optimal) and maximum (worst) of x^T Q x over all x in {0,1}^n."""

    min_energy: float
    max_energy: float


def enumerate_energies(Q: NDArray[np.float64]) -> NDArray[np.float64]:
    """x^T Q x for every x in {0,1}^n, indexed by the integer whose bit k is x_k."""
    n = Q.shape[0]
    if n > _MAX_ENUMERATION_VARIABLES:
        raise ValueError(
            f"Exhaustive enumeration limited to {_MAX_ENUMERATION_VARIABLES} variables"
        )
    bits = np.arange(n)
    energies = np.empty(2**n)
    chunk = 2 ** min(n, _CHUNK_BITS)
    for lo in range(0, 2**n, chunk):
        idx = np.arange(lo, lo + chunk)
        X = ((idx[:, None] >> bits) & 1).astype(np.float64)
        energies[lo : lo + chunk] = np.einsum("ij,ij->i", X @ Q, X)
    return energies


def _bits(index: int, n: int) -> NDArray[np.int8]:
    return np.array([(index >> b) & 1 for b in range(n)], dtype=np.int8)


def energy_bounds(Q: NDArray[np.float64]) -> EnergyBounds:
    """Exact energy range, recomputed as x @ Q @ x like the solvers so optima compare equal."""
    n = Q.shape[0]
    energies = enumerate_energies(Q)
    x_min = _bits(int(np.argmin(energies)), n)
    x_max = _bits(int(np.argmax(energies)), n)
    return EnergyBounds(min_energy=float(x_min @ Q @ x_min), max_energy=float(x_max @ Q @ x_max))


def approximation_ratio(energy: float, bounds: EnergyBounds) -> float:
    """r = (E_max - E) / (E_max - E_min): 1 at the optimum, 0 at the worst; sign-agnostic."""
    span = bounds.max_energy - bounds.min_energy
    if span <= 0:
        return 1.0
    return (bounds.max_energy - energy) / span


def optimality_gap(energy: float, bounds: EnergyBounds) -> float:
    """Relative gap (E - E_min) / |E_min|; the absolute gap E - E_min when E_min = 0."""
    gap = energy - bounds.min_energy
    scale = abs(bounds.min_energy)
    return gap / scale if scale > 0 else gap


OPTIMUM_TOL = 1e-6


@dataclass(frozen=True)
class DistributionQuality:
    """`success_probability` is the shot fraction on any optimal bitstring, not the best sampled."""

    shots: int
    success_probability: float
    mean_energy: float
    best_energy: float


def counts_quality(
    counts: dict[str, int], Q: NDArray[np.float64], bounds: EnergyBounds
) -> DistributionQuality:
    """Quality of Qiskit-style counts (qubit 0 is the rightmost character)."""
    n = Q.shape[0]
    total = sum(counts.values())
    if total == 0:
        raise ValueError("empty counts")
    energies = np.empty(len(counts))
    weights = np.empty(len(counts))
    for k, (bitstring, count) in enumerate(counts.items()):
        x = np.array([int(b) for b in bitstring[::-1]][:n], dtype=np.float64)
        energies[k] = float(x @ Q @ x)
        weights[k] = count
    hit = energies <= bounds.min_energy + OPTIMUM_TOL
    return DistributionQuality(
        shots=int(total),
        success_probability=float(weights[hit].sum() / total),
        mean_energy=float(weights @ energies / total),
        best_energy=float(energies.min()),
    )


@dataclass(frozen=True)
class RandomBaseline:
    """Uniform sampling: exact success probability and mean energy, best of `shots` seeded draws."""

    shots: int
    success_probability: float
    mean_energy: float
    best_energy: float
    num_optimal: int


def random_sampling_baseline(
    Q: NDArray[np.float64], shots: int, rng: np.random.Generator
) -> RandomBaseline:
    energies = enumerate_energies(Q)
    n = Q.shape[0]
    min_energy = float(energies.min())
    num_optimal = int(np.sum(energies <= min_energy + OPTIMUM_TOL))
    samples = rng.integers(0, 2**n, size=shots)
    return RandomBaseline(
        shots=int(shots),
        success_probability=num_optimal / 2**n,
        mean_energy=float(energies.mean()),
        best_energy=float(energies[samples].min()),
        num_optimal=num_optimal,
    )
