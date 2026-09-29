"""Solution-quality metrics for QUBO solvers that are valid for signed energies."""

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
    """Exact energy range by exhaustive enumeration.

    The extremal assignments are re-evaluated as x @ Q @ x, the same expression the
    solvers use, so an optimal solver result compares equal to `min_energy`.
    """
    n = Q.shape[0]
    energies = enumerate_energies(Q)
    x_min = _bits(int(np.argmin(energies)), n)
    x_max = _bits(int(np.argmax(energies)), n)
    return EnergyBounds(min_energy=float(x_min @ Q @ x_min), max_energy=float(x_max @ Q @ x_max))


def approximation_ratio(energy: float, bounds: EnergyBounds) -> float:
    """r = (E_max - E) / (E_max - E_min): 1 at the optimum, 0 at the worst assignment.

    Unlike E_min / E, this is monotone in E for energies of either sign.
    """
    span = bounds.max_energy - bounds.min_energy
    if span <= 0:
        return 1.0
    return (bounds.max_energy - energy) / span


def optimality_gap(energy: float, bounds: EnergyBounds) -> float:
    """Relative gap (E - E_min) / |E_min|; the absolute gap E - E_min when E_min = 0."""
    gap = energy - bounds.min_energy
    scale = abs(bounds.min_energy)
    return gap / scale if scale > 0 else gap
