from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from qexec.optimization.solvers.metrics import EnergyBounds, approximation_ratio


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be in [0, trials]")
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    phat = successes / trials
    denom = 1 + z**2 / trials
    centre = (phat + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(phat * (1 - phat) / trials + z**2 / (4 * trials**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def binomial_test_greater(successes: int, trials: int, p0: float) -> float:
    return float(stats.binomtest(successes, trials, p0, alternative="greater").pvalue)


def binomial_test_less(successes: int, trials: int, p0: float) -> float:
    return float(stats.binomtest(successes, trials, p0, alternative="less").pvalue)


def sign_test_greater(num_above: int, num_runs: int) -> float:
    """Across-run sign-test p-value; runs are not pooled as they differ by more than shot noise."""
    return binomial_test_greater(num_above, num_runs, 0.5)


def count_energies(counts: dict[str, int], Q: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
    """(energy, count) per distinct Qiskit bitstring (qubit 0 rightmost)."""
    n = Q.shape[0]
    energies = np.empty(len(counts))
    weights = np.empty(len(counts), dtype=np.int64)
    for k, (bitstring, count) in enumerate(counts.items()):
        x = np.array([int(b) for b in bitstring[::-1]][:n], dtype=np.float64)
        energies[k] = float(x @ Q @ x)
        weights[k] = int(count)
    return energies, weights


@dataclass(frozen=True)
class RatioAdvantage:
    """Approximation ratio of the mean energy minus the uniform value, with a bootstrap CI."""

    ratio: float
    uniform_ratio: float
    difference: float
    ci_low: float
    ci_high: float
    resamples: int


def bootstrap_ratio_advantage(
    energies: NDArray[np.float64],
    weights: NDArray[np.int64],
    bounds: EnergyBounds,
    uniform_ratio: float,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    rng: np.random.Generator,
) -> RatioAdvantage:
    """Shot-level multinomial bootstrap of the mean-energy ratio minus the uniform value."""
    total = int(weights.sum())
    if total <= 0:
        raise ValueError("empty counts")
    probs = weights / total
    draws = rng.multinomial(total, probs, size=resamples)
    means = draws @ energies / total
    span = bounds.max_energy - bounds.min_energy
    ratios = (bounds.max_energy - means) / span if span > 0 else np.ones_like(means)
    alpha = 1 - confidence
    low, high = np.quantile(ratios - uniform_ratio, [alpha / 2, 1 - alpha / 2])
    ratio = approximation_ratio(float(probs @ energies), bounds)
    return RatioAdvantage(
        ratio=ratio,
        uniform_ratio=uniform_ratio,
        difference=ratio - uniform_ratio,
        ci_low=float(low),
        ci_high=float(high),
        resamples=resamples,
    )
