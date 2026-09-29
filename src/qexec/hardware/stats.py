"""Tests and intervals for one hardware job's final distribution against uniform sampling.

Per job, the shots are treated as independent draws from the job's output distribution
(the usual shot-noise model; drift within a job is not modelled):

* success probability: Wilson score interval, and a one-sided exact binomial test of
  H0: P(optimum) <= p0 against P(optimum) > p0, where p0 is the exact uniform mass on the
  optimal bitstrings (num_optimal / 2^n; 2^-n for a unique optimum);
* mean energy: shot-level (multinomial) bootstrap percentile interval of the approximation
  ratio of the mean energy minus the exact uniform value (a constant).

These statements are about one job. Runs differ by far more than shot noise (different
COBYLA starting angles and hardware drift), so runs are not pooled here; with 3 runs the
only across-run statement is a sign count (`sign_test_greater`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from qexec.optimization.solvers.metrics import EnergyBounds, approximation_ratio


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
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
    """One-sided exact binomial p-value for P(success) > p0."""
    return float(stats.binomtest(successes, trials, p0, alternative="greater").pvalue)


def binomial_test_less(successes: int, trials: int, p0: float) -> float:
    """One-sided exact binomial p-value for P(success) < p0."""
    return float(stats.binomtest(successes, trials, p0, alternative="less").pvalue)


def sign_test_greater(num_above: int, num_runs: int) -> float:
    """One-sided sign-test p-value: P(at least `num_above` of `num_runs` runs above the
    baseline) if each run were equally likely to fall above or below it. With 3 runs the
    smallest attainable value is 1/8."""
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
    """Shot-level bootstrap: resample the job's shots (multinomial on its empirical
    distribution) and recompute the mean-energy approximation ratio."""
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
