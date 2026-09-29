from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

DEFAULT_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95

Statistic = Callable[[NDArray[np.float64]], float]


def _as_array(values: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("need at least one value")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite")
    return arr


def bootstrap_ci(
    values: ArrayLike,
    *,
    statistic: Statistic = np.mean,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for `statistic`; collapses to the value for constant input."""
    arr = _as_array(values)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if arr.size == 1 or np.all(arr == arr[0]):
        value = float(statistic(arr))
        return value, value
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    resampled = arr[idx]
    if statistic is np.mean:
        boot = resampled.mean(axis=1)
    else:
        boot = np.array([statistic(row) for row in resampled])
    alpha = (1 - confidence) / 2
    low, high = np.quantile(boot, [alpha, 1 - alpha])
    return float(low), float(high)


@dataclass(frozen=True)
class Summary:
    """Mean, sample std (ddof=1) and a bootstrap CI of the mean."""

    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float
    median: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}n": self.n,
            f"{prefix}mean": self.mean,
            f"{prefix}std": self.std,
            f"{prefix}ci_low": self.ci_low,
            f"{prefix}ci_high": self.ci_high,
            f"{prefix}median": self.median,
        }


def summarize(
    values: ArrayLike, *, confidence: float = DEFAULT_CONFIDENCE, seed: int = 0
) -> Summary:
    arr = _as_array(values)
    low, high = bootstrap_ci(arr, confidence=confidence, seed=seed)
    return Summary(
        n=int(arr.size),
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        ci_low=low,
        ci_high=high,
        median=float(np.median(arr)),
    )


@dataclass(frozen=True)
class PairedComparison:
    """Paired differences a - b; for costs, a negative mean difference means `a` is cheaper."""

    n: int
    mean_diff: float
    std_diff: float
    ci_low: float
    ci_high: float
    median_diff: float
    wilcoxon_p: float
    frac_a_lower: float

    @property
    def significant(self) -> bool:
        """Bootstrap CI of the mean difference excludes zero."""
        return self.ci_low > 0 or self.ci_high < 0

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "n": self.n,
            "mean_diff": self.mean_diff,
            "std_diff": self.std_diff,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "median_diff": self.median_diff,
            "wilcoxon_p": self.wilcoxon_p,
            "frac_a_lower": self.frac_a_lower,
            "ci_excludes_zero": self.significant,
        }


def paired_comparison(
    a: ArrayLike, b: ArrayLike, *, confidence: float = DEFAULT_CONFIDENCE, seed: int = 0
) -> PairedComparison:
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    if arr_a.shape != arr_b.shape:
        raise ValueError("paired samples must have the same length")
    diff = arr_a - arr_b
    low, high = bootstrap_ci(diff, confidence=confidence, seed=seed)
    # Wilcoxon is undefined when every difference is zero; there is no evidence either way.
    all_zero = bool(np.all(diff == 0))
    p_value = 1.0 if all_zero else float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)
    return PairedComparison(
        n=int(diff.size),
        mean_diff=float(diff.mean()),
        std_diff=float(diff.std(ddof=1)) if diff.size > 1 else 0.0,
        ci_low=low,
        ci_high=high,
        median_diff=float(np.median(diff)),
        wilcoxon_p=p_value,
        frac_a_lower=float(np.mean(diff < 0)),
    )


def holm_adjust(p_values: ArrayLike) -> NDArray[np.float64]:
    """Holm (1979) step-down adjusted p-values; reject H_i at alpha iff adjusted p <= alpha."""
    p = np.asarray(p_values, dtype=np.float64).ravel()
    if p.size == 0:
        return p
    if np.any((p < 0) | (p > 1)) or not np.all(np.isfinite(p)):
        raise ValueError("p-values must be in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    stepped = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(stepped, 1.0)
    return adjusted


def cluster_bootstrap_ci(
    values: ArrayLike,
    clusters: ArrayLike,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI of the mean resampling whole clusters (e.g. days) with replacement."""
    arr = _as_array(values)
    labels = np.asarray(clusters).ravel()
    if labels.shape != arr.shape:
        raise ValueError("values and clusters must have the same length")
    uniques, codes = np.unique(labels, return_inverse=True)
    k = uniques.size
    sums = np.bincount(codes, weights=arr, minlength=k)
    counts = np.bincount(codes, minlength=k).astype(np.float64)
    if k == 1:
        value = float(arr.mean())
        return value, value
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_resamples, k))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(boot, [alpha, 1 - alpha])
    return float(low), float(high)
