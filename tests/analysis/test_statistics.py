"""Bootstrap intervals and paired comparisons used by the multi-seed experiments."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qexec.analysis.statistics import (
    bootstrap_ci,
    cluster_bootstrap_ci,
    holm_adjust,
    paired_comparison,
    summarize,
)

finite = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)


@given(values=st.lists(finite, min_size=2, max_size=40))
def test_bootstrap_interval_contains_the_sample_mean(values):
    low, high = bootstrap_ci(values, seed=1)
    mean = float(np.mean(values))
    assert low <= mean + 1e-6 * (1 + abs(mean))
    assert high >= mean - 1e-6 * (1 + abs(mean))


def test_bootstrap_is_seeded_and_covers_the_true_mean_at_about_the_nominal_rate():
    assert bootstrap_ci([1.0, 2.0, 5.0], seed=3) == bootstrap_ci([1.0, 2.0, 5.0], seed=3)
    rng = np.random.default_rng(0)
    covered = 0
    trials = 200
    for k in range(trials):
        low, high = bootstrap_ci(rng.normal(1.0, 2.0, 40), n_resamples=2000, seed=k)
        covered += low <= 1.0 <= high
    assert 0.88 <= covered / trials <= 0.99


def test_degenerate_samples_give_a_point_interval():
    assert bootstrap_ci([4.0]) == (4.0, 4.0)
    assert bootstrap_ci([2.0, 2.0, 2.0]) == (2.0, 2.0)
    s = summarize([3.0])
    assert (s.n, s.mean, s.std, s.ci_low, s.ci_high) == (1, 3.0, 0.0, 3.0, 3.0)


def test_summary_uses_sample_std():
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s.mean == 2.5 and s.median == 2.5
    assert s.std == pytest.approx(np.std([1, 2, 3, 4], ddof=1))


def test_paired_comparison_detects_a_consistent_difference():
    rng = np.random.default_rng(2)
    b = rng.normal(10, 5, 30)
    a = b - 1.0 + rng.normal(0, 0.1, 30)  # a is cheaper in every pair
    c = paired_comparison(a, b, seed=0)
    assert c.mean_diff == pytest.approx(-1.0, abs=0.1)
    assert c.ci_high < 0 and c.significant
    assert c.wilcoxon_p < 1e-5
    assert c.frac_a_lower == 1.0


def test_paired_comparison_of_identical_samples_is_null():
    x = [1.0, 2.0, 3.0]
    c = paired_comparison(x, x)
    assert c.mean_diff == 0 and c.wilcoxon_p == 1.0 and not c.significant


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        summarize([1.0, float("nan")])
    with pytest.raises(ValueError):
        paired_comparison([1.0, 2.0], [1.0])


def test_holm_adjust_matches_hand_computation():

    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.005])
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> running max 0.06
    assert np.allclose(adjusted, [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(holm_adjust([0.5, 0.9]), [1.0, 1.0])
    assert holm_adjust([]).size == 0
    with pytest.raises(ValueError):
        holm_adjust([1.5])


def test_cluster_bootstrap_ci_brackets_the_mean_and_collapses_for_one_cluster():

    rng = np.random.default_rng(0)
    values = rng.normal(1.0, 1.0, 120)
    clusters = np.repeat(np.arange(12), 10)
    low, high = cluster_bootstrap_ci(values, clusters, n_resamples=2000)
    assert low < values.mean() < high
    assert cluster_bootstrap_ci([1.0, 3.0], [0, 0]) == (2.0, 2.0)
    with pytest.raises(ValueError):
        cluster_bootstrap_ci([1.0, 2.0], [0])
