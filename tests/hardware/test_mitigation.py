"""Counts post-processing: readout-error inversion and zero-noise extrapolation."""

import numpy as np
import pytest

from qexec.hardware.mitigation import MeasurementErrorMitigator, ZeroNoiseExtrapolation


def test_readout_mitigation_inverts_an_asymmetric_confusion_matrix():
    # Real readout error is asymmetric (|1> decays to 0 more often than 0 flips to 1).
    # Calibration rows are prepared states: P(read 1 | 0) = 2%, P(read 0 | 1) = 20%.
    mitigator = MeasurementErrorMitigator()
    mitigator.calibrate([{"0": 980, "1": 20}, {"0": 200, "1": 800}], n_qubits=1)
    # True distribution (0.3, 0.7) is read as (0.3*0.98 + 0.7*0.2, 0.3*0.02 + 0.7*0.8).
    measured = {"0": 434, "1": 566}
    result = mitigator.mitigate(measured, np.array([[-1.0]]))
    assert result.mitigated_counts["0"] == pytest.approx(300, abs=1)
    assert result.mitigated_counts["1"] == pytest.approx(700, abs=1)


def test_readout_mitigation_recovers_distribution_under_independent_bit_flips():
    n, error = 3, 0.05
    A = MeasurementErrorMitigator.independent_error_matrix(n, error)
    true = np.zeros(2**n)
    true[[0b101, 0b011]] = [0.6, 0.4]
    measured_probs = A.T @ true  # A[prepared, measured]
    measured = {format(i, "03b")[::-1]: round(p * 100_000) for i, p in enumerate(measured_probs)}
    result = MeasurementErrorMitigator(A).mitigate(measured, np.zeros((n, n)))
    recovered = np.zeros(2**n)
    for bitstring, count in result.mitigated_counts.items():
        recovered[int(bitstring[::-1], 2)] = count / 100_000
    np.testing.assert_allclose(recovered, true, atol=2e-3)


@pytest.mark.parametrize("extrapolation", ["linear", "polynomial"])
def test_zne_recovers_the_zero_noise_intercept_of_linear_decay(extrapolation):
    # E(c) = -2 P1(c) with P1 = 0.9 - 0.1 c, so E(0) = -1.8.
    runs = [{"1": 800, "0": 200}, {"1": 700, "0": 300}, {"1": 600, "0": 400}]
    zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 2.0, 3.0], extrapolation=extrapolation)
    result = zne.mitigate(runs[0], np.array([[-2.0]]), noise_scaled_counts=runs[1:])
    assert result.mitigated_energy == pytest.approx(-1.8)
    assert result.metadata["expected_values"] == pytest.approx([-1.6, -1.4, -1.2])


def test_zne_exponential_keeps_the_sign_of_negative_energies():
    # Regression: the exponential fit took log of negative values and lost the sign.
    zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 2.0, 3.0], extrapolation="exponential")
    values = [-10 * np.exp(-0.5 * c) for c in (1.0, 2.0, 3.0)]
    assert zne._extrapolate([1.0, 2.0, 3.0], values) == pytest.approx(-10.0)
    with pytest.raises(ValueError, match="one sign"):
        zne._extrapolate([1.0, 2.0], [-1.0, 1.0])


def test_zne_without_noise_scaled_runs_returns_raw_result():
    counts = {"1": 10, "0": 5}
    result = ZeroNoiseExtrapolation().mitigate(counts, np.array([[-2.0]]))
    assert result.mitigated_counts == counts
    assert result.improvement == 0.0
