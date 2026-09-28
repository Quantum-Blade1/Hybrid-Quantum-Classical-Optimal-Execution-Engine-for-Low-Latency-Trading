"""Post-processing error mitigation on measured counts: ZNE, readout-matrix inversion, voting.

These operate on counts only; noise-scaled circuits (e.g. gate folding) for ZNE and
calibration circuits for readout mitigation must be produced by the caller.
"""

import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Counts = dict[str, int]
_DEFAULT_READOUT_ERROR = 0.02
_MIN_MITIGATED_PROBABILITY = 0.001


def _bits(bitstring: str) -> NDArray[np.int_]:
    """Qiskit bitstring (qubit 0 rightmost) to a binary vector with qubit 0 first."""
    return np.array([int(b) for b in bitstring[::-1]])


def best_from_counts(counts: Counts, Q: NDArray[np.float64]) -> tuple[NDArray[np.int_], float]:
    """Lowest-energy measured n-bit string; (zeros, inf) when none has length n."""
    n = Q.shape[0]
    best_x = np.zeros(n, dtype=int)
    best_e = float("inf")
    for bitstring in counts:
        x = _bits(bitstring)
        if len(x) == n:
            e = float(x @ Q @ x)
            if e < best_e:
                best_e, best_x = e, x
    return best_x, best_e


def mean_energy(counts: Counts, Q: NDArray[np.float64]) -> float:
    """Count-weighted mean of x^T Q x over n-bit strings (normalised by all shots)."""
    n = Q.shape[0]
    total = sum(counts.values())
    value = 0.0
    for bitstring, count in counts.items():
        x = _bits(bitstring)
        if len(x) == n:
            value += float(x @ Q @ x) * count / total
    return value


@dataclass
class MitigationResult:
    """Raw and mitigated counts, best solutions and energies; improvement = raw - mitigated."""

    raw_counts: Counts
    mitigated_counts: Counts
    raw_solution: NDArray[np.int_]
    mitigated_solution: NDArray[np.int_]
    raw_energy: float
    mitigated_energy: float
    improvement: float
    technique: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def improvement_percent(self) -> float:
        if abs(self.raw_energy) < 1e-10:
            return 0.0
        return (self.raw_energy - self.mitigated_energy) / abs(self.raw_energy) * 100


class ErrorMitigator(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def mitigate(self, counts: Counts, Q: NDArray[np.float64], **kwargs: Any) -> MitigationResult:
        """Mitigate `counts` of a circuit whose bitstrings are scored with `Q`."""


class ZeroNoiseExtrapolation(ErrorMitigator):
    """Extrapolates the mean energy measured at noise factors c_k to c = 0.

    `extrapolation` is "linear", "polynomial" (degree <= 2) or "exponential" (a e^{bc},
    requiring energies of one sign). Mitigated counts are the raw counts reweighted by
    1 / (1 + |E(x) - E_0|), a heuristic that favours bitstrings near the extrapolated energy.
    """

    def __init__(
        self, noise_factors: Sequence[float] = (1.0, 1.5, 2.0), extrapolation: str = "linear"
    ) -> None:
        if extrapolation not in ("linear", "polynomial", "exponential"):
            raise ValueError(f"Unknown extrapolation {extrapolation!r}")
        self.noise_factors = list(noise_factors)
        self.extrapolation = extrapolation

    @property
    def name(self) -> str:
        return f"ZNE-{self.extrapolation}"

    def mitigate(
        self,
        counts: Counts,
        Q: NDArray[np.float64],
        noise_scaled_counts: list[Counts] | None = None,
        **kwargs: Any,
    ) -> MitigationResult:
        """`counts` is the factor-1 run; `noise_scaled_counts` the runs at the other factors."""
        raw_solution, raw_energy = best_from_counts(counts, Q)
        if noise_scaled_counts is None or len(noise_scaled_counts) < 2:
            logger.warning("ZNE requires counts at three or more noise levels; returning raw")
            return MitigationResult(
                raw_counts=counts,
                mitigated_counts=counts,
                raw_solution=raw_solution,
                mitigated_solution=raw_solution,
                raw_energy=raw_energy,
                mitigated_energy=raw_energy,
                improvement=0.0,
                technique=self.name,
                metadata={"warning": "No noise-scaled data provided"},
            )

        expected_values = [mean_energy(c, Q) for c in [counts, *noise_scaled_counts]]
        factors = self.noise_factors[: len(expected_values)]
        mitigated_energy = self._extrapolate(factors, expected_values[: len(factors)])
        mitigated_counts = self._reweight_counts(counts, Q, mitigated_energy)
        mit_solution, _ = best_from_counts(mitigated_counts, Q)

        return MitigationResult(
            raw_counts=counts,
            mitigated_counts=mitigated_counts,
            raw_solution=raw_solution,
            mitigated_solution=mit_solution,
            raw_energy=raw_energy,
            mitigated_energy=mitigated_energy,
            improvement=raw_energy - mitigated_energy,
            technique=self.name,
            metadata={"noise_factors": factors, "expected_values": expected_values},
        )

    def _extrapolate(self, factors: list[float], values: list[float]) -> float:
        if self.extrapolation == "linear":
            return float(np.polyfit(factors, values, 1)[1])
        if self.extrapolation == "polynomial":
            degree = min(2, len(factors) - 1)
            return float(np.polyval(np.polyfit(factors, values, degree), 0))
        signs = np.sign(values)
        if not (np.all(signs > 0) or np.all(signs < 0)):
            raise ValueError("Exponential extrapolation needs expectation values of one sign")
        intercept = np.polyfit(factors, np.log(np.abs(values)), 1)[1]
        return float(signs[0] * np.exp(intercept))

    @staticmethod
    def _reweight_counts(counts: Counts, Q: NDArray[np.float64], target_energy: float) -> Counts:
        n = Q.shape[0]
        mitigated = {}
        for bitstring, count in counts.items():
            x = _bits(bitstring)
            if len(x) == n:
                weight = 1.0 / (abs(float(x @ Q @ x) - target_energy) + 1.0)
                mitigated[bitstring] = int(count * weight * 2)
        return mitigated


class MeasurementErrorMitigator(ErrorMitigator):
    """Readout mitigation p_true = pinv(A) p_measured with confusion matrix A[prepared, measured].

    Negative quasi-probabilities are clipped and the result renormalised. Without calibration,
    A assumes independent 2% bit-flip readout errors.
    """

    def __init__(self, confusion_matrix: NDArray[np.float64] | None = None) -> None:
        self._confusion_matrix = confusion_matrix

    @property
    def name(self) -> str:
        return "MeasurementError"

    def calibrate(self, calibration_counts: list[Counts], n_qubits: int) -> None:
        """`calibration_counts[s]` are the counts measured after preparing basis state s."""
        n_states = 2**n_qubits
        matrix = np.zeros((n_states, n_states))
        for prepared_state, counts in enumerate(calibration_counts):
            total = sum(counts.values())
            for measured, count in counts.items():
                measured_idx = int(measured[::-1], 2) if len(measured) == n_qubits else 0
                if measured_idx < n_states:
                    matrix[prepared_state, measured_idx] = count / total
        self._confusion_matrix = matrix
        logger.info("Readout mitigation calibrated for %d qubits", n_qubits)

    @staticmethod
    def independent_error_matrix(n_qubits: int, error_rate: float) -> NDArray[np.float64]:
        """A[i, j] = (1 - e)^(n - d) e^d with d the Hamming distance, rows normalised."""
        states = np.arange(2**n_qubits)
        hamming = np.array([[bin(i ^ j).count("1") for j in states] for i in states])
        matrix = (1 - error_rate) ** (n_qubits - hamming) * error_rate**hamming
        return np.asarray(matrix / matrix.sum(axis=1, keepdims=True), dtype=np.float64)

    def mitigate(self, counts: Counts, Q: NDArray[np.float64], **kwargs: Any) -> MitigationResult:
        n = Q.shape[0]
        raw_solution, raw_energy = best_from_counts(counts, Q)
        if self._confusion_matrix is None:
            logger.warning(
                "Readout mitigation not calibrated; assuming %.0f%% independent readout error",
                100 * _DEFAULT_READOUT_ERROR,
            )
            self._confusion_matrix = self.independent_error_matrix(n, _DEFAULT_READOUT_ERROR)

        n_states = 2**n
        total = sum(counts.values())
        prob_vector = np.zeros(n_states)
        for bitstring, count in counts.items():
            if len(bitstring) == n:
                prob_vector[int(bitstring[::-1], 2)] = count / total

        try:
            mitigated_probs = np.linalg.pinv(self._confusion_matrix) @ prob_vector
            mitigated_probs = np.maximum(mitigated_probs, 0)
            mitigated_probs /= mitigated_probs.sum() + 1e-10
        except np.linalg.LinAlgError:
            logger.exception("Confusion-matrix inversion failed; using raw probabilities")
            mitigated_probs = prob_vector

        mitigated_counts = {
            format(idx, f"0{n}b")[::-1]: int(prob * total)
            for idx, prob in enumerate(mitigated_probs)
            if prob > _MIN_MITIGATED_PROBABILITY
        }
        mit_solution, mit_energy = best_from_counts(mitigated_counts, Q)
        return MitigationResult(
            raw_counts=counts,
            mitigated_counts=mitigated_counts,
            raw_solution=raw_solution,
            mitigated_solution=mit_solution,
            raw_energy=raw_energy,
            mitigated_energy=mit_energy,
            improvement=raw_energy - mit_energy,
            technique=self.name,
        )


class MajorityVoting(ErrorMitigator):
    """Pools counts from repeated runs: plain sum, or weighted by 1/(1 + |E|) ("weighted")."""

    def __init__(self, voting_method: str = "weighted") -> None:
        if voting_method not in ("simple", "weighted"):
            raise ValueError(f"Unknown voting method {voting_method!r}")
        self.voting_method = voting_method

    @property
    def name(self) -> str:
        return f"MajorityVote-{self.voting_method}"

    def mitigate(
        self,
        counts: Counts,
        Q: NDArray[np.float64],
        additional_counts: list[Counts] | None = None,
        **kwargs: Any,
    ) -> MitigationResult:
        all_counts = [counts, *(additional_counts or [])]
        raw_solution, raw_energy = best_from_counts(counts, Q)
        if self.voting_method == "simple":
            mitigated_counts = self._simple_vote(all_counts)
        else:
            mitigated_counts = self._weighted_vote(all_counts, Q)
        mit_solution, mit_energy = best_from_counts(mitigated_counts, Q)
        return MitigationResult(
            raw_counts=counts,
            mitigated_counts=mitigated_counts,
            raw_solution=raw_solution,
            mitigated_solution=mit_solution,
            raw_energy=raw_energy,
            mitigated_energy=mit_energy,
            improvement=raw_energy - mit_energy,
            technique=self.name,
            metadata={"num_runs": len(all_counts)},
        )

    @staticmethod
    def _simple_vote(all_counts: list[Counts]) -> Counts:
        aggregated: Counter[str] = Counter()
        for counts in all_counts:
            aggregated.update(counts)
        return dict(aggregated)

    @staticmethod
    def _weighted_vote(all_counts: list[Counts], Q: NDArray[np.float64]) -> Counts:
        n = Q.shape[0]
        aggregated: Counter[str] = Counter()
        for counts in all_counts:
            for bitstring, count in counts.items():
                x = _bits(bitstring)
                if len(x) == n:
                    aggregated[bitstring] += int(count * (1.0 / (abs(float(x @ Q @ x)) + 1.0)))
        return dict(aggregated)


class ErrorMitigationPipeline:
    """Applies techniques in sequence, each on the previous stage's mitigated counts."""

    def __init__(self, techniques: list[ErrorMitigator] | None = None) -> None:
        self.techniques = techniques or [
            MeasurementErrorMitigator(),
            MajorityVoting(voting_method="weighted"),
        ]

    def run(
        self,
        counts: Counts,
        Q: NDArray[np.float64],
        additional_data: dict[str, Any] | None = None,
    ) -> list[MitigationResult]:
        """One result per technique that succeeded; failing techniques are logged and skipped."""
        results = []
        current_counts = counts
        for technique in self.techniques:
            try:
                result = technique.mitigate(current_counts, Q, **(additional_data or {}))
            except (ValueError, np.linalg.LinAlgError):
                logger.exception("%s failed; skipping", technique.name)
                continue
            results.append(result)
            current_counts = result.mitigated_counts
            logger.info("%s: improvement=%.4f", technique.name, result.improvement)
        return results

    @staticmethod
    def summary(results: list[MitigationResult]) -> dict[str, Any]:
        if not results:
            return {}
        return {
            "initial_energy": results[0].raw_energy,
            "final_energy": results[-1].mitigated_energy,
            "total_improvement": results[0].raw_energy - results[-1].mitigated_energy,
            "techniques_applied": [r.technique for r in results],
            "stage_improvements": [r.improvement for r in results],
        }
