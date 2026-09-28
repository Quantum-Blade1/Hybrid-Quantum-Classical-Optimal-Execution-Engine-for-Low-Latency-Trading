"""Result type shared by the classical QUBO solvers."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

BinaryVector = NDArray[np.number]
"""0/1 decision vector; integer or float dtype."""


@dataclass
class QUBOResult:
    """Best solution found, its energy x^T Q x, and solver effort."""

    solution: BinaryVector
    energy: float
    num_evaluations: int
    solve_time: float
    solver_name: str
    iterations: int = 0
    history: list[float] | None = None

    def __repr__(self) -> str:
        head = (
            f"QUBOResult({self.solver_name})\n"
            f"  Energy: {self.energy:.4f}\n"
            f"  Evaluations: {self.num_evaluations:,}\n"
            f"  Time: {self.solve_time:.3f}s\n"
        )
        shown = self.solution[:10]
        suffix = "..." if len(self.solution) > 10 else ""
        return f"{head}  Solution: {shown}{suffix}"


class QUBOSolver(Protocol):
    """Any solver returning a `QUBOResult` for a QUBO matrix."""

    def solve(self, Q: NDArray[np.float64]) -> QUBOResult: ...
