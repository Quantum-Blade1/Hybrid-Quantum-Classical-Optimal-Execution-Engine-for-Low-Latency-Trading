from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qexec.optimization.solvers.qaoa import Counts, QAOAResult, run_qaoa

if TYPE_CHECKING:
    from qiskit_ibm_runtime import QiskitRuntimeService


def save_account(token: str, channel: str = "ibm_quantum", overwrite: bool = True) -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService

    QiskitRuntimeService.save_account(channel=channel, token=token, overwrite=overwrite)


def connect_service(
    token: str | None = None,
    channel: str | None = None,
    instance: str | None = None,
) -> QiskitRuntimeService:
    from qiskit_ibm_runtime import QiskitRuntimeService

    if channel is None:
        channel = "ibm_cloud" if instance else "ibm_quantum"
    kwargs: dict[str, str] = {"channel": channel}
    if token is not None:
        kwargs["token"] = token
    if instance:
        kwargs["instance"] = instance
    return QiskitRuntimeService(**kwargs)


def get_backend(
    service: QiskitRuntimeService, name: str | None = None, min_qubits: int = 0
) -> BackendV2:
    if name:
        return service.backend(name)
    from qiskit_ibm_runtime import least_busy

    backends = service.backends(
        filters=lambda b: b.configuration().n_qubits >= min_qubits and b.status().operational
    )
    return least_busy(backends)


def backend_properties(backend: BackendV2) -> dict[str, Any]:
    return {
        "name": backend.name,
        "num_qubits": backend.num_qubits,
        "version": str(getattr(backend, "version", "unknown")),
    }


def transpile_for(backend: BackendV2, optimization_level: int = 3) -> PassManager:
    return generate_preset_pass_manager(optimization_level=optimization_level, backend=backend)


def extract_counts(pub_result: Any) -> Counts:
    """Counts from a SamplerV2 PubResult, whatever the classical register is called."""
    data = pub_result.data
    for attr in ("meas", "c", "cr"):
        if hasattr(data, attr):
            counts: Counts = getattr(data, attr).get_counts()
            return counts
    public = [a for a in dir(data) if not a.startswith("_")]
    for attr in public:
        obj = getattr(data, attr)
        if hasattr(obj, "get_counts"):
            counts = obj.get_counts()
            return counts
    raise AttributeError(f"No measurement data found in DataBin. Attributes: {public}")


def _json_default(obj: object) -> int | float | list[Any]:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return list(obj.tolist())
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON record and fsync it, so it survives a later crash or quota cut-off."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_sampler_job(
    isa_circuit: QuantumCircuit,
    backend: BackendV2,
    shots: int,
    log_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> tuple[Counts, str]:
    from qiskit_ibm_runtime import SamplerV2

    submitted = datetime.now(timezone.utc).isoformat()
    job = SamplerV2(mode=backend).run([isa_circuit], shots=shots)
    job_id: str = job.job_id()
    counts = extract_counts(job.result()[0])
    append_jsonl(
        log_path,
        {
            "submitted_utc": submitted,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "backend": backend.name,
            "shots": shots,
            "transpiled_depth": isa_circuit.depth(),
            "metadata": metadata or {},
            "counts": counts,
        },
    )
    return counts, job_id


def run_qaoa_on_hardware(
    Q: NDArray[np.float64],
    backend: BackendV2,
    *,
    p: int,
    shots: int,
    maxiter: int,
    seed: int,
    log_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> tuple[QAOAResult, list[str]]:
    """QAOA on an IBM backend (final sampling at 5x shots); returns (result, job IDs in order)."""
    pm = transpile_for(backend)
    job_ids: list[str] = []
    base_meta = dict(metadata or {}, num_qubits=int(Q.shape[0]), p=p)

    def sample(circuit: QuantumCircuit, n_shots: int) -> Counts:
        meta = dict(base_meta, **(circuit.metadata or {}), evaluation=len(job_ids))
        counts, job_id = run_sampler_job(pm.run(circuit), backend, n_shots, log_path, meta)
        job_ids.append(job_id)
        return counts

    result = run_qaoa(
        Q,
        p=p,
        sample=sample,
        shots=shots,
        maxiter=maxiter,
        final_shots=shots * 5,
        rng=np.random.default_rng(seed),
    )
    return result, job_ids
