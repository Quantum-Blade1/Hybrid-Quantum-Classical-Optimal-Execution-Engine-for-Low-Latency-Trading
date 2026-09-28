"""
IBM Quantum hardware access via qiskit-ibm-runtime.

    connect_service      QiskitRuntimeService for the ibm_quantum or ibm_cloud channel
    get_backend          named backend, or the least busy one with enough qubits
    backend_properties   name / qubit count / version of a backend
    transpile_for        preset pass manager (ISA circuits) for a backend
    extract_counts       counts from a SamplerV2 PubResult
    run_sampler_job      run one circuit with SamplerV2 and persist job_id + raw counts
    run_qaoa_on_hardware QAOA variational loop (qexec.optimization.solvers.qaoa.run_qaoa)
                         with every job persisted as it returns

Every hardware job is appended to a JSONL file as soon as its result comes
back, so raw counts and job IDs survive a later crash or quota cut-off.

Requires the optional dependency: pip install -e ".[hardware]"
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from qexec.optimization.solvers.qaoa import QAOAResult, run_qaoa


def save_account(token: str, channel: str = "ibm_quantum", overwrite: bool = True) -> None:
    """Store IBM Quantum credentials locally (qiskit-ibm-runtime account file)."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    QiskitRuntimeService.save_account(channel=channel, token=token, overwrite=overwrite)


def connect_service(
    token: Optional[str] = None,
    channel: Optional[str] = None,
    instance: Optional[str] = None,
):
    """
    Create a QiskitRuntimeService.

    Args:
        token: API key/token; None uses the saved account.
        channel: "ibm_quantum" or "ibm_cloud"; defaults to "ibm_cloud" when an
            instance CRN is given, else "ibm_quantum".
        instance: IBM Cloud CRN (ibm_cloud channel).
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    if channel is None:
        channel = "ibm_cloud" if instance else "ibm_quantum"
    kwargs = {"channel": channel}
    if token is not None:
        kwargs["token"] = token
    if instance:
        kwargs["instance"] = instance
    return QiskitRuntimeService(**kwargs)


def get_backend(service, name: Optional[str] = None, min_qubits: int = 0):
    """Named backend, or the least busy operational backend with >= min_qubits."""
    if name:
        return service.backend(name)
    from qiskit_ibm_runtime import least_busy

    backends = service.backends(
        filters=lambda b: b.configuration().n_qubits >= min_qubits and b.status().operational
    )
    return least_busy(backends)


def backend_properties(backend) -> Dict[str, object]:
    return {
        "name": backend.name,
        "num_qubits": backend.num_qubits,
        "version": str(getattr(backend, 'version', 'unknown')),
    }


def transpile_for(backend, optimization_level: int = 3):
    """Preset pass manager producing ISA circuits for ``backend``."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    return generate_preset_pass_manager(optimization_level=optimization_level, backend=backend)


def extract_counts(pub_result) -> Dict[str, int]:
    """Extract measurement counts from a SamplerV2 PubResult, handling different DataBin attribute names."""
    data = pub_result.data
    for attr in ('meas', 'c', 'cr'):
        if hasattr(data, attr):
            return getattr(data, attr).get_counts()
    for attr in dir(data):
        if not attr.startswith('_'):
            obj = getattr(data, attr)
            if hasattr(obj, 'get_counts'):
                return obj.get_counts()
    raise AttributeError(
        f"No measurement data found in DataBin. Attributes: {[a for a in dir(data) if not a.startswith('_')]}"
    )


def append_jsonl(path: str, record: Dict) -> None:
    """Append one JSON record and fsync, so it is on disk before the caller continues."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def run_sampler_job(
    isa_circuit,
    backend,
    shots: int,
    log_path: str,
    metadata: Optional[Dict] = None,
) -> Tuple[Dict[str, int], str]:
    """
    Run one ISA circuit with SamplerV2 and persist the result immediately.

    The JSONL record holds timestamps, job_id, backend, shots, transpiled
    depth, the caller's metadata and the full raw counts.

    Returns:
        (counts, job_id)
    """
    from qiskit_ibm_runtime import SamplerV2

    submitted = datetime.now(timezone.utc).isoformat()
    job = SamplerV2(mode=backend).run([isa_circuit], shots=shots)
    job_id = job.job_id()
    result = job.result()
    counts = extract_counts(result[0])
    append_jsonl(log_path, {
        "submitted_utc": submitted,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "backend": backend.name,
        "shots": shots,
        "transpiled_depth": isa_circuit.depth(),
        "metadata": metadata or {},
        "counts": counts,
    })
    return counts, job_id


def run_qaoa_on_hardware(
    Q: np.ndarray,
    backend,
    p: int,
    shots: int,
    maxiter: int,
    seed: int,
    log_path: str,
    metadata: Optional[Dict] = None,
) -> Tuple[QAOAResult, List[str]]:
    """
    QAOA on an IBM backend: COBYLA over ``shots``-shot expectation values,
    then one final sampling with ``5 * shots`` shots.

    Every SamplerV2 job (optimisation and final) is appended to ``log_path``
    with ``metadata`` plus the stage ("optimize"/"final"), evaluation index,
    and the circuit's gammas and betas.

    Returns:
        (QAOAResult, job IDs in submission order)
    """
    pm = transpile_for(backend)
    job_ids: List[str] = []
    base_meta = dict(metadata or {}, num_qubits=int(Q.shape[0]), p=p)

    def sample(circuit, n_shots: int) -> Dict[str, int]:
        meta = dict(base_meta, **(circuit.metadata or {}), evaluation=len(job_ids))
        counts, job_id = run_sampler_job(pm.run(circuit), backend, n_shots, log_path, meta)
        job_ids.append(job_id)
        return counts

    result = run_qaoa(
        Q, p=p, sample=sample, shots=shots, maxiter=maxiter,
        final_shots=shots * 5, rng=np.random.default_rng(seed),
    )
    return result, job_ids
