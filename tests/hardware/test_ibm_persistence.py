"""IBM job persistence with a fake SamplerV2: every job's ID and raw counts reach disk.

qiskit-ibm-runtime is never imported: a stand-in module is placed in sys.modules.
"""

import json
import sys
import types
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit

from qexec.hardware import ibm
from qexec.optimization.toy import toy_execution_qubo

BACKEND = SimpleNamespace(name="fake_fez", num_qubits=156)


def read_log(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


class FakeSampler:
    """SamplerV2 stand-in; `on_run(job_number)` runs before each job is created."""

    jobs_run = 0
    on_run = None
    fail_on_job: int | None = None

    def __init__(self, mode):
        assert mode is BACKEND

    def run(self, circuits, shots):
        cls = FakeSampler
        cls.jobs_run += 1
        number = cls.jobs_run
        if cls.on_run is not None:
            cls.on_run(number)
        n = circuits[0].num_qubits
        counts = {format(i, f"0{n}b"): shots // 2**n for i in range(2**n)}

        def result():
            if number == cls.fail_on_job:
                raise RuntimeError("job cancelled: quota exhausted")
            data = SimpleNamespace(meas=SimpleNamespace(get_counts=lambda: counts))
            return [SimpleNamespace(data=data)]

        return SimpleNamespace(job_id=lambda: f"job-{number}", result=result)


@pytest.fixture
def fake_runtime(monkeypatch):
    module = types.ModuleType("qiskit_ibm_runtime")
    module.SamplerV2 = FakeSampler
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", module)
    monkeypatch.setattr(FakeSampler, "jobs_run", 0)
    monkeypatch.setattr(FakeSampler, "on_run", None)
    monkeypatch.setattr(FakeSampler, "fail_on_job", None)
    # Transpilation for a real device is out of scope here; pass circuits through.
    monkeypatch.setattr(ibm, "transpile_for", lambda backend: SimpleNamespace(run=lambda c: c))
    return FakeSampler


def measured_circuit(n: int = 2) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


def test_each_job_is_on_disk_before_the_next_is_submitted(fake_runtime, tmp_path):
    log = tmp_path / "jobs.jsonl"
    lines_seen_at_submit = []
    fake_runtime.on_run = lambda number: lines_seen_at_submit.append(len(read_log(log)))

    for k in range(3):
        counts, job_id = ibm.run_sampler_job(
            measured_circuit(), BACKEND, 400, log, metadata={"k": k}
        )
        assert job_id == f"job-{k + 1}"
        record = read_log(log)[-1]
        assert record["job_id"] == job_id
        assert record["counts"] == counts == {"00": 100, "01": 100, "10": 100, "11": 100}
        assert record["shots"] == 400
        assert record["backend"] == "fake_fez"
        assert record["metadata"] == {"k": k}
        assert record["transpiled_depth"] == measured_circuit().depth()
    assert lines_seen_at_submit == [0, 1, 2]


def test_completed_jobs_survive_a_later_failure(fake_runtime, tmp_path):
    log = tmp_path / "jobs.jsonl"
    fake_runtime.fail_on_job = 2
    ibm.run_sampler_job(measured_circuit(), BACKEND, 100, log)
    with pytest.raises(RuntimeError, match="quota"):
        ibm.run_sampler_job(measured_circuit(), BACKEND, 100, log)
    assert [r["job_id"] for r in read_log(log)] == ["job-1"]


def test_hardware_qaoa_logs_every_evaluation_and_the_final_sample(
    fake_runtime, tmp_path, bitstrings
):
    log = tmp_path / "jobs.jsonl"
    Q = toy_execution_qubo(4)
    result, job_ids = ibm.run_qaoa_on_hardware(
        Q, BACKEND, p=1, shots=160, maxiter=4, seed=0, log_path=log, metadata={"run": 7}
    )
    records = read_log(log)
    assert [r["job_id"] for r in records] == job_ids
    assert len(job_ids) == result.num_iterations + 1
    stages = [r["metadata"]["stage"] for r in records]
    assert stages == ["optimize"] * result.num_iterations + ["final"]
    assert [r["metadata"]["evaluation"] for r in records] == list(range(len(records)))
    assert all(r["metadata"]["run"] == 7 and r["metadata"]["num_qubits"] == 4 for r in records)
    assert [r["shots"] for r in records] == [160] * result.num_iterations + [800]
    assert records[-1]["counts"] == result.counts
    assert result.energy == pytest.approx(min(float(x @ Q @ x) for x in bitstrings(4)))


def test_extract_counts_finds_a_custom_register_name():
    data = SimpleNamespace(c_out=SimpleNamespace(get_counts=lambda: {"1": 3}))
    assert ibm.extract_counts(SimpleNamespace(data=data)) == {"1": 3}
    with pytest.raises(AttributeError, match="No measurement data"):
        ibm.extract_counts(SimpleNamespace(data=SimpleNamespace(other=1)))
