import json
from pathlib import Path

import numpy as np
import pytest

from qexec.hardware.jobs import (
    RunLabel,
    analyse_counts,
    assign_runs,
    job_role,
    latest_by_job,
    read_jobs,
    screen_and_analyse,
)
from qexec.optimization.toy import toy_execution_qubo

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "ibm_jobs_sample.jsonl"


def test_bit_order_is_qiskit_little_endian():
    # toy n=4 optimum: x = (x0, x1, x2, x3) = (1, 0, 1, 0); Qiskit prints qubit 0 rightmost.
    Q = toy_execution_qubo(4)
    x = np.array([1, 0, 1, 0])
    reversed_x = np.array([0, 1, 0, 1])
    assert float(x @ Q @ x) < float(reversed_x @ Q @ reversed_x)
    optimal = analyse_counts("a", "", 4, 5000, {"0101": 5000})
    assert optimal.success_probability == 1.0
    assert optimal.approx_ratio_mean == pytest.approx(1.0)
    wrong = analyse_counts("b", "", 4, 5000, {"1010": 5000})
    assert wrong.success_probability == 0.0
    assert wrong.approx_ratio_mean < 1.0


def test_roles_follow_shot_counts():
    assert [job_role(s) for s in (1000, 2000, 4000)] == ["optimization"] * 3
    assert [job_role(s) for s in (5000, 10000, 20000)] == ["final"] * 3
    assert job_role(3000) == "unknown"


def test_fixture_screening_and_metrics():
    screening = screen_and_analyse(read_jobs(FIXTURE))
    ids = [a.job_id for a in screening.analysed]
    assert ids == ["FAKE-fixture-0001", "FAKE-fixture-0002", "FAKE-fixture-0003"]
    reasons = {e["job_id"]: e["reason"] for e in screening.excluded}
    assert "status" in reasons["FAKE-fixture-0004"]
    assert "not a benchmark size" in reasons["FAKE-fixture-0005"]
    assert "neither" in reasons["FAKE-fixture-0006"]

    first = screening.analysed[0]
    assert first.role == "final" and first.n == 4
    assert first.success_probability == pytest.approx(0.4)
    assert first.optimal_found
    assert first.uniform_success_probability == pytest.approx(1 / 16)
    assert first.uniform_prob_optimum_in_shots == pytest.approx(1.0)
    Q = toy_execution_qubo(4)
    energies = {
        k: float(np.array([int(b) for b in k[::-1]]) @ Q @ np.array([int(b) for b in k[::-1]]))
        for k in ("0101", "1010", "0110", "1001", "0000")
    }
    counts = {"0101": 4000, "1010": 1000, "0110": 2000, "1001": 1500, "0000": 1500}
    mean = sum(energies[k] * c for k, c in counts.items()) / 10000
    assert first.mean_energy == pytest.approx(mean)
    assert screening.analysed[1].role == "optimization"


def test_malformed_lines_and_key_length(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"job_id": 1}\nnot json\n')
    with pytest.raises(ValueError, match="not JSON"):
        read_jobs(bad)
    record = {"job_id": "x", "status": "DONE", "num_bits": 4, "shots": 5000, "counts": {"01": 5}}
    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps(record) + "\n\n")
    screening = screen_and_analyse(read_jobs(good))
    assert screening.excluded[0]["reason"] == "count keys do not match num_bits"


def test_duplicate_job_records_keep_the_latest_status(tmp_path):
    # The recovery file lists a job once per status query: QUEUED, then CANCELLED.
    lines = [
        {"job_id": "a", "status": "QUEUED", "created": "t1"},
        {"job_id": "b", "status": "DONE", "num_bits": 4, "shots": 2000, "counts": {"0101": 2000}},
        {"job_id": "a", "status": "CANCELLED", "created": "t1"},
    ]
    path = tmp_path / "dup.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    records = read_jobs(path)
    assert [r["job_id"] for r in records] == ["a", "b"]
    assert records[0]["status"] == "CANCELLED"
    screening = screen_and_analyse(records)
    assert [e["job_id"] for e in screening.excluded] == ["a"]
    assert screening.excluded[0]["reason"] == "status 'CANCELLED'"
    assert latest_by_job([]) == []


def test_runs_are_assigned_per_n_by_creation_time():
    # Each run ends with its final job; trailing loop jobs form an incomplete run.
    jobs = [
        ("a1", 4, "optimization", "00:01"),
        ("b1", 10, "optimization", "00:02"),
        ("a2", 4, "optimization", "00:03"),
        ("aF", 4, "final", "00:04"),
        ("b2", 10, "optimization", "00:05"),
        ("bF", 10, "final", "00:06"),
        ("a3", 4, "optimization", "00:07"),
        ("a4", 4, "optimization", "00:08"),
        ("aG", 4, "final", "00:09"),
        ("a5", 4, "optimization", "00:10"),
    ]
    labels = assign_runs(reversed(jobs))  # input order does not matter
    assert labels["a1"] == RunLabel(1, 1, True)
    assert labels["aF"] == RunLabel(1, 3, True)
    assert labels["b2"] == RunLabel(1, 2, True)
    assert labels["bF"] == RunLabel(1, 3, True)
    assert labels["a3"] == RunLabel(2, 1, True)
    assert labels["aG"] == RunLabel(2, 3, True)
    assert labels["a5"] == RunLabel(3, 1, False)
