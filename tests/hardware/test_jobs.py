"""Recovered IBM job analysis on a fake fixture: bit order, roles, exclusions, metrics."""

import json
from pathlib import Path

import numpy as np
import pytest

from qexec.hardware.jobs import analyse_counts, job_role, read_jobs, screen_and_analyse
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
