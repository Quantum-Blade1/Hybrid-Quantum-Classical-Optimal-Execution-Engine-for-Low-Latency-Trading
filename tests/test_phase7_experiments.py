"""Phase 7 experiment plumbing: graceful skips, hardware analysis on the SYNTHETIC fixture,
optional figures, and a quick real-data run on synthetic bars."""

import dataclasses
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from experiments import hardware_analysis, run_all
from experiments.common import Experiment, SkipExperiment

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "ibm_jobs_sample.jsonl"


@dataclasses.dataclass(frozen=True)
class _Config:
    seed: int = 0


def _experiment(reason):
    def run(config, rec):  # pragma: no cover - must not be reached when skipped
        raise AssertionError("run() called despite a failing precheck")

    return Experiment(
        "skippy", _Config(), _Config(), run, lambda c: [c.seed], precheck=lambda c: reason
    )


def test_precheck_skips_before_writing(tmp_path, capsys):
    exp = _experiment("inputs missing")
    with pytest.raises(SkipExperiment, match="inputs missing"):
        exp.execute(quick=False, results_dir=tmp_path)
    assert not (tmp_path / "skippy").exists()
    exp.main(["--results-dir", str(tmp_path)])
    assert "skipped (inputs missing)" in capsys.readouterr().out


def test_run_all_skips_experiments_without_inputs(tmp_path, monkeypatch, capsys):
    # In an empty working directory there are no bars and no recovered IBM counts.
    monkeypatch.chdir(tmp_path)
    argv = ["run_all", "--results-dir", "out", "--only", "real_data_dev", "real_data_test"]
    monkeypatch.setattr(sys, "argv", [*argv, "hardware"])
    run_all.main()  # must not exit non-zero
    out = capsys.readouterr().out
    assert out.count("skipped") == 3
    assert "make data" in out and "ibm_fez_recovered.jsonl" in out
    assert not (tmp_path / "out").exists()


def test_hardware_analysis_on_synthetic_fixture(tmp_path):
    exp = dataclasses.replace(
        hardware_analysis.EXPERIMENT,
        quick=hardware_analysis.Config(
            jobs_path=str(FIXTURE), simulator_runs=str(tmp_path / "none.csv"), synthetic=True
        ),
    )
    exp.execute(quick=True, results_dir=tmp_path)
    out = tmp_path / "hardware"
    jobs = pd.read_csv(out / "jobs.csv")
    assert sorted(jobs["job_id"]) == ["FAKE-fixture-0001", "FAKE-fixture-0002", "FAKE-fixture-0003"]
    assert len(pd.read_csv(out / "excluded.csv")) == 3
    summary = pd.read_csv(out / "summary.csv")
    assert list(summary["n"]) == [4, 6]  # final jobs only
    assert summary["success_ratio_vs_uniform"].iloc[0] == pytest.approx(0.4 * 16)
    manifest = json.loads((out / "manifest.json").read_text())
    assert "SYNTHETIC" in manifest["data"]
    assert "FAKE-fixture-0001" in manifest["job_ids"]


def test_simulator_reference_reads_toy_qaoa_runs(tmp_path):
    runs = pd.DataFrame(
        {
            "family": ["toy", "toy", "random"],
            "n": [4, 4, 4],
            "solver": ["QAOA_Ideal", "QAOA_Ideal", "QAOA_Ideal"],
            "p": [1, 1, 1],
            "success_probability": [0.1, 0.3, 0.9],
            "approx_ratio_mean": [0.8, 0.9, 0.1],
            "approx_ratio_best": [1.0, 1.0, 1.0],
            "optimal_found": [True, False, True],
        }
    )
    runs.to_csv(tmp_path / "runs.csv", index=False)
    ref = hardware_analysis.simulator_reference(tmp_path / "runs.csv")
    assert ref["sim_success_probability"].iloc[0] == pytest.approx(0.2)
    assert ref["sim_optimal_found"].iloc[0] == pytest.approx(0.5)
    assert hardware_analysis.simulator_reference(tmp_path / "missing.csv").empty


def test_optional_figures_are_skipped_when_inputs_are_absent(tmp_path, monkeypatch, capsys):
    from figures import make_figures

    argv = ["make_figures", "--results-dir", str(tmp_path / "r"), "--output-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", [*argv, "--only", "fig_real", "fig_hw_ibm"])
    make_figures.main()
    out = capsys.readouterr().out
    assert "0/8 figures (8 optional skipped)" in out
    assert not list(tmp_path.glob("*.pdf"))


@pytest.mark.slow
def test_real_data_quick_run_on_synthetic_bars(tmp_path):
    from experiments import real_data

    real_data.DEV.execute(quick=True, results_dir=tmp_path)
    out = tmp_path / "real_data_dev"
    comparisons = pd.read_csv(out / "comparisons.csv")
    primary = comparisons[comparisons["variant"] == "primary"]
    assert len(primary) == 12  # 2 symbols x {QUBO, Hybrid} x {TWAP, VWAP, AC}
    assert primary["p_holm"].between(0, 1).all()
    assert (primary["p_holm"] >= primary["wilcoxon_p"] - 1e-12).all()
    gaps = pd.read_csv(out / "gaps.csv")
    # Every model cost is of a schedule summing to the order (VWAP included).
    assert (gaps["expected_cost_VWAP"] < 5 * gaps["expected_cost_TWAP"]).all()
    assert (gaps["objective_AC"] <= gaps["objective_TWAP"] + 1e-9).all()
    orders = pd.read_csv(out / "orders.csv")
    assert set(real_data.STRATEGIES) <= set(orders["strategy"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["data"].startswith("synthetic")
