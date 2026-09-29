import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from qexec.experiment import ExperimentRecorder, load_manifest, to_jsonable, verify_files


@dataclass(frozen=True)
class DemoConfig:
    sizes: list[int] = field(default_factory=lambda: [4, 6])
    shots: int = 100
    noise: float = 0.02


def test_manifest_records_config_seeds_environment_and_file_hashes(tmp_path):
    with ExperimentRecorder("demo", DemoConfig(), seeds=[1, 2, 3], root=tmp_path) as rec:
        rec.write_table("runs", pd.DataFrame({"seed": [1, 2, 3], "x": [0.5, 0.25, 0.125]}))
        rec.write_json("summary", {"mean": np.float64(0.29), "n": np.int64(3)})
        rec.note("quick", True)

    manifest = load_manifest(tmp_path / "demo")
    assert manifest["experiment"] == "demo"
    assert manifest["status"] == "ok"
    assert manifest["config"] == {"sizes": [4, 6], "shots": 100, "noise": 0.02}
    assert manifest["config_type"] == "DemoConfig"
    assert manifest["seeds"] == [1, 2, 3]
    assert manifest["quick"] is True
    assert set(manifest["files"]) == {"runs.csv", "summary.json"}
    assert {"commit", "dirty"} <= set(manifest["git"])
    env = manifest["environment"]
    assert env["python"] and env["packages"]["numpy"] == np.__version__
    assert manifest["wall_time_s"] >= 0 and manifest["started_utc"].endswith("+00:00")
    assert json.loads((tmp_path / "demo" / "summary.json").read_text()) == {
        "mean": 0.29,
        "n": 3,
    }
    assert verify_files(tmp_path / "demo") == []


def test_tampered_or_stale_files_are_detected_and_removed(tmp_path):
    with ExperimentRecorder("demo", DemoConfig(), seeds=[0], root=tmp_path) as rec:
        rec.write_table("runs", pd.DataFrame({"x": [1]}))
        rec.write_table("old", pd.DataFrame({"x": [1]}))
    (tmp_path / "demo" / "runs.csv").write_text("x\n2\n")
    assert verify_files(tmp_path / "demo") == ["runs.csv"]

    with ExperimentRecorder("demo", DemoConfig(), seeds=[0], root=tmp_path) as rec:
        rec.write_table("runs", pd.DataFrame({"x": [3]}))
    assert sorted(p.name for p in (tmp_path / "demo").iterdir()) == ["manifest.json", "runs.csv"]


def test_failed_run_is_marked_in_the_manifest(tmp_path):
    with (
        pytest.raises(RuntimeError),
        ExperimentRecorder("demo", DemoConfig(), seeds=[0], root=tmp_path),
    ):
        raise RuntimeError("boom")
    assert load_manifest(tmp_path / "demo")["status"] == "failed: RuntimeError"


@pytest.mark.parametrize("bad", ["", "../x", ".hidden", "a/b"])
def test_invalid_experiment_names_are_rejected(bad, tmp_path):
    with pytest.raises(ValueError):
        ExperimentRecorder(bad, DemoConfig(), seeds=[], root=tmp_path)


def test_files_cannot_be_written_twice_or_shadow_the_manifest(tmp_path):
    with ExperimentRecorder("demo", DemoConfig(), seeds=[0], root=tmp_path) as rec:
        rec.write_json("summary", {})
        with pytest.raises(ValueError):
            rec.write_json("summary", {})
        with pytest.raises(ValueError):
            rec.write_json("manifest", {})


def test_to_jsonable_makes_strict_json():
    value = to_jsonable({"a": np.array([1.0, np.nan]), "b": (np.int32(2), float("inf"))})
    assert value == {"a": [1.0, None], "b": [2, None]}
    json.dumps(value, allow_nan=False)
