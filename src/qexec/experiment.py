from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import pandas as pd

MANIFEST = "manifest.json"
_TRACKED_PACKAGES = ("qexec", "numpy", "scipy", "pandas", "matplotlib", "qiskit", "qiskit-aer")


def to_jsonable(obj: Any) -> Any:  # noqa: PLR0911 - one return per supported type
    """Convert recursively to JSON types; non-finite floats become None (strict JSON)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return to_jsonable(obj.value)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def git_state(cwd: Path | None = None) -> dict[str, Any]:
    """Commit hash and dirty flag of tracked files; both None outside a git repository."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "packages": {name: _package_version(name) for name in _TRACKED_PACKAGES},
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil  # noqa: PLC0415 - optional at import time of this module

        info["memory_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:  # pragma: no cover - psutil is a declared dependency
        info["memory_gb"] = None
    if sys.platform == "darwin":
        try:
            info["cpu_model"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            info["cpu_model"] = None
    return info


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExperimentRecorder:
    """Context manager owning `<root>/<name>/` for one run; stale files are removed on entry."""

    def __init__(
        self,
        name: str,
        config: Any,
        seeds: Sequence[int],
        *,
        root: Path = Path("results"),
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if not name or "/" in name or name.startswith("."):
            raise ValueError(f"Invalid experiment name {name!r}")
        self.name = name
        self.config = config
        self.seeds = [int(s) for s in seeds]
        self.directory = Path(root) / name
        self.extra = dict(extra or {})
        self._files: list[str] = []
        self._start = 0.0
        self._started_at = ""

    def __enter__(self) -> ExperimentRecorder:
        self.directory.mkdir(parents=True, exist_ok=True)
        for stale in self.directory.iterdir():
            if stale.is_file():
                stale.unlink()
        self._started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._start = time.perf_counter()
        return self

    def _register(self, filename: str) -> Path:
        if filename == MANIFEST or "/" in filename:
            raise ValueError(f"Invalid results file name {filename!r}")
        if filename in self._files:
            raise ValueError(f"{filename} already written in this run")
        self._files.append(filename)
        return self.directory / filename

    def write_table(self, name: str, df: pd.DataFrame, float_format: str = "%.10g") -> Path:
        path = self._register(f"{name}.csv")
        df.to_csv(path, index=False, float_format=float_format)
        return path

    def write_json(self, name: str, obj: Any) -> Path:
        path = self._register(f"{name}.json")
        path.write_text(json.dumps(to_jsonable(obj), indent=2, sort_keys=False) + "\n")
        return path

    def note(self, key: str, value: Any) -> None:
        self.extra[key] = value

    def manifest(self, wall_time_s: float, status: str) -> dict[str, Any]:
        return {
            "experiment": self.name,
            "status": status,
            "config_type": type(self.config).__name__,
            "config": to_jsonable(self.config),
            "seeds": self.seeds,
            "git": git_state(),
            "environment": environment(),
            "started_utc": self._started_at,
            "wall_time_s": round(wall_time_s, 3),
            "files": {f: _sha256(self.directory / f) for f in self._files},
            **to_jsonable(self.extra),
        }

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        status = "ok" if exc_type is None else f"failed: {exc_type.__name__}"
        manifest = self.manifest(time.perf_counter() - self._start, status)
        (self.directory / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")


def load_manifest(directory: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((Path(directory) / MANIFEST).read_text())
    return data


def verify_files(directory: Path) -> list[str]:
    """Files whose SHA-256 no longer matches the manifest, or that are missing."""
    manifest = load_manifest(directory)
    bad = []
    for name, digest in manifest["files"].items():
        path = Path(directory) / name
        if not path.exists() or _sha256(path) != digest:
            bad.append(name)
    return bad
