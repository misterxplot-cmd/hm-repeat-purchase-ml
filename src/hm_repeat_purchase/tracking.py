"""Run identifiers, manifests, environment capture and stage timing."""
from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import date, datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import psutil

from .config import Config, project_root

TRACKED_PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn", "catboost", "psycopg", "psycopg-binary",
    "pyarrow", "matplotlib", "seaborn", "PyYAML", "psutil",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(config: Config) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{config.mode}-{stamp}-{config.fingerprint()[:6]}"


def git_info() -> dict[str, Any]:
    root = project_root()

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def environment_info(config: Config | None = None) -> dict[str, Any]:
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    vm = psutil.virtual_memory()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / 2**30, 1),
        "packages": versions,
        "compute": {
            "catboost_task_type": config.get("resources.task_type", "CPU") if config else None,
            "catboost_threads": config.get("resources.catboost_threads") if config else None,
        },
    }


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return relpath(obj)
    return obj


def relpath(p: Path) -> str:
    """Project-relative POSIX path, so manifests never contain personal absolute paths."""
    p = Path(p)
    try:
        return p.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return f"<external>/{p.name}"


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def peak_rss_mb() -> float | None:
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows
    if peak is None:
        try:
            import resource  # type: ignore

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except Exception:
            return None
    return round(peak / 2**20, 1)


class StageTimer:
    """Collects wall time and process memory per stage into a JSON file (appending)."""

    def __init__(self, path: Path):
        self.path = path
        self.records: list[dict] = read_json(path) if path.exists() else []

    @contextmanager
    def stage(self, name: str, **extra: Any) -> Iterator[dict]:
        rec: dict[str, Any] = {"stage": name, "started_at": utc_now(), **extra}
        t0 = time.perf_counter()
        rss0 = psutil.Process().memory_info().rss
        try:
            yield rec
            rec["status"] = "ok"
        except BaseException as exc:
            rec["status"] = f"failed: {type(exc).__name__}"
            raise
        finally:
            rec["wall_seconds"] = round(time.perf_counter() - t0, 3)
            rec["rss_start_mb"] = round(rss0 / 2**20, 1)
            rec["rss_end_mb"] = round(psutil.Process().memory_info().rss / 2**20, 1)
            rec["process_peak_rss_mb"] = peak_rss_mb()
            self.records.append(rec)
            write_json(self.path, self.records)


def run_paths(config: Config, run_id: str) -> dict[str, Path]:
    """Published (reports/runs/<run_id>) and local-only (artifacts/runs/<run_id>) locations."""
    pub = config.reports_dir / "runs" / run_id
    loc = config.artifacts_dir / "runs" / run_id
    return {
        "published": pub,
        "local": loc,
        "models": loc / "models",
        "predictions": loc / "predictions",
        "scoring": loc / "scoring",
        "figures": pub / "figures",
    }


def latest_run_file(config: Config) -> Path:
    return config.artifacts_dir / "runs" / f"LATEST_{config.mode}.txt"


def resolve_run_id(config: Config, run_id: str | None) -> str:
    if run_id:
        return run_id
    f = latest_run_file(config)
    if not f.exists():
        raise SystemExit(f"No run id given and {relpath(f)} does not exist; run `train` first or pass --run-id")
    return f.read_text(encoding="utf-8").strip()
