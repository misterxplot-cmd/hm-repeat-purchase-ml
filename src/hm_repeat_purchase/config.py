"""YAML configuration with `extends`, path resolution and environment overrides."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    env = os.environ.get("HMRP_PROJECT_ROOT")
    return Path(env).resolve() if env else _PACKAGE_ROOT


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    parent = data.pop("extends", None)
    if parent:
        data = _deep_merge(_load_yaml((path.parent / parent).resolve()), data)
    return data


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


@dataclass(frozen=True)
class Config:
    raw: dict
    path: Path
    data_dir_override: Path | None = None

    @classmethod
    def load(cls, path: str | Path, data_dir: str | Path | None = None) -> "Config":
        p = Path(path)
        if not p.is_absolute():
            candidate = (Path.cwd() / p)
            p = candidate if candidate.exists() else project_root() / p
        p = p.resolve()
        override = Path(data_dir) if data_dir else None
        return cls(raw=_load_yaml(p), path=p, data_dir_override=override)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def mode(self) -> str:
        return str(self.raw["mode"])

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def resolve(self, value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (project_root() / p)

    def data_dir(self) -> Path:
        if self.data_dir_override is not None:
            return self.data_dir_override.resolve()
        env = os.environ.get("HNM_DATA_DIR")
        if env:
            return Path(env).resolve()
        return self.resolve(self.raw["data"]["dir"])

    def data_file(self, role: str) -> Path:
        return self.data_dir() / self.raw["data"]["files"][role]

    @property
    def reports_dir(self) -> Path:
        return self.resolve(self.raw["outputs"]["reports_dir"])

    @property
    def artifacts_dir(self) -> Path:
        return self.resolve(self.raw["outputs"]["artifacts_dir"])

    @property
    def derived_dir(self) -> Path:
        return self.resolve(self.raw["outputs"]["derived_dir"])

    def as_json(self) -> dict:
        return _jsonable(self.raw)

    def fingerprint(self) -> str:
        blob = json.dumps(self.as_json(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]
