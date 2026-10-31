"""Shared fixtures. Integration tests use a separate database (HMRP_PG_TEST_DB, default hmrp_test)
that is dropped and recreated once per test session; they are skipped with an explicit reason when
PostgreSQL is not reachable."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from hm_repeat_purchase import db  # noqa: E402
from hm_repeat_purchase.config import Config, project_root  # noqa: E402

import synthetic  # noqa: E402

_PG_STATUS: dict = {}


def _pg_available() -> tuple[bool, str]:
    if "ok" in _PG_STATUS:
        return _PG_STATUS["ok"], _PG_STATUS["reason"]
    db.load_dotenv()
    test_db = os.environ.get("HMRP_PG_TEST_DB", "hmrp_test")
    try:
        with db.connect(dbname="postgres", autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT pid FROM pg_stat_activity WHERE datname = %s", (test_db,))
                cur.execute(f'DROP DATABASE IF EXISTS "{test_db}" WITH (FORCE)')
                cur.execute(f'CREATE DATABASE "{test_db}"')
        _PG_STATUS.update(ok=True, reason="")
    except Exception as exc:  # noqa: BLE001
        _PG_STATUS.update(ok=False, reason=f"PostgreSQL not reachable: {str(exc).splitlines()[0]}")
    return _PG_STATUS["ok"], _PG_STATUS["reason"]


@pytest.fixture(scope="session")
def pg_dbname() -> str:
    ok, reason = _pg_available()
    if not ok:
        pytest.skip(reason)
    return os.environ.get("HMRP_PG_TEST_DB", "hmrp_test")


@pytest.fixture()
def pg_conn(pg_dbname):
    """Connection to the test database with a clean set of project schemas."""
    with db.connect(dbname=pg_dbname) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS meta, staging, raw, derived CASCADE")
        conn.commit()
        db.ensure_schema(conn)
        yield conn


def make_config(tmp_path: Path, data_dir: Path, **overrides) -> Config:
    """A fixture-mode config writing every output under tmp_path."""
    raw = yaml.safe_load((project_root() / "tests" / "fixtures" / "fixture.yaml").read_text(encoding="utf-8"))
    raw["data"]["dir"] = str(data_dir)
    raw["outputs"] = {"reports_dir": str(tmp_path / "reports"), "artifacts_dir": str(tmp_path / "artifacts"),
                      "derived_dir": str(tmp_path / "derived")}
    for dotted, value in overrides.items():
        node = raw
        *head, last = dotted.split("__")
        for h in head:
            node = node.setdefault(h, {})
        node[last] = value
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return Config.load(path)


@pytest.fixture()
def synthetic_data(tmp_path) -> dict[str, Path]:
    return synthetic.generate(tmp_path / "data")


@pytest.fixture()
def use_test_db(pg_dbname, monkeypatch):
    monkeypatch.setenv("HMRP_PG_DB", pg_dbname)
    return pg_dbname
