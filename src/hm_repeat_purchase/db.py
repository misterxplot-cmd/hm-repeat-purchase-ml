"""PostgreSQL connection, .env loading and named SQL blocks from sql/*.sql."""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

from .config import project_root

_BLOCK_RE = re.compile(r"^-- name:\s*(\S+)\s*$", re.MULTILINE)


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader: KEY=VALUE lines; existing environment variables win."""
    path = path or project_root() / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def conninfo(dbname: str | None = None) -> dict[str, Any]:
    load_dotenv()
    password = os.environ.get("HMRP_PG_PASSWORD")
    if not password:
        raise RuntimeError("HMRP_PG_PASSWORD is not set (copy .env.example to .env and set a local password)")
    return {
        "host": os.environ.get("HMRP_PG_HOST", "127.0.0.1"),
        "port": int(os.environ.get("HMRP_PG_PORT", "55432")),
        "user": os.environ.get("HMRP_PG_USER", "hmrp"),
        "password": password,
        "dbname": dbname or os.environ.get("HMRP_PG_DB", "hmrp"),
        "connect_timeout": 10,
        "application_name": "hm_repeat_purchase",
    }


def connect(dbname: str | None = None, autocommit: bool = False) -> psycopg.Connection:
    return psycopg.connect(**conninfo(dbname), autocommit=autocommit)


@lru_cache(maxsize=None)
def _blocks(file_name: str) -> dict[str, str]:
    text = (project_root() / "sql" / file_name).read_text(encoding="utf-8")
    parts = _BLOCK_RE.split(text)
    blocks: dict[str, str] = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        lines = [ln for ln in body.strip().splitlines()]
        # Drop leading comment lines that document the block.
        while lines and lines[0].lstrip().startswith("--"):
            lines.pop(0)
        text_block = "\n".join(lines).strip()
        if text_block.upper().startswith("COPY"):
            text_block = text_block.rstrip(";")
        blocks[name] = text_block
    return blocks


def sql_block(file_name: str, name: str, **parts: Any) -> sql.Composed | sql.SQL:
    """Return a named SQL block; keyword args fill {placeholders} with Identifiers/Literals/SQL."""
    # Always format, so that escaped braces ({{n}} in regexes) become literal braces.
    return sql.SQL(_blocks(file_name)[name]).format(**{k: _as_sql(v) for k, v in parts.items()})


def _as_sql(value: Any) -> sql.Composable:
    if isinstance(value, sql.Composable):
        return value
    return sql.Literal(value)


def ident(qualified: str) -> sql.Identifier:
    return sql.Identifier(*qualified.split("."))


def partition_name(parent: str, suffix: str) -> str:
    schema, table = parent.split(".")
    safe = re.sub(r"[^a-z0-9_]", "_", suffix.lower())
    return f"{schema}.{table}__{safe}"


def fetch_one(conn: psycopg.Connection, query, params: dict | None = None) -> dict:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(query, params or {})
        row = cur.fetchone()
    return dict(row) if row else {}


def fetch_all(conn: psycopg.Connection, query, params: dict | None = None) -> list[dict]:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(query, params or {})
        return [dict(r) for r in cur.fetchall()]


def table_exists(conn: psycopg.Connection, qualified: str) -> bool:
    row = fetch_one(conn, "SELECT to_regclass(%(t)s) IS NOT NULL AS ok", {"t": qualified})
    return bool(row["ok"])


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(sql_block("schema.sql", "create_schema"))
    conn.commit()


def copy_to_bytes(conn: psycopg.Connection, query) -> bytes:
    chunks: list[bytes] = []
    with conn.cursor() as cur:
        with cur.copy(query) as cp:
            for data in cp:
                chunks.append(bytes(data))
    return b"".join(chunks)
