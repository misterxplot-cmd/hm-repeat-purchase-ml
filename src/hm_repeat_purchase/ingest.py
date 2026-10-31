"""Source manifest, download and idempotent bulk ingest (COPY FROM STDIN) into PostgreSQL."""
from __future__ import annotations

import csv
import hashlib
import logging
import re
import shutil
import time
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

from . import db
from .config import Config
from .tracking import relpath, utc_now, write_json

log = logging.getLogger(__name__)

INGEST_SCHEMA_VERSION = "hmrp_ingest_v1"
ROLES = ("transactions", "customers", "articles")

EXPECTED_HEADERS = {
    "transactions": ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
    "customers": ["customer_id", "FN", "Active", "club_member_status", "fashion_news_frequency", "age", "postal_code"],
    "articles": [
        "article_id", "product_code", "prod_name", "product_type_no", "product_type_name", "product_group_name",
        "graphical_appearance_no", "graphical_appearance_name", "colour_group_code", "colour_group_name",
        "perceived_colour_value_id", "perceived_colour_value_name", "perceived_colour_master_id",
        "perceived_colour_master_name", "department_no", "department_name", "index_code", "index_name",
        "index_group_no", "index_group_name", "section_no", "section_name", "garment_group_no",
        "garment_group_name", "detail_desc",
    ],
}
RAW_COLUMNS = {
    "customers": ["customer_id", "fn", "active", "club_member_status", "fashion_news_frequency", "age", "postal_code"],
    "articles": [h.lower() for h in EXPECTED_HEADERS["articles"]],
}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# --------------------------------------------------------------------------- manifest / download

def sha256_file(path: Path, chunk: int = 1 << 23) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def file_manifest(config: Config, role: str, verify_hash: bool = True) -> dict[str, Any]:
    path = config.data_file(role)
    expected = (config.get("data.source.expected") or {}).get(role) or {}
    entry: dict[str, Any] = {
        "role": role,
        "file_name": path.name,
        "path": relpath(path),
        "exists": path.exists(),
        "source_url": (config.get("data.source.urls") or {}).get(role),
        "expected_sha256": expected.get("sha256"),
        "expected_size_bytes": expected.get("size_bytes"),
    }
    if not path.exists():
        return entry
    entry["size_bytes"] = path.stat().st_size
    with path.open("r", encoding="utf-8", newline="") as fh:
        entry["header"] = next(csv.reader(fh))
    entry["header_ok"] = entry["header"] == EXPECTED_HEADERS[role]
    if verify_hash:
        t0 = time.perf_counter()
        entry["sha256"] = sha256_file(path)
        entry["sha256_seconds"] = round(time.perf_counter() - t0, 2)
        entry["sha256_matches_source"] = (
            None if not expected.get("sha256") else entry["sha256"] == expected["sha256"]
        )
    entry["size_matches_source"] = (
        None if not expected.get("size_bytes") else entry["size_bytes"] == expected["size_bytes"]
    )
    return entry


def download(config: Config, force: bool = False) -> list[dict]:
    """Download missing source CSVs from the configured mirror and verify size and sha256."""
    results = []
    config.data_dir().mkdir(parents=True, exist_ok=True)
    for role in ROLES:
        target = config.data_file(role)
        url = config.get("data.source.urls")[role]
        if target.exists() and not force:
            results.append({"role": role, "action": "present"})
            continue
        part = target.with_suffix(target.suffix + ".part")
        log.info("downloading %s -> %s", url, relpath(target))
        req = urllib.request.Request(url, headers={"User-Agent": "hm-repeat-purchase/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 22)
        part.replace(target)
        results.append({"role": role, "action": "downloaded"})
    for res in results:
        m = file_manifest(config, res["role"])
        res.update({k: m.get(k) for k in ("size_bytes", "sha256", "sha256_matches_source", "size_matches_source")})
        if m.get("sha256_matches_source") is False or m.get("size_matches_source") is False:
            raise RuntimeError(f"{res['role']}: checksum/size mismatch with source metadata")
    return results


# --------------------------------------------------------------------------- loads bookkeeping

def _raw_parent(role: str) -> str:
    return f"raw.{role}"


def raw_partition(role: str, load_id: int) -> str:
    return db.partition_name(_raw_parent(role), f"load_{load_id}")


def completed_load(conn: psycopg.Connection, role: str, sha256: str) -> dict | None:
    row = db.fetch_one(
        conn,
        "SELECT * FROM meta.load WHERE file_role = %(r)s AND sha256 = %(s)s AND schema_version = %(v)s"
        " AND status = 'completed'",
        {"r": role, "s": sha256, "v": INGEST_SCHEMA_VERSION},
    )
    return row or None


def active_load_ids(conn: psycopg.Connection) -> dict[str, int]:
    """Latest completed load per role (the loads a dataset will be built from)."""
    rows = db.fetch_all(
        conn,
        "SELECT DISTINCT ON (file_role) file_role, load_id FROM meta.load WHERE status = 'completed'"
        " AND schema_version = %(v)s ORDER BY file_role, finished_at DESC",
        {"v": INGEST_SCHEMA_VERSION},
    )
    return {r["file_role"]: int(r["load_id"]) for r in rows}


def _drop_load(conn: psycopg.Connection, role: str, load_id: int, new_status: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(db.ident(raw_partition(role, load_id))))
        if role == "transactions":
            cur.execute("DELETE FROM raw.transactions_rejected WHERE load_id = %s", (load_id,))
        cur.execute("UPDATE meta.load SET status = %s, finished_at = coalesce(finished_at, now()) WHERE load_id = %s",
                    (new_status, load_id))
    conn.commit()


def _cleanup_incomplete(conn: psycopg.Connection, role: str) -> None:
    rows = db.fetch_all(conn, "SELECT load_id FROM meta.load WHERE file_role = %(r)s AND status IN ('running', 'failed')",
                        {"r": role})
    for r in rows:
        _drop_load(conn, role, int(r["load_id"]), "failed")
    if role == "transactions":
        with conn.cursor() as cur:
            cur.execute("TRUNCATE staging.transactions_text, staging.date_map")
        conn.commit()


# --------------------------------------------------------------------------- loaders

def _check_header(path: Path, role: str) -> None:
    with path.open("r", encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    if header != EXPECTED_HEADERS[role]:
        raise ValueError(f"{path.name}: header {header} != expected {EXPECTED_HEADERS[role]}")


def _load_transactions(conn: psycopg.Connection, path: Path, load_id: int) -> dict[str, int]:
    """Stream the file into staging with (load_id, src_row) prefixes; type-check in SQL."""
    part = raw_partition("transactions", load_id)
    structural: list[tuple[int, str]] = []
    rows_read = 0
    copy_stmt = ("COPY staging.transactions_text (load_id, src_row, t_dat, customer_id, article_id, price, "
                 "sales_channel_id) FROM STDIN WITH (FORMAT csv)")
    with conn.cursor() as cur:
        with cur.copy(copy_stmt) as cp, path.open("rb") as fh:
            fh.readline()  # header, validated before
            while True:
                lines = fh.readlines(1 << 24)
                if not lines:
                    break
                out = []
                for line in lines:
                    rows_read += 1
                    # Fast structural check: exactly 5 unquoted fields.
                    if line.count(b",") != 4 or b'"' in line:
                        structural.append((rows_read, line.decode("utf-8", "replace").rstrip("\r\n")))
                        continue
                    if not line.endswith(b"\n"):
                        line += b"\n"
                    out.append(b"%d,%d,%s" % (load_id, rows_read, line))
                cp.write(b"".join(out))
    conn.commit()
    log.info("transactions: %s rows streamed to staging (%s structural rejects)", rows_read, len(structural))

    # Validate the distinct date strings once (a few hundred values) instead of per row.
    dates = [r["t_dat"] for r in db.fetch_all(conn, db.sql_block("ingest.sql", "staging_distinct_dates"),
                                               {"load_id": load_id})]
    mapping = []
    for txt in dates:
        parsed = None
        if txt is not None and _DATE_RE.match(txt):
            try:
                parsed = date.fromisoformat(txt)
            except ValueError:
                parsed = None
        if txt is not None:
            mapping.append((load_id, txt, parsed))
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO staging.date_map (load_id, txt, d) VALUES (%s, %s, %s)", mapping)
        cur.execute(db.sql_block("ingest.sql", "create_raw_partition", part=db.ident(part),
                                 parent=db.ident("raw.transactions"), load_id=load_id))
        cur.execute(db.sql_block("ingest.sql", "insert_typed_transactions", part=db.ident(part)), {"load_id": load_id})
        cur.execute(db.sql_block("ingest.sql", "insert_rejected_transactions"), {"load_id": load_id})
        if structural:
            cur.executemany(
                "INSERT INTO raw.transactions_rejected (load_id, src_row, reason, raw_line) VALUES (%s, %s, 'structure', %s)",
                [(load_id, r, line[:2000]) for r, line in structural],
            )
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(sql.SQL("ALTER TABLE {} ADD PRIMARY KEY (load_id, src_row)").format(db.ident(part)))
        cur.execute(sql.SQL("ANALYZE {}").format(db.ident(part)))
    conn.commit()

    loaded = db.fetch_one(conn, db.sql_block("ingest.sql", "raw_partition_totals", part=db.ident(part)))
    rejected = db.fetch_one(
        conn,
        "SELECT count(*) FILTER (WHERE reason = 'structure') AS s, count(*) FILTER (WHERE reason <> 'structure') AS t"
        " FROM raw.transactions_rejected WHERE load_id = %(l)s",
        {"l": load_id},
    )
    staged = db.fetch_one(conn, db.sql_block("ingest.sql", "staging_totals"), {"load_id": load_id})
    with conn.cursor() as cur:
        cur.execute("TRUNCATE staging.transactions_text, staging.date_map")
    conn.commit()
    counts = {
        "rows_read": rows_read,
        "rows_staged": int(staged["n_rows"]),
        "rows_loaded": int(loaded["n_rows"]),
        "rows_rejected_structure": int(rejected["s"]),
        "rows_rejected_type": int(rejected["t"]),
        "sum_price_loaded": float(loaded["sum_price"]),
    }
    if counts["rows_loaded"] + counts["rows_rejected_type"] + counts["rows_rejected_structure"] != rows_read:
        raise RuntimeError(f"transactions row accounting mismatch: {counts}")
    return counts


def _load_reference(conn: psycopg.Connection, path: Path, role: str, load_id: int) -> dict[str, int]:
    """customers.csv / articles.csv: parsed with the csv module (articles has quoted multi-comma text)."""
    part = raw_partition(role, load_id)
    cols = RAW_COLUMNS[role]
    n_fields = len(EXPECTED_HEADERS[role])
    rows_read = loaded = structural = 0
    with conn.cursor() as cur:
        cur.execute(db.sql_block("ingest.sql", "create_raw_partition", part=db.ident(part),
                                 parent=db.ident(_raw_parent(role)), load_id=load_id))
        copy_stmt = sql.SQL("COPY {} ({}) FROM STDIN").format(
            db.ident(part), sql.SQL(", ").join(sql.Identifier(c) for c in ["load_id", "src_row", *cols]))
        with cur.copy(copy_stmt) as cp, path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            next(reader)
            for rec in reader:
                rows_read += 1
                if len(rec) != n_fields:
                    structural += 1
                    continue
                # Empty CSV fields are stored as NULL; values are otherwise kept as source text.
                cp.write_row([load_id, rows_read, *[v if v != "" else None for v in rec]])
                loaded += 1
        cur.execute(sql.SQL("ALTER TABLE {} ADD PRIMARY KEY (load_id, src_row)").format(db.ident(part)))
        cur.execute(sql.SQL("ANALYZE {}").format(db.ident(part)))
    conn.commit()
    return {"rows_read": rows_read, "rows_loaded": loaded, "rows_rejected_structure": structural,
            "rows_rejected_type": 0}


# --------------------------------------------------------------------------- orchestration

def ingest(config: Config, conn: psycopg.Connection, mode: str = "reuse", roles: tuple[str, ...] = ROLES) -> dict:
    """mode='reuse': skip a file whose sha256 already has a completed load (no duplication).
    mode='replace': drop that load's partition and load again. Other loads are never touched."""
    if mode not in ("reuse", "replace"):
        raise ValueError(mode)
    db.ensure_schema(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_lock(hashtext('hmrp_ingest'))")
    results: dict[str, Any] = {"schema_version": INGEST_SCHEMA_VERSION, "mode": mode, "started_at": utc_now(),
                               "files": {}}
    try:
        for role in roles:
            path = config.data_file(role)
            if not path.exists():
                raise FileNotFoundError(f"{role}: {relpath(path)} not found (set HNM_DATA_DIR or run `hmrp download`)")
            _check_header(path, role)
            manifest = file_manifest(config, role, verify_hash=True)
            if manifest.get("sha256_matches_source") is False:
                raise RuntimeError(f"{role}: sha256 differs from source metadata in config")
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO meta.source_file (sha256, file_role, file_name, size_bytes) VALUES (%s, %s, %s, %s)"
                    " ON CONFLICT (sha256) DO NOTHING",
                    (manifest["sha256"], role, manifest["file_name"], manifest["size_bytes"]),
                )
            conn.commit()
            existing = completed_load(conn, role, manifest["sha256"])
            if existing and mode == "reuse":
                log.info("%s: reusing completed load %s", role, existing["load_id"])
                results["files"][role] = {**manifest, "load_id": existing["load_id"], "action": "reused",
                                          **{k: existing[k] for k in ("rows_read", "rows_loaded",
                                                                     "rows_rejected_structure", "rows_rejected_type")}}
                continue
            if existing and mode == "replace":
                _drop_load(conn, role, int(existing["load_id"]), "superseded")
            _cleanup_incomplete(conn, role)
            row = db.fetch_one(
                conn,
                "INSERT INTO meta.load (file_role, sha256, schema_version, status) VALUES (%(r)s, %(s)s, %(v)s,"
                " 'running') RETURNING load_id",
                {"r": role, "s": manifest["sha256"], "v": INGEST_SCHEMA_VERSION},
            )
            conn.commit()
            load_id = int(row["load_id"])
            t0 = time.perf_counter()
            try:
                if role == "transactions":
                    counts = _load_transactions(conn, path, load_id)
                else:
                    counts = _load_reference(conn, path, role, load_id)
            except Exception as exc:
                conn.rollback()
                with conn.cursor() as cur:
                    cur.execute("UPDATE meta.load SET status = 'failed', finished_at = now(), error = %s"
                                " WHERE load_id = %s", (str(exc)[:4000], load_id))
                conn.commit()
                raise
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE meta.load SET status = 'completed', finished_at = now(), rows_read = %s, rows_loaded = %s,"
                    " rows_rejected_structure = %s, rows_rejected_type = %s WHERE load_id = %s",
                    (counts["rows_read"], counts["rows_loaded"], counts["rows_rejected_structure"],
                     counts["rows_rejected_type"], load_id),
                )
            conn.commit()
            results["files"][role] = {**manifest, "load_id": load_id, "action": "loaded",
                                      "load_seconds": round(time.perf_counter() - t0, 1), **counts}
            log.info("%s: load %s completed %s", role, load_id, counts)
    finally:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(hashtext('hmrp_ingest'))")
        conn.commit()
    results["finished_at"] = utc_now()
    out = config.reports_dir / "data" / f"ingest_manifest_{config.mode}.json"
    write_json(out, results)
    return results
