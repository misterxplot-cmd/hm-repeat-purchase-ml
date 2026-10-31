"""Many-to-one JOIN guards and the synthetic JOIN-trap demonstration."""
from __future__ import annotations

import math
from typing import Any

import psycopg
from psycopg import sql

from . import db


class JoinCardinalityError(RuntimeError):
    pass


def key_uniqueness(conn: psycopg.Connection, table: str, key: str) -> dict[str, int]:
    row = db.fetch_one(conn, db.sql_block("validation_checks.sql", "key_uniqueness",
                                          table=db.ident(table), key=sql.Identifier(key)))
    return {k: int(v) for k, v in row.items()}


def checked_many_to_one(conn: psycopg.Connection, fact: str, dim: str, key: str, measure: str,
                        rel_tol: float = 1e-9) -> dict[str, Any]:
    """Refuse the JOIN unless the dimension key is unique; then verify rows and sum are preserved."""
    uniq = key_uniqueness(conn, dim, key)
    result: dict[str, Any] = {"fact": fact, "dim": dim, "key": key, "measure": measure, "dim_key": uniq}
    if uniq["n_rows"] != uniq["n_distinct"] + uniq["n_null"] or uniq["n_null"] > 0:
        result["status"] = "rejected_dim_key_not_unique"
        raise JoinCardinalityError(f"{dim}.{key} is not a unique non-null key: {uniq}")
    row = db.fetch_one(conn, db.sql_block("validation_checks.sql", "many_to_one_join", fact=db.ident(fact),
                                          dim=db.ident(dim), key=sql.Identifier(key), measure=sql.Identifier(measure)))
    row = {k: (float(v) if k.endswith("_sum") else int(v)) for k, v in row.items()}
    result.update(row)
    rows_ok = row["fact_rows"] == row["joined_rows"]
    sum_ok = math.isclose(row["fact_sum"], row["joined_sum"], rel_tol=rel_tol, abs_tol=1e-9)
    result["rows_preserved"] = rows_ok
    result["sum_preserved"] = sum_ok
    if not (rows_ok and sum_ok):
        result["status"] = "failed_rows_or_sum_changed"
        raise JoinCardinalityError(f"JOIN changed rows or sum: {result}")
    result["status"] = "passed"
    return result


def demo_join_trap(conn: psycopg.Connection) -> dict[str, Any]:
    """DEMONSTRATION on synthetic rows (not a finding in H&M data).

    One customer-day with three item rows (two identical), and an article reference table where one
    article_id is deliberately duplicated. The unchecked JOIN multiplies rows and spend; the guard
    rejects it; the corrected version joins against a reference with a unique key.
    """
    out: dict[str, Any] = {"kind": "demonstration_on_synthetic_data"}
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE demo_tx (t_dat date, customer_id text, article_id text, price double precision) ON COMMIT DROP;
            INSERT INTO demo_tx VALUES
                ('2020-01-10', 'c1', '0000000001', 0.010),
                ('2020-01-10', 'c1', '0000000001', 0.010),
                ('2020-01-10', 'c1', '0000000002', 0.025),
                ('2020-01-11', 'c2', '0000000002', 0.025);
            CREATE TEMP TABLE demo_articles_dup (article_id text, product_group_name text) ON COMMIT DROP;
            INSERT INTO demo_articles_dup VALUES
                ('0000000001', 'Garment Upper body'),
                ('0000000002', 'Accessories'),
                ('0000000002', 'Accessories (duplicate row)');
            CREATE TEMP TABLE demo_articles_ok (article_id text, product_group_name text) ON COMMIT DROP;
            INSERT INTO demo_articles_ok VALUES
                ('0000000001', 'Garment Upper body'),
                ('0000000002', 'Accessories');
        """)
        cur.execute("SELECT count(*), sum(price) FROM demo_tx")
        n, s = cur.fetchone()
        cur.execute("SELECT count(*), sum(t.price) FROM demo_tx t JOIN demo_articles_dup a USING (article_id)")
        n_bad, s_bad = cur.fetchone()
        cur.execute("""SELECT count(*) FROM (SELECT customer_id, t_dat FROM demo_tx GROUP BY 1, 2) d""")
        (n_days,) = cur.fetchone()
    out["fact_rows"] = int(n)
    out["fact_sum_price"] = float(s)
    out["customer_days"] = int(n_days)
    out["unchecked_join_rows"] = int(n_bad)
    out["unchecked_join_sum_price"] = float(s_bad)
    try:
        checked_many_to_one(conn, "pg_temp.demo_tx", "pg_temp.demo_articles_dup", "article_id", "price")
        out["guard_on_duplicate_dim"] = "accepted (unexpected)"
    except JoinCardinalityError as exc:
        out["guard_on_duplicate_dim"] = f"rejected: {exc}"
    ok = checked_many_to_one(conn, "pg_temp.demo_tx", "pg_temp.demo_articles_ok", "article_id", "price")
    out["corrected_join"] = {k: ok[k] for k in ("fact_rows", "joined_rows", "fact_sum", "joined_sum",
                                                "n_unmatched", "status")}
    conn.rollback()
    return out
