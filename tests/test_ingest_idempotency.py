"""Bulk ingest: header check, type quarantine, row accounting, idempotent re-runs."""
from __future__ import annotations

import pytest
from psycopg import sql

import synthetic
from conftest import make_config
from hm_repeat_purchase import db
from hm_repeat_purchase.ingest import ingest, raw_partition

pytestmark = pytest.mark.integration


def _count(conn, load_id):
    return db.fetch_one(conn, sql.SQL("SELECT count(*) AS n, sum(price) AS s FROM {}")
                        .format(db.ident(raw_partition("transactions", load_id))))


def test_reingest_same_file_does_not_duplicate(tmp_path, pg_conn, synthetic_data):
    cfg = make_config(tmp_path, synthetic_data["transactions"].parent)
    first = ingest(cfg, pg_conn)["files"]["transactions"]
    second = ingest(cfg, pg_conn)["files"]["transactions"]
    assert first["action"] == "loaded" and second["action"] == "reused"
    assert first["load_id"] == second["load_id"]
    n_lines = sum(1 for _ in synthetic_data["transactions"].open(encoding="utf-8")) - 1
    assert first["rows_read"] == first["rows_loaded"] == n_lines
    assert _count(pg_conn, first["load_id"])["n"] == n_lines
    total = db.fetch_one(pg_conn, "SELECT count(*) AS n FROM raw.transactions")["n"]
    assert total == n_lines


def test_replace_mode_reloads_exactly_once(tmp_path, pg_conn, synthetic_data):
    cfg = make_config(tmp_path, synthetic_data["transactions"].parent)
    first = ingest(cfg, pg_conn)["files"]["transactions"]
    again = ingest(cfg, pg_conn, mode="replace")["files"]["transactions"]
    assert again["action"] == "loaded" and again["load_id"] != first["load_id"]
    status = db.fetch_one(pg_conn, "SELECT status FROM meta.load WHERE load_id = %(l)s", {"l": first["load_id"]})
    assert status["status"] == "superseded"
    total = db.fetch_one(pg_conn, "SELECT count(*) AS n FROM raw.transactions")["n"]
    assert total == again["rows_loaded"]


def test_identical_rows_are_kept(tmp_path, pg_conn):
    rows = [["2020-01-01", synthetic.cid(1), "0000000007", "0.0101", 2]] * 3
    synthetic.write_csv(tmp_path / "transactions_train.csv", synthetic.TX_HEADER, rows)
    synthetic.write_csv(tmp_path / "customers.csv", synthetic.CUST_HEADER, [[synthetic.cid(1), "", "", "", "", "", ""]])
    synthetic.write_csv(tmp_path / "articles.csv", synthetic.ART_HEADER, synthetic.articles_rows(7))
    cfg = make_config(tmp_path / "o", tmp_path)
    res = ingest(cfg, pg_conn)["files"]["transactions"]
    c = _count(pg_conn, res["load_id"])
    assert c["n"] == 3 and c["s"] == pytest.approx(0.0303)


def test_type_violations_are_quarantined_not_imputed(tmp_path, pg_conn):
    good = ["2020-01-01", synthetic.cid(1), "0000000007", "0.02", 1]
    rows = [good,
            ["2019-02-30", synthetic.cid(1), "0000000007", "0.02", 1],     # impossible date
            ["2020-01-02", synthetic.cid(1), "0000000007", "abc", 1],      # non-numeric price
            ["2020-01-03", "not-a-customer", "7", "0.02", 1]]              # bad ids
    synthetic.write_csv(tmp_path / "transactions_train.csv", synthetic.TX_HEADER, rows)
    with (tmp_path / "transactions_train.csv").open("a", encoding="utf-8") as fh:
        fh.write("2020-01-04,only,three\n")                                 # structural error
    synthetic.write_csv(tmp_path / "customers.csv", synthetic.CUST_HEADER, [[synthetic.cid(1), "", "", "", "", "", ""]])
    synthetic.write_csv(tmp_path / "articles.csv", synthetic.ART_HEADER, synthetic.articles_rows(7))
    cfg = make_config(tmp_path / "o", tmp_path)
    res = ingest(cfg, pg_conn)["files"]["transactions"]
    assert res["rows_read"] == 5 and res["rows_loaded"] == 1
    assert res["rows_rejected_type"] == 3 and res["rows_rejected_structure"] == 1
    reasons = {r["reason"] for r in db.fetch_all(pg_conn, "SELECT reason FROM raw.transactions_rejected")}
    assert {"invalid_t_dat", "invalid_price", "structure"} <= reasons
    assert any("invalid_customer_id" in r and "invalid_article_id" in r for r in reasons)


def test_header_mismatch_is_rejected(tmp_path, pg_conn):
    synthetic.write_csv(tmp_path / "transactions_train.csv", ["t_dat", "customer_id", "article_id", "price"],
                        [["2020-01-01", synthetic.cid(1), "0000000007", "0.02"]])
    cfg = make_config(tmp_path / "o", tmp_path)
    with pytest.raises(ValueError, match="header"):
        ingest(cfg, pg_conn, roles=("transactions",))
