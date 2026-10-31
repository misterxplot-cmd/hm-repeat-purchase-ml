"""Many-to-one JOIN guard: a non-unique dimension key is rejected; a correct JOIN keeps rows and sums."""
from __future__ import annotations

import pytest

from hm_repeat_purchase.joins import JoinCardinalityError, checked_many_to_one, demo_join_trap

pytestmark = pytest.mark.integration


def _tables(conn, dim_rows):
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE f (article_id text, price double precision)")
        cur.execute("INSERT INTO f VALUES ('0000000001', 0.01), ('0000000001', 0.01), ('0000000002', 0.03),"
                    " ('0000000003', 0.05)")
        cur.execute("CREATE TEMP TABLE dim (article_id text, grp text)")
        cur.executemany("INSERT INTO dim VALUES (%s, %s)", dim_rows)


def test_duplicate_dimension_key_is_rejected(pg_conn):
    _tables(pg_conn, [("0000000001", "a"), ("0000000002", "b"), ("0000000002", "b-dup"), ("0000000003", "c")])
    with pytest.raises(JoinCardinalityError):
        checked_many_to_one(pg_conn, "pg_temp.f", "pg_temp.dim", "article_id", "price")
    pg_conn.rollback()


def test_null_dimension_key_is_rejected(pg_conn):
    _tables(pg_conn, [("0000000001", "a"), (None, "x")])
    with pytest.raises(JoinCardinalityError):
        checked_many_to_one(pg_conn, "pg_temp.f", "pg_temp.dim", "article_id", "price")
    pg_conn.rollback()


def test_correct_join_preserves_rows_and_sum_and_counts_unmatched(pg_conn):
    _tables(pg_conn, [("0000000001", "a"), ("0000000002", "b")])
    res = checked_many_to_one(pg_conn, "pg_temp.f", "pg_temp.dim", "article_id", "price")
    assert res["status"] == "passed"
    assert res["fact_rows"] == res["joined_rows"] == 4
    assert res["fact_sum"] == pytest.approx(res["joined_sum"]) == pytest.approx(0.10)
    assert res["n_unmatched"] == 1
    pg_conn.rollback()


def test_join_trap_demonstration(pg_conn):
    demo = demo_join_trap(pg_conn)
    assert demo["kind"] == "demonstration_on_synthetic_data"
    assert demo["unchecked_join_rows"] > demo["fact_rows"]                 # rows multiplied
    assert demo["unchecked_join_sum_price"] > demo["fact_sum_price"]       # spend inflated
    assert demo["guard_on_duplicate_dim"].startswith("rejected")
    assert demo["corrected_join"]["joined_rows"] == demo["fact_rows"]
    assert demo["corrected_join"]["joined_sum"] == pytest.approx(demo["fact_sum_price"])
