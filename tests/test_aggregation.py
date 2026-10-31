"""SQL aggregation compared with an independent pandas computation on a fixed synthetic sample."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from psycopg import sql

from conftest import make_config
from hm_repeat_purchase import db
from hm_repeat_purchase import features as feat
from hm_repeat_purchase.features import build_dataset, load_features, load_labels, sample_bucket
from hm_repeat_purchase.ingest import ingest, raw_partition

pytestmark = pytest.mark.integration

CUTOFF = date(2020, 2, 26)


def reference_snapshot(tx: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Plain pandas: rows in [cutoff-90, cutoff) -> per customer features; written independently of the SQL."""
    t = tx.copy()
    t["t_dat"] = pd.to_datetime(t["t_dat"]).dt.date
    hist = t[(t["t_dat"] >= cutoff - timedelta(days=90)) & (t["t_dat"] < cutoff)]
    out = []
    for cust, g in hist.groupby("customer_id"):
        days = sorted(set(g["t_dat"]))
        last30 = [x for x in days if x >= cutoff - timedelta(days=30)]
        out.append({
            "customer_id": cust,
            "purchase_days_90d": len(days),
            "avg_scaled_spend_per_purchase_day_90d": g["price"].sum() / len(days),
            "recency_days": (cutoff - max(days)).days,
            "rows_90d": len(g),
            "spend_90d": g["price"].sum(),
            "purchase_days_30d": len(last30),
            "mean_gap_days_90d": ((max(days) - min(days)).days / (len(days) - 1)) if len(days) > 1 else np.nan,
            "share_rows_channel_2_90d": float((g["sales_channel_id"] == 2).mean()),
        })
    return pd.DataFrame(out).set_index("customer_id").sort_index()


def test_sql_snapshot_matches_pandas_reference(tmp_path, pg_conn, synthetic_data):
    cfg = make_config(tmp_path, synthetic_data["transactions"].parent)
    ingest(cfg, pg_conn)
    ds = build_dataset(cfg, pg_conn)["dataset_id"]
    tx = pd.read_csv(synthetic_data["transactions"], dtype={"customer_id": str, "article_id": str})
    ref = reference_snapshot(tx, CUTOFF)
    got = load_features(cfg, ds, [CUTOFF]).set_index("customer_id").sort_index()
    assert list(got.index) == list(ref.index)
    for col in ref.columns:
        np.testing.assert_allclose(got[col].to_numpy(float), ref[col].to_numpy(float), rtol=1e-12, equal_nan=True,
                                   err_msg=col)
    # Label: any purchase row in [cutoff, cutoff + 30).
    t = tx.assign(t_dat=pd.to_datetime(tx["t_dat"]).dt.date)
    fut = set(t[(t["t_dat"] >= CUTOFF) & (t["t_dat"] < CUTOFF + timedelta(days=30))]["customer_id"])
    lab = load_labels(cfg, ds, [CUTOFF], "development").merge(
        feat.load_tie_keys(cfg, ds)[["customer_sk", "customer_id"]], on="customer_sk")
    expected = lab["customer_id"].isin(fut).astype(int)
    assert (lab["target"].to_numpy() == expected.to_numpy()).all()


def test_customer_day_keeps_multiplicity_and_sums(tmp_path, pg_conn, synthetic_data):
    cfg = make_config(tmp_path, synthetic_data["transactions"].parent)
    ingest(cfg, pg_conn)
    ds = build_dataset(cfg, pg_conn)["dataset_id"]
    tx = pd.read_csv(synthetic_data["transactions"], dtype={"customer_id": str, "article_id": str})
    ref = tx.groupby(["customer_id", "t_dat"]).agg(n_rows=("price", "size"), spend=("price", "sum"),
                                                     n_articles=("article_id", "nunique")).reset_index()
    rows = db.fetch_all(pg_conn, sql.SQL(
        "SELECT k.customer_id, d.t_dat::text AS t_dat, d.n_rows, d.spend, d.n_distinct_articles FROM {} d"
        " JOIN {} k USING (customer_sk) ORDER BY 1, 2").format(feat.part("day", ds), feat.part("key", ds)))
    got = pd.DataFrame(rows)
    assert len(got) == len(ref)
    m = got.merge(ref, on=["customer_id", "t_dat"], validate="one_to_one")
    assert (m["n_rows_x"] == m["n_rows_y"]).all()
    np.testing.assert_allclose(m["spend_x"], m["spend_y"], rtol=1e-12)
    assert (m["n_distinct_articles"] == m["n_articles"]).all()
    # Identical rows exist in the fixture and are all kept.
    dup = tx.duplicated(keep=False).sum()
    assert dup > 0 and int(got["n_rows"].sum()) == len(tx)


def test_article_id_keeps_leading_zeros(tmp_path, pg_conn, synthetic_data):
    cfg = make_config(tmp_path, synthetic_data["transactions"].parent)
    res = ingest(cfg, pg_conn)
    part = raw_partition("transactions", res["files"]["transactions"]["load_id"])
    ids = {r["article_id"] for r in db.fetch_all(pg_conn, sql.SQL("SELECT DISTINCT article_id FROM {}")
                                                 .format(db.ident(part)))}
    assert "0000000001" in ids and all(len(i) == 10 for i in ids)


def test_sql_sampling_matches_python_hash(tmp_path, pg_conn, synthetic_data):
    cfg = make_config(tmp_path, synthetic_data["transactions"].parent, dataset__sample_fraction=0.3)
    ingest(cfg, pg_conn)
    ds = build_dataset(cfg, pg_conn)["dataset_id"]
    got = set(feat.load_tie_keys(cfg, ds)["customer_id"])
    tx = pd.read_csv(synthetic_data["transactions"], dtype={"customer_id": str})
    cust = pd.read_csv(synthetic_data["customers"], dtype={"customer_id": str})
    ids = set(tx["customer_id"]) | set(cust["customer_id"])
    expected = {c for c in ids if sample_bucket(c, "hmrp-sample-v1", 10000) < 3000}
    assert got == expected and 0 < len(got) < len(ids)
