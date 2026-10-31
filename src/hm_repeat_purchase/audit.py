"""Source data audit over raw partitions: sizes, keys, dates, prices, duplicates, channels, coverage, JOINs."""
from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any

import pandas as pd
import psycopg

from . import db, plots
from .config import Config
from .ingest import active_load_ids, raw_partition
from .joins import JoinCardinalityError, checked_many_to_one
from .tracking import relpath, utc_now, write_json

log = logging.getLogger(__name__)


def _num(v: Any) -> Any:
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, list):
        return [_num(x) for x in v]
    return v


def _q(conn: psycopg.Connection, name: str, timings: dict, params: dict | None = None, **parts) -> list[dict]:
    t0 = time.perf_counter()
    rows = db.fetch_all(conn, db.sql_block("audit.sql", name, **parts), params)
    timings[name] = round(time.perf_counter() - t0, 2)
    return [{k: _num(v) for k, v in r.items()} for r in rows]


def run_audit(config: Config, conn: psycopg.Connection) -> dict[str, Any]:
    loads = active_load_ids(conn)
    parts = {
        "tx": db.ident(raw_partition("transactions", loads["transactions"])),
        "cust": db.ident(raw_partition("customers", loads["customers"])),
        "art": db.ident(raw_partition("articles", loads["articles"])),
    }
    t: dict[str, float] = {}
    load_rows = db.fetch_all(conn, "SELECT l.*, s.file_name, s.size_bytes FROM meta.load l JOIN meta.source_file s"
                                   " USING (sha256) WHERE load_id = ANY(%(ids)s) ORDER BY load_id",
                             {"ids": list(loads.values())})
    out: dict[str, Any] = {"generated_at": utc_now(), "loads": load_rows}
    out["transactions"] = _q(conn, "tx_overview", t, tx=parts["tx"])[0]
    out["price"] = _q(conn, "tx_price_stats", t, tx=parts["tx"])[0]
    out["price"]["quantile_levels"] = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    out["channels"] = _q(conn, "tx_channels", t, tx=parts["tx"])
    monthly = pd.DataFrame(_q(conn, "tx_monthly_channel", t, tx=parts["tx"]))
    daily = pd.DataFrame(_q(conn, "tx_daily_by_channel", t, tx=parts["tx"]))
    hist = pd.DataFrame(_q(conn, "tx_price_log_histogram", t, tx=parts["tx"]))
    out["identical_rows"] = _q(conn, "tx_identical_rows", t, tx=parts["tx"])
    dup_extra = sum(int(r["n_rows"]) - int(r["n_groups"]) for r in out["identical_rows"])
    out["identical_rows_summary"] = {
        "rows_in_groups_with_multiplicity_gt_1": int(sum(int(r["n_rows"]) for r in out["identical_rows"]
                                                        if int(r["multiplicity"]) > 1)),
        "rows_beyond_first_in_group": int(dup_extra),
        "policy": "kept: no order_id/quantity in source, identical rows may be several units",
    }
    out["coverage"] = {
        "customers": _q(conn, "tx_customer_coverage", t, tx=parts["tx"], cust=parts["cust"])[0],
        "articles": _q(conn, "tx_article_coverage", t, tx=parts["tx"], art=parts["art"])[0],
    }
    out["customers_csv"] = _q(conn, "customers_overview", t, cust=parts["cust"])[0]
    out["customers_categoricals"] = _q(conn, "customers_categoricals", t, cust=parts["cust"])
    out["articles_csv"] = _q(conn, "articles_overview", t, art=parts["art"])[0]
    out["rejected_by_reason"] = _q(conn, "rejected_by_reason", t, {"load_id": loads["transactions"]})

    joins: dict[str, Any] = {}
    tx_name = raw_partition("transactions", loads["transactions"])
    for name, dim, key in (("transactions_to_articles", raw_partition("articles", loads["articles"]), "article_id"),
                           ("transactions_to_customers", raw_partition("customers", loads["customers"]),
                            "customer_id")):
        t0 = time.perf_counter()
        try:
            joins[name] = checked_many_to_one(conn, tx_name, dim, key, "price")
        except JoinCardinalityError as exc:
            joins[name] = {"status": "rejected", "error": str(exc)}
        joins[name]["seconds"] = round(time.perf_counter() - t0, 2)
    out["join_checks"] = joins

    dates = pd.to_datetime(daily.groupby("t_dat")["n_rows"].sum().index)
    full = pd.date_range(dates.min(), dates.max(), freq="D")
    out["calendar"] = {"n_dates_with_rows": int(len(dates)), "n_calendar_days": int(len(full)),
                       "missing_dates": [d.date().isoformat() for d in full.difference(dates)]}
    daily_tot = daily.groupby("t_dat")["n_rows"].sum()
    out["daily_rows_distribution"] = {k: float(v) for k, v in daily_tot.describe().items()}
    out["query_seconds"] = t

    rep = config.reports_dir / "data"
    fig_dir = rep / "figures"
    daily.to_csv(rep / "source_daily_by_channel.csv", index=False)
    monthly.to_csv(rep / "source_monthly_by_channel.csv", index=False)
    hist.to_csv(rep / "source_price_log10_histogram.csv", index=False)
    plots.daily_rows_by_channel(daily, fig_dir / "daily_rows_by_channel.png")
    plots.monthly_channel_share(monthly, fig_dir / "monthly_channel_share.png")
    plots.price_histogram(hist, fig_dir / "price_log10_histogram.png")
    out["figures"] = [relpath(p) for p in sorted(fig_dir.glob("*.png"))]
    write_json(rep / "source_audit.json", out)
    log.info("audit written to %s", relpath(rep / "source_audit.json"))
    return out
