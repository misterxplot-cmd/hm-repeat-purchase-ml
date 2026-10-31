"""Dataset build: customer key map, customer_day, calendar, snapshots, labels, Parquet export."""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

from . import db, splits
from .config import Config
from .ingest import ROLES, active_load_ids, raw_partition
from .tracking import relpath, utc_now, write_json

log = logging.getLogger(__name__)

FEATURE_BUILD_VERSION = "snap_v1"

CORE_FEATURES = ["purchase_days_90d", "avg_scaled_spend_per_purchase_day_90d", "recency_days"]
EXTENDED_FEATURES = CORE_FEATURES + [
    "purchase_days_7d", "purchase_days_30d",
    "rows_7d", "rows_30d", "rows_90d",
    "spend_7d", "spend_30d", "spend_90d",
    "avg_price_per_row_90d", "first_purchase_age_days_90d", "mean_gap_days_90d",
    "share_rows_channel_2_90d", "recent_share_purchase_days_30d",
]
FEATURE_SETS = {"core_v1": CORE_FEATURES, "ext_v1": EXTENDED_FEATURES}
AUDIT_COLUMNS = ["cutoff_date", "customer_sk", "customer_id", "history_start", "history_end_exclusive",
                 "feature_build_version", "eligibility_reason"]
LABEL_COLUMNS = ["cutoff_date", "customer_sk", "target_start", "target_end_exclusive", "target"]

PARENTS = {
    "key": "derived.customer_key",
    "day": "derived.customer_day",
    "snap": "derived.snapshot",
    "label": "derived.label",
}


def part(kind: str, dataset_id: str) -> sql.Identifier:
    return db.ident(db.partition_name(PARENTS[kind], dataset_id))


def dataset_dir(config: Config, dataset_id: str) -> Path:
    return config.derived_dir / dataset_id


def tie_key(customer_ids: pd.Series, salt: str) -> np.ndarray:
    """Deterministic, target-independent tie-break key: first 7 bytes of sha256(salt:customer_id)."""
    salt_b = salt.encode("utf-8") + b":"
    return np.fromiter(
        (int.from_bytes(hashlib.sha256(salt_b + c.encode("utf-8")).digest()[:7], "big") for c in customer_ids),
        dtype=np.int64, count=len(customer_ids),
    )


def sample_bucket(customer_id: str, salt: str, buckets: int) -> int:
    """Python twin of the SQL sampling rule in customer_day.sql (used in tests)."""
    digest = hashlib.sha256(f"{salt}:{customer_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


# --------------------------------------------------------------------------- identity

def dataset_identity(config: Config, conn: psycopg.Connection) -> tuple[str, dict]:
    loads = active_load_ids(conn)
    missing = [r for r in ROLES if r not in loads]
    if missing:
        raise RuntimeError(f"no completed load for {missing}; run `hmrp ingest` first")
    load_rows = db.fetch_all(conn, "SELECT load_id, file_role, sha256, schema_version, rows_loaded FROM meta.load"
                                   " WHERE load_id = ANY(%(ids)s)", {"ids": list(loads.values())})
    fraction = float(config.get("dataset.sample_fraction"))
    buckets = int(config.get("dataset.hash_buckets"))
    threshold = int(round(fraction * buckets))
    if not 0 < threshold <= buckets:
        raise ValueError(f"sample_fraction {fraction} gives an empty sample with {buckets} buckets")
    identity = {
        "mode": config.mode,
        "sources": {r["file_role"]: {"sha256": r["sha256"], "ingest_schema": r["schema_version"],
                                     "load_id": r["load_id"], "rows_loaded": r["rows_loaded"]}
                    for r in sorted(load_rows, key=lambda r: r["file_role"])},
        "sampling": {
            "fraction": fraction,
            "hash_rule": "int(sha256(salt + ':' + customer_id).hex[:8], 16) % buckets < threshold",
            "salt": config.get("dataset.hash_salt"),
            "buckets": buckets,
            "threshold": threshold,
        },
        "feature_build_version": FEATURE_BUILD_VERSION,
        "time": {k: config.get(f"time.{k}") for k in ("history_days", "horizon_days", "step_days",
                                                      "grid_start", "grid_end")},
    }
    blob = json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    dataset_id = f"{config.mode}_{hashlib.sha256(blob).hexdigest()[:10]}"
    return dataset_id, identity


# --------------------------------------------------------------------------- helpers

def _exec(conn: psycopg.Connection, query, params: dict | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(query, params or {})


def _explain_analyze(conn: psycopg.Connection, query, params: dict) -> str:
    """Runs the statement once (it is part of the planned build) and returns the plan with timings."""
    with psycopg.ClientCursor(conn) as cur:
        cur.execute(sql.SQL("EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) ") + query, params)
        return "\n".join(r[0] for r in cur.fetchall())


def _drop_dataset(conn: psycopg.Connection, dataset_id: str) -> None:
    for kind in ("label", "snap", "day", "key"):
        _exec(conn, sql.SQL("DROP TABLE IF EXISTS {}").format(part(kind, dataset_id)))
    _exec(conn, "DELETE FROM derived.calendar WHERE dataset_id = %(d)s", {"d": dataset_id})
    _exec(conn, "DELETE FROM meta.dataset WHERE dataset_id = %(d)s", {"d": dataset_id})
    conn.commit()


def _copy_csv_df(conn: psycopg.Connection, query, dtypes: dict, date_cols: list[str]) -> pd.DataFrame:
    raw = db.copy_to_bytes(conn, query)
    df = pd.read_csv(io.BytesIO(raw), dtype=dtypes)
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.date
    return df


FEATURE_DTYPES = {
    "customer_sk": "int32", "customer_id": "string", "feature_build_version": "string",
    "eligibility_reason": "string",
    "purchase_days_90d": "int32", "recency_days": "int32", "purchase_days_7d": "int32",
    "purchase_days_30d": "int32", "rows_7d": "int32", "rows_30d": "int32", "rows_90d": "int32",
    "first_purchase_age_days_90d": "int32",
    "avg_scaled_spend_per_purchase_day_90d": "float64", "spend_7d": "float64", "spend_30d": "float64",
    "spend_90d": "float64", "avg_price_per_row_90d": "float64", "mean_gap_days_90d": "float64",
    "share_rows_channel_2_90d": "float64", "recent_share_purchase_days_30d": "float64",
}


def features_query(dataset_id: str, cutoff: date, history_days: int) -> sql.Composed:
    return db.sql_block("customer_snapshots.sql", "snapshot_features_select", day_part=part("day", dataset_id),
                        cutoff=cutoff, history_start=cutoff - timedelta(days=history_days))


# --------------------------------------------------------------------------- coverage

def source_coverage(conn: psycopg.Connection, tx_load_id: int, truncation_ratio: float = 0.5) -> dict[str, Any]:
    """Calendar coverage of the full source (not the sample): missing dates, dips, last-day check."""
    rows = db.fetch_all(conn, sql.SQL("SELECT t_dat, count(*) AS n FROM {} GROUP BY t_dat ORDER BY t_dat")
                        .format(db.ident(raw_partition("transactions", tx_load_id))))
    daily = pd.Series({r["t_dat"]: int(r["n"]) for r in rows})
    full_range = pd.date_range(min(daily.index), max(daily.index), freq="D").date
    missing = [d for d in full_range if d not in daily.index]
    rolling_median = daily.rolling(28, min_periods=7).median().shift(1)
    ratio = (daily / rolling_median).dropna()
    dips = ratio[ratio < 0.5]
    last_day = max(daily.index)
    last_ratio = float(ratio.get(last_day, float("nan")))
    observation_end = last_day + timedelta(days=1)
    decision = "max(t_dat) + 1 day: no missing dates and the last day is not truncated"
    if missing:
        decision = f"max(t_dat) + 1 day; {len(missing)} missing calendar dates recorded"
    if not math.isnan(last_ratio) and last_ratio < truncation_ratio:
        observation_end = last_day
        decision = f"last day {last_day} has {last_ratio:.2f} of the trailing median volume: excluded as possibly truncated"
    return {
        "min_date": min(daily.index), "max_date": last_day, "n_dates_with_rows": int(len(daily)),
        "n_calendar_days": int(len(full_range)), "missing_dates": missing,
        "low_volume_days_ratio_below_0_5": {str(d): round(float(v), 3) for d, v in dips.items()},
        "last_day_rows": int(daily[last_day]), "last_day_ratio_to_trailing_28d_median": last_ratio,
        "observation_end_exclusive": observation_end, "observation_end_decision": decision,
        "truncation_ratio_threshold": truncation_ratio,
    }


# --------------------------------------------------------------------------- build

def _trunc(config: Config) -> float:
    return float(config.get("time.truncation_ratio_threshold", 0.5))


def build_dataset(config: Config, conn: psycopg.Connection, rebuild: bool = False) -> dict[str, Any]:
    db.ensure_schema(conn)
    dataset_id, identity = dataset_identity(config, conn)
    existing = db.fetch_one(conn, "SELECT status FROM meta.dataset WHERE dataset_id = %(d)s", {"d": dataset_id})
    out_dir = dataset_dir(config, dataset_id)
    summary_path = config.reports_dir / "data" / dataset_id / "dataset_summary.json"
    if existing.get("status") == "completed" and not rebuild:
        if (out_dir / "_SUCCESS").exists() and summary_path.exists():
            log.info("dataset %s already built; reusing (pass --rebuild to rebuild)", dataset_id)
            return {"dataset_id": dataset_id, "action": "reused", "summary": relpath(summary_path)}
        log.info("dataset %s in DB but exports missing; re-exporting", dataset_id)
        return _export_and_summarize(config, conn, dataset_id, identity, timings={}, explain={}, action="re-exported")
    if existing:
        _drop_dataset(conn, dataset_id)

    timings: dict[str, float] = {}
    explain: dict[str, str] = {}
    src = identity["sources"]
    params = {
        "dataset_id": dataset_id,
        "transactions_load_id": src["transactions"]["load_id"],
        "customers_load_id": src["customers"]["load_id"],
        "salt": identity["sampling"]["salt"],
        "buckets": identity["sampling"]["buckets"],
        "threshold": identity["sampling"]["threshold"],
    }
    _exec(conn, "INSERT INTO meta.dataset (dataset_id, mode, sample_fraction, hash_rule, transactions_load_id,"
                " customers_load_id, articles_load_id, identity, status) VALUES (%(d)s, %(m)s, %(f)s, %(h)s,"
                " %(t)s, %(c)s, %(a)s, %(i)s, 'building')",
          {"d": dataset_id, "m": config.mode, "f": identity["sampling"]["fraction"],
           "h": identity["sampling"]["hash_rule"], "t": src["transactions"]["load_id"],
           "c": src["customers"]["load_id"], "a": src["articles"]["load_id"], "i": json.dumps(identity, default=str)})
    for kind in ("key", "day", "snap", "label"):
        _exec(conn, db.sql_block("customer_day.sql", "create_dataset_partition", part=part(kind, dataset_id),
                                 parent=db.ident(PARENTS[kind]), dataset_id=dataset_id))
    conn.commit()

    t0 = time.perf_counter()
    _exec(conn, db.sql_block("customer_day.sql", "customer_key", key_part=part("key", dataset_id)), params)
    _exec(conn, db.sql_block("customer_day.sql", "customer_key_constraints", key_part=part("key", dataset_id)))
    conn.commit()
    timings["customer_key"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    day_query = db.sql_block("customer_day.sql", "customer_day", day_part=part("day", dataset_id),
                             key_part=part("key", dataset_id))
    if config.get("sql.explain_analyze_heavy"):
        explain["customer_day_insert"] = _explain_analyze(conn, day_query, params)
    else:
        _exec(conn, day_query, params)
    conn.commit()
    _exec(conn, db.sql_block("customer_day.sql", "customer_day_constraints", day_part=part("day", dataset_id)))
    conn.commit()
    timings["customer_day"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    coverage = source_coverage(conn, src["transactions"]["load_id"], _trunc(config))
    calendar = splits.build_calendar(config, coverage["min_date"], coverage["observation_end_exclusive"])
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO derived.calendar (dataset_id, cutoff_date, history_start, history_end_exclusive, target_start,"
            " target_end_exclusive, role, is_eligible, eligibility_reason) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [(dataset_id, *(date.fromisoformat(r[k]) for k in ("cutoff_date", "history_start", "history_end_exclusive",
                                                               "target_start", "target_end_exclusive")),
              r["role"], r["is_eligible"], r["eligibility_reason"]) for r in calendar])
    conn.commit()
    timings["coverage_calendar"] = time.perf_counter() - t0

    history_days = int(config.get("time.history_days"))
    t0 = time.perf_counter()
    for i, cutoff in enumerate(splits.eligible_cutoffs(calendar)):
        w = splits.config_window(config, cutoff)
        if w.target_end_exclusive > coverage["observation_end_exclusive"]:
            raise RuntimeError(f"cutoff {cutoff}: target window not observed")
        q = db.sql_block("customer_snapshots.sql", "insert_snapshot", snap_part=part("snap", dataset_id),
                         features=features_query(dataset_id, cutoff, history_days))
        p = {"dataset_id": dataset_id, "cutoff": cutoff, "history_start": w.history_start,
             "feature_build_version": FEATURE_BUILD_VERSION}
        t_cut = time.perf_counter()
        explain_this = i == 0 and config.get("sql.explain_analyze_heavy")
        if explain_this:
            explain[f"snapshot_insert_{cutoff}"] = _explain_analyze(conn, q, p)
        else:
            _exec(conn, q, p)
        # Fresh statistics keep the label join a hash join; without them the growing, unindexed
        # partition is estimated as empty and the planner picks a nested loop (measured: ~10 min/cutoff).
        _exec(conn, sql.SQL("ANALYZE {}").format(part("snap", dataset_id)))
        lq = db.sql_block("labels.sql", "insert_labels", label_part=part("label", dataset_id),
                          snap_part=part("snap", dataset_id), day_part=part("day", dataset_id))
        lp = {"dataset_id": dataset_id, "cutoff": cutoff, "target_end": w.target_end_exclusive}
        if explain_this:
            explain[f"label_insert_{cutoff}"] = _explain_analyze(conn, lq, lp)
        else:
            _exec(conn, lq, lp)
        conn.commit()
        timings[f"snapshot_label_{cutoff}"] = time.perf_counter() - t_cut
    _exec(conn, db.sql_block("customer_snapshots.sql", "snapshot_constraints", snap_part=part("snap", dataset_id)))
    _exec(conn, db.sql_block("labels.sql", "label_constraints", label_part=part("label", dataset_id)))
    conn.commit()
    timings["snapshots_labels"] = time.perf_counter() - t0

    result = _export_and_summarize(config, conn, dataset_id, identity, timings, explain, action="built",
                                   coverage=coverage, calendar=calendar)
    _exec(conn, "UPDATE meta.dataset SET status = 'completed', completed_at = now() WHERE dataset_id = %(d)s",
          {"d": dataset_id})
    conn.commit()
    return result


def _validate_dataset(config: Config, conn: psycopg.Connection, dataset_id: str, identity: dict,
                      observation_end: date) -> dict[str, Any]:
    params = {"transactions_load_id": identity["sources"]["transactions"]["load_id"],
              "observation_end_exclusive": observation_end, "history_days": int(config.get("time.history_days"))}
    recon = db.fetch_one(conn, db.sql_block("validation_checks.sql", "customer_day_reconciliation",
                                            key_part=part("key", dataset_id), day_part=part("day", dataset_id)), params)
    recon = {k: float(v) if "sum" in k else int(v) for k, v in recon.items()}
    recon["rows_match"] = recon["raw_rows"] == recon["day_rows"]
    recon["sum_match_rel_1e-9"] = math.isclose(recon["raw_sum_price"], recon["day_sum_price"], rel_tol=1e-9)
    alignment = db.fetch_all(conn, db.sql_block("validation_checks.sql", "snapshot_label_alignment",
                                                snap_part=part("snap", dataset_id), label_part=part("label", dataset_id)))
    immature = db.fetch_one(conn, db.sql_block("validation_checks.sql", "label_maturity",
                                               label_part=part("label", dataset_id)), params)
    bounds = db.fetch_one(conn, db.sql_block("validation_checks.sql", "history_bounds",
                                             snap_part=part("snap", dataset_id)), params)
    key_uniq = db.fetch_one(conn, sql.SQL("SELECT count(*) AS n, count(DISTINCT customer_id) AS n_ids,"
                                          " count(DISTINCT customer_sk) AS n_sk FROM {}").format(part("key", dataset_id)))
    checks = {
        "customer_day_reconciliation": recon,
        "customer_key_one_to_one": int(key_uniq["n"]) == int(key_uniq["n_ids"]) == int(key_uniq["n_sk"]),
        "snapshot_label_one_to_one": all(int(r["n_snapshot"]) == int(r["n_labelled"]) for r in alignment),
        "n_immature_labels": int(immature["n_immature"]),
        "n_history_bound_violations": int(bounds["n_violations"]),
    }
    failed = [k for k, ok in [("rows_match", recon["rows_match"]), ("sum_match", recon["sum_match_rel_1e-9"]),
                              ("customer_key_one_to_one", checks["customer_key_one_to_one"]),
                              ("snapshot_label_one_to_one", checks["snapshot_label_one_to_one"]),
                              ("immature_labels", checks["n_immature_labels"] == 0),
                              ("history_bounds", checks["n_history_bound_violations"] == 0)] if not ok]
    checks["failed"] = failed
    if failed:
        raise RuntimeError(f"dataset validation failed: {failed}; details: {checks}")
    return checks, alignment


def _export_and_summarize(config: Config, conn: psycopg.Connection, dataset_id: str, identity: dict,
                          timings: dict, explain: dict, action: str, coverage: dict | None = None,
                          calendar: list[dict] | None = None) -> dict[str, Any]:
    if coverage is None:
        coverage = source_coverage(conn, identity["sources"]["transactions"]["load_id"], _trunc(config))
    if calendar is None:
        calendar = db.fetch_all(conn, "SELECT * FROM derived.calendar WHERE dataset_id = %(d)s ORDER BY cutoff_date",
                                {"d": dataset_id})
        calendar = [{k: (v.isoformat() if isinstance(v, date) else v) for k, v in r.items() if k != "dataset_id"}
                    for r in calendar]
    checks, alignment = _validate_dataset(config, conn, dataset_id, identity, coverage["observation_end_exclusive"])

    t0 = time.perf_counter()
    out_dir = dataset_dir(config, dataset_id)
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    keys = _copy_csv_df(conn, sql.SQL("COPY (SELECT customer_sk, customer_id, in_customers_csv::int AS in_customers_csv,"
                                      " in_transactions::int AS in_transactions FROM {}"
                                      " ORDER BY customer_sk) TO STDOUT WITH (FORMAT csv, HEADER true)")
                        .format(part("key", dataset_id)), {"customer_sk": "int32", "customer_id": "string"}, [])
    keys["tie_key"] = tie_key(keys["customer_id"], str(config.get("evaluation.tie_salt")))
    keys.to_parquet(out_dir / "customer_keys.parquet", index=False)
    cohort = []
    for cutoff in splits.eligible_cutoffs(calendar):
        f = _copy_csv_df(conn, db.sql_block("customer_snapshots.sql", "export_snapshot",
                                            snap_part=part("snap", dataset_id), key_part=part("key", dataset_id),
                                            cutoff=cutoff), FEATURE_DTYPES,
                         ["cutoff_date", "history_start", "history_end_exclusive"])
        lab = _copy_csv_df(conn, db.sql_block("labels.sql", "export_labels", label_part=part("label", dataset_id),
                                              cutoff=cutoff),
                           {"customer_sk": "int32", "target": "int8"},
                           ["cutoff_date", "target_start", "target_end_exclusive"])
        f.to_parquet(out_dir / "features" / f"cutoff={cutoff}.parquet", index=False)
        lab.to_parquet(out_dir / "labels" / f"cutoff={cutoff}.parquet", index=False)
        role = next(r["role"] for r in calendar if r["cutoff_date"] == cutoff.isoformat())
        cohort.append({"cutoff_date": cutoff, "role": role, "n_rows": int(len(f)),
                       "n_customers": int(f["customer_sk"].nunique()), "n_positive": int(lab["target"].sum()),
                       "prevalence": float(lab["target"].mean()) if len(lab) else None})
    timings["export_parquet"] = time.perf_counter() - t0

    distribution = db.fetch_all(conn, db.sql_block("validation_checks.sql", "customer_day_distribution",
                                                   day_part=part("day", dataset_id)))
    dist = pd.DataFrame(distribution).astype(int)
    n_with = int(dist["n_customers"].sum())
    sizes = db.fetch_all(conn, """
        SELECT c.relname AS table_name, pg_total_relation_size(c.oid) AS bytes
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname IN ('raw', 'derived') AND c.relkind = 'r' AND c.relname LIKE %(p)s
        ORDER BY 1""", {"p": f"%{dataset_id}%"})
    summary = {
        "dataset_id": dataset_id,
        "action": action,
        "built_at": utc_now(),
        "identity": identity,
        "coverage": coverage,
        "calendar": calendar,
        "cohort_by_cutoff": cohort,
        "n_snapshot_rows_total": int(sum(c["n_rows"] for c in cohort)),
        "checks": checks,
        "customer_keys": {"n_keys": int(len(keys)), "n_in_customers_csv": int(keys["in_customers_csv"].sum()),
                          "n_in_transactions": int(keys["in_transactions"].sum())},
        "purchase_days_per_customer": {
            "n_customers_with_purchases": n_with,
            "n_customers_with_2plus_purchase_days": int(dist.loc[dist["purchase_days"] >= 2, "n_customers"].sum()),
            "share_2plus": float(dist.loc[dist["purchase_days"] >= 2, "n_customers"].sum() / n_with) if n_with else None,
            "quantiles": _weighted_quantiles(dist["purchase_days"].to_numpy(), dist["n_customers"].to_numpy(),
                                             [0.25, 0.5, 0.75, 0.9, 0.99]),
            "histogram_capped_50": _capped_hist(dist, 50),
        },
        "table_sizes_bytes": {r["table_name"]: int(r["bytes"]) for r in sizes},
        "timings_seconds": {k: round(v, 2) for k, v in timings.items()},
        "exports": {"dir": relpath(out_dir), "n_feature_files": len(cohort)},
    }
    rep_dir = config.reports_dir / "data" / dataset_id
    write_json(rep_dir / "dataset_summary.json", summary)
    for name, plan in explain.items():
        (rep_dir / f"explain_{name}.txt").write_text(plan + "\n", encoding="utf-8")
    (out_dir / "_SUCCESS").write_text(utc_now() + "\n", encoding="utf-8")
    write_json(out_dir / "identity.json", {"dataset_id": dataset_id, **identity})
    return {"dataset_id": dataset_id, "action": action, "summary": relpath(rep_dir / "dataset_summary.json")}


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs: list[float]) -> dict[str, float]:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w) / w.sum()
    return {str(q): float(v[np.searchsorted(cum, q)]) for q in qs}


def _capped_hist(dist: pd.DataFrame, cap: int) -> dict[str, int]:
    d = dist.copy()
    d["bucket"] = np.where(d["purchase_days"] >= cap, f"{cap}+", d["purchase_days"].astype(str))
    return {str(k): int(v) for k, v in d.groupby("bucket", sort=False)["n_customers"].sum().items()}


# --------------------------------------------------------------------------- loading for modelling

class TestAccessError(RuntimeError):
    pass


def latest_dataset_id(config: Config, conn: psycopg.Connection | None = None) -> str:
    if conn is not None:
        dataset_id, _ = dataset_identity(config, conn)
        return dataset_id
    candidates = sorted((p for p in config.derived_dir.glob(f"{config.mode}_*") if (p / "_SUCCESS").exists()),
                        key=lambda p: (p / "_SUCCESS").stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"no built dataset for mode {config.mode}; run `hmrp build-features`")
    return candidates[-1].name


def load_calendar(config: Config, dataset_id: str) -> list[dict]:
    summary = json.loads((config.reports_dir / "data" / dataset_id / "dataset_summary.json").read_text(encoding="utf-8"))
    return summary["calendar"]


def load_features(config: Config, dataset_id: str, cutoffs: list[date], columns: list[str] | None = None) -> pd.DataFrame:
    base = dataset_dir(config, dataset_id) / "features"
    frames = [pd.read_parquet(base / f"cutoff={c}.parquet", columns=columns) for c in cutoffs]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_labels(config: Config, dataset_id: str, cutoffs: list[date], purpose: str) -> pd.DataFrame:
    """Labels of the test cutoff are only readable with purpose='final_test' (evaluate stage)."""
    test_cutoff = config.get("time.test_cutoff")
    if test_cutoff in cutoffs and purpose != "final_test":
        raise TestAccessError(f"labels of test cutoff {test_cutoff} requested for purpose={purpose!r}")
    base = dataset_dir(config, dataset_id) / "labels"
    frames = [pd.read_parquet(base / f"cutoff={c}.parquet") for c in cutoffs]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


MODEL_FRAME_COLUMNS = ["cutoff_date", "customer_sk", *EXTENDED_FEATURES]


def load_labelled(config: Config, dataset_id: str, cutoffs: list[date], purpose: str,
                  columns: list[str] | None = None) -> pd.DataFrame:
    feats = load_features(config, dataset_id, cutoffs, columns)
    labels = load_labels(config, dataset_id, cutoffs, purpose)
    merged = feats.merge(labels[["cutoff_date", "customer_sk", "target", "target_start", "target_end_exclusive"]],
                         on=["cutoff_date", "customer_sk"], how="left", validate="one_to_one")
    if merged["target"].isna().any():
        raise RuntimeError("snapshot rows without labels")
    merged["target"] = merged["target"].astype("int8")
    return merged


def load_tie_keys(config: Config, dataset_id: str) -> pd.DataFrame:
    return pd.read_parquet(dataset_dir(config, dataset_id) / "customer_keys.parquet",
                           columns=["customer_sk", "customer_id", "tie_key"])
