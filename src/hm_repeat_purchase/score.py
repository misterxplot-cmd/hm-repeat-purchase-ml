"""Batch scoring for a CRM-style export at an as_of_date. Never reads or writes target."""
from __future__ import annotations

import hashlib
import io
import logging
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import psycopg

from . import db
from . import features as feat
from . import metrics
from .calibrate import apply_platt
from .config import Config
from .tracking import relpath, read_json, resolve_run_id, run_paths, utc_now, write_json
from .train import load_frozen

log = logging.getLogger(__name__)

EXPORT_COLUMNS = ["customer_id", "as_of_date", "horizon_days", "raw_score", "raw_probability",
                  "calibrated_probability", "rank", "selected_for_top_segment", "model_version",
                  "feature_set_version", "run_id"]


class ScoringValidityError(RuntimeError):
    pass


def scoring_features(config: Config, conn: psycopg.Connection, dataset_id: str, as_of: date) -> pd.DataFrame:
    """Features at as_of from the same SQL as the training snapshots (no labels involved)."""
    q = db.sql_block("customer_snapshots.sql", "export_scoring_features", key_part=feat.part("key", dataset_id),
                     features=feat.features_query(dataset_id, as_of, int(config.get("time.history_days"))))
    raw = db.copy_to_bytes(conn, q)
    return pd.read_csv(io.BytesIO(raw), dtype=feat.FEATURE_DTYPES)


def check_validity(as_of: date, model_meta: dict, calibration: dict, decision: dict, coverage: dict,
                   history_days: int, horizon_days: int) -> dict[str, Any]:
    """The model, its calibrator and the selection must only rely on labels matured by as_of."""
    max_dev_label_end = max(date.fromisoformat(f["validation_cutoff"]) for f in decision["folds"]) + timedelta(
        days=horizon_days)
    checks = {
        "max_train_label_end_exclusive": model_meta["max_train_label_end_exclusive"],
        "calibration_target_end_exclusive": calibration["calibration_target_end_exclusive"],
        "selection_label_end_exclusive": max_dev_label_end.isoformat(),
        "history_start": (as_of - timedelta(days=history_days)).isoformat(),
        "data_min_date": str(coverage["min_date"]),
        "observation_end_exclusive": str(coverage["observation_end_exclusive"]),
    }
    errors = []
    if date.fromisoformat(model_meta["max_train_label_end_exclusive"]) > as_of:
        errors.append("model trained on labels that end after as_of_date")
    if date.fromisoformat(calibration["calibration_target_end_exclusive"]) > as_of:
        errors.append("calibrator fitted on labels that end after as_of_date; no earlier calibrator version exists")
    if max_dev_label_end > as_of:
        errors.append("model selection used labels that end after as_of_date")
    if as_of - timedelta(days=history_days) < date.fromisoformat(str(coverage["min_date"])):
        errors.append("history window starts before the first observed date")
    if as_of > date.fromisoformat(str(coverage["observation_end_exclusive"])):
        errors.append("as_of_date is after the end of observed data: features would be incomplete")
    checks["errors"] = errors
    checks["valid"] = not errors
    return checks


def score(config: Config, conn: psycopg.Connection, as_of: date, run_id: str | None = None,
          model_name: str | None = None) -> dict[str, Any]:
    run_id = resolve_run_id(config, run_id)
    paths = run_paths(config, run_id)
    decision = read_json(paths["published"] / "selection_decision.json")
    calibration = read_json(paths["published"] / "calibration.json")
    model_name = model_name or decision["selected_model"]
    dataset_id = decision["dataset_id"]
    summary = read_json(config.reports_dir / "data" / dataset_id / "dataset_summary.json")
    model, meta = load_frozen(paths, model_name)
    horizon = int(config.get("time.horizon_days"))
    validity = check_validity(as_of, meta, calibration, decision, summary["coverage"],
                              int(config.get("time.history_days")), horizon)
    if not validity["valid"]:
        raise ScoringValidityError(f"as_of_date {as_of} not allowed for {meta['model_version']}: {validity['errors']}")

    df = scoring_features(config, conn, dataset_id, as_of)
    if "target" in df.columns:
        raise AssertionError("scoring frame must not contain target")
    ties = feat.load_tie_keys(config, dataset_id)[["customer_sk", "tie_key"]]
    df = df.merge(ties, on="customer_sk", how="left", validate="one_to_one")
    raw_score = model.raw_score(df)
    raw_prob = model.raw_probability(df)
    cal = calibration["models"][model_name]
    cal_prob = apply_platt(raw_score, cal["params"])
    order = metrics.ranking_order(raw_score, df["tie_key"].to_numpy())
    rank = np.empty(len(df), dtype=np.int64)
    rank[order] = np.arange(1, len(df) + 1)
    primary = float(config.get("evaluation.primary_budget"))
    k = metrics.budget_k(len(df), primary)
    out = pd.DataFrame({
        "customer_id": df["customer_id"].astype(str),
        "as_of_date": as_of.isoformat(),
        "horizon_days": horizon,
        "raw_score": raw_score,
        "raw_probability": raw_prob if raw_prob is not None else np.nan,
        "calibrated_probability": cal_prob if cal_prob is not None else np.nan,
        "rank": rank,
        "selected_for_top_segment": rank <= k,
        "model_version": meta["model_version"],
        "feature_set_version": meta["feature_set_version"],
        "run_id": run_id,
    }).sort_values("rank")[EXPORT_COLUMNS]

    consistency = None
    grid = {r["cutoff_date"] for r in summary["calendar"] if r["is_eligible"]}
    if as_of.isoformat() in grid:
        snap = feat.load_features(config, dataset_id, [as_of], feat.MODEL_FRAME_COLUMNS)
        a = df.set_index("customer_sk")[meta["features"]].sort_index()
        b = snap.set_index("customer_sk")[meta["features"]].sort_index()
        consistency = {"compared_with_snapshot": True, "same_customers": bool(a.index.equals(b.index)),
                       "max_abs_diff": float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy()))) if a.index.equals(b.index)
                       else None}

    paths["scoring"].mkdir(parents=True, exist_ok=True)
    out_file = paths["scoring"] / f"scores_{model_name}_asof_{as_of}.csv"
    out.to_csv(out_file, index=False)
    metadata = {
        "run_id": run_id, "model": model_name, "model_version": meta["model_version"],
        "feature_set_version": meta["feature_set_version"], "as_of_date": as_of, "horizon_days": horizon,
        "scored_at": utc_now(), "n_scored": int(len(out)), "primary_budget": primary, "k": k,
        "ranking_source": "raw_score",
        "primary_probability_field": cal["primary_probability"],
        "calibration_fallback_used": cal["fallback_used"],
        "target_included": False,
        "label_window_observed": as_of + timedelta(days=horizon) <= date.fromisoformat(
            str(summary["coverage"]["observation_end_exclusive"])),
        "validity_checks": validity,
        "feature_consistency_with_training_snapshot": consistency,
        "output_file": relpath(out_file), "output_rows": int(len(out)),
        "output_sha256": hashlib.sha256(out_file.read_bytes()).hexdigest(),
        "columns": EXPORT_COLUMNS,
        "raw_score_quantiles": {str(q): float(np.quantile(raw_score, q)) for q in (0.01, 0.25, 0.5, 0.75, 0.95, 0.99)},
        "calibrated_probability_mean": float(np.mean(cal_prob)) if cal_prob is not None else None,
        "top_segment_calibrated_probability_mean": (float(out.loc[out["selected_for_top_segment"],
                                                                  "calibrated_probability"].mean())
                                                    if cal_prob is not None else None),
    }
    write_json(paths["published"] / "scoring" / f"scoring_{model_name}_asof_{as_of}.json", metadata)
    log.info("scored %s customers at %s -> %s", len(out), as_of, relpath(out_file))
    return metadata
