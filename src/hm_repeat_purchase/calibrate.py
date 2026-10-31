"""Platt (sigmoid) calibration of frozen models on the separate calibration cutoff."""
from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import features as feat
from . import metrics, splits
from .config import Config
from .tracking import StageTimer, read_json, resolve_run_id, run_paths, utc_now, write_json
from .train import load_frozen

log = logging.getLogger(__name__)


class CalibrationError(RuntimeError):
    pass


def fit_platt(raw_score: np.ndarray, y: np.ndarray, min_class_count: int) -> dict[str, Any]:
    """p = sigmoid(a * raw_score + b). Structural failures return status 'fallback_raw_probability'."""
    y = np.asarray(y).astype(int)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    base = {"n": int(len(y)), "n_positive": n_pos, "n_negative": n_neg, "min_class_count": min_class_count}
    if min(n_pos, n_neg) < min_class_count:
        return {**base, "status": "fallback_raw_probability",
                "reason": f"class count below {min_class_count} (pos={n_pos}, neg={n_neg})"}
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(np.asarray(raw_score, dtype=float).reshape(-1, 1), y)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0:
        return {**base, "status": "fallback_raw_probability", "reason": f"invalid transform a={a}, b={b}", "a": a, "b": b}
    return {**base, "status": "ok", "a": a, "b": b, "n_iter": int(lr.n_iter_[0])}


def apply_platt(raw_score: np.ndarray, params: dict) -> np.ndarray | None:
    if params.get("status") != "ok":
        return None
    z = params["a"] * np.asarray(raw_score, dtype=float) + params["b"]
    return 1.0 / (1.0 + np.exp(-z))


def calibrate(config: Config, run_id: str | None = None) -> dict[str, Any]:
    run_id = resolve_run_id(config, run_id)
    paths = run_paths(config, run_id)
    decision = read_json(paths["published"] / "selection_decision.json")
    out_path = paths["published"] / "calibration.json"
    if (paths["published"] / "test_metrics.json").exists():
        raise SystemExit("final test already evaluated for this run; calibrators are frozen")
    timer = StageTimer(paths["published"] / "stage_timings.json")
    dataset_id = decision["dataset_id"]
    cal_cutoff: date = config.get("time.calibration_cutoff")
    window = splits.config_window(config, cal_cutoff)
    min_count = int(config.get("calibration.min_class_count"))
    with timer.stage("calibration_load"):
        df = feat.load_labelled(config, dataset_id, [cal_cutoff], "calibration", feat.MODEL_FRAME_COLUMNS)
    ties = feat.load_tie_keys(config, dataset_id)
    tie = df[["customer_sk"]].merge(ties, on="customer_sk", how="left")["tie_key"].to_numpy()
    result: dict[str, Any] = {"run_id": run_id, "method": "platt_sigmoid_on_raw_score", "calibration_window": window.as_dict(),
                              "calibration_target_end_exclusive": window.target_end_exclusive,
                              "n": int(len(df)), "n_positive": int(df["target"].sum()),
                              "prevalence": float(df["target"].mean()), "models": {}, "fitted_at": utc_now()}
    y = df["target"].to_numpy()
    for name in decision["candidates"]:
        with timer.stage(f"calibrate_{name}"):
            model, meta = load_frozen(paths, name)
            if date.fromisoformat(meta["max_train_label_end_exclusive"]) > cal_cutoff:
                raise CalibrationError(f"{name}: training labels end after the calibration cutoff")
            s = model.raw_score(df)
            p_raw = model.raw_probability(df)
            params = fit_platt(s, y, min_count)
            p_cal = apply_platt(s, params)
            in_sample = metrics.evaluate_scores(y, s, tie, config.get("evaluation.budgets"), probability=p_raw,
                                                calibrated=p_cal)
            result["models"][name] = {
                "model_version": meta["model_version"],
                "params": params,
                "primary_probability": "calibrated_probability" if params["status"] == "ok" else "raw_probability",
                "fallback_used": params["status"] != "ok",
                "calibration_slice_metrics": {k: in_sample[k] for k in ("average_precision", "roc_auc",
                                                                        "raw_probability", "calibrated_probability")},
            }
        if params["status"] != "ok":
            log.warning("%s: calibration fallback (%s)", name, params.get("reason"))
    write_json(out_path, result)
    return {"run_id": run_id, "calibration": {m: r["params"]["status"] for m, r in result["models"].items()}}
