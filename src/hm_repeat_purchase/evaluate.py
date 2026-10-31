"""One-time evaluation of frozen candidates on the closed final test cutoff."""
from __future__ import annotations

import hashlib
import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from . import features as feat
from . import metrics, models, splits
from .calibrate import apply_platt
from .config import Config
from .tracking import StageTimer, read_json, relpath, resolve_run_id, run_paths, utc_now, write_json
from .train import load_frozen

log = logging.getLogger(__name__)

CURVE_SHARES = [round(x, 3) for x in np.arange(0.01, 0.301, 0.01)]


def evaluate(config: Config, run_id: str | None = None, reevaluate_reason: str | None = None) -> dict[str, Any]:
    run_id = resolve_run_id(config, run_id)
    paths = run_paths(config, run_id)
    pub = paths["published"]
    decision_path = pub / "selection_decision.json"
    decision = read_json(decision_path)
    calibration = read_json(pub / "calibration.json")
    log_path = pub / "evaluation_log.json"
    history = read_json(log_path) if log_path.exists() else []
    if (pub / "test_metrics.json").exists() and not reevaluate_reason:
        raise SystemExit("final test already evaluated for this run; pass --reevaluate-reason to re-run transparently")
    timer = StageTimer(pub / "stage_timings.json")
    dataset_id = decision["dataset_id"]
    test_cutoff: date = config.get("time.test_cutoff")
    window = splits.config_window(config, test_cutoff)
    budgets = list(config.get("evaluation.budgets"))
    primary = float(config.get("evaluation.primary_budget"))

    with timer.stage("test_load"):
        df = feat.load_labelled(config, dataset_id, [test_cutoff], "final_test", feat.MODEL_FRAME_COLUMNS)
    ties = feat.load_tie_keys(config, dataset_id)
    df = df.merge(ties, on="customer_sk", how="left", validate="one_to_one")
    y, tie = df["target"].to_numpy(), df["tie_key"].to_numpy()

    names = ["recency_baseline", *decision["candidates"]]
    scores: dict[str, np.ndarray] = {}
    results: dict[str, Any] = {}
    curves, reliab = [], []
    preds = df[["customer_sk", "customer_id", "cutoff_date", "target"]].copy()
    with timer.stage("test_predict_and_metrics"):
        for name in names:
            if name == "recency_baseline":
                model, meta = models.baseline(), {"model_version": "recency_baseline", "max_train_label_end_exclusive": None}
                p_raw = p_cal = None
                cal_status = "not_applicable"
            else:
                model, meta = load_frozen(paths, name)
                if date.fromisoformat(meta["max_train_label_end_exclusive"]) > test_cutoff:
                    raise RuntimeError(f"{name}: training labels overlap the test window")
                p_raw = model.raw_probability(df)
                cal = calibration["models"][name]
                p_cal = apply_platt(model.raw_score(df), cal["params"])
                cal_status = cal["params"]["status"]
            s = model.raw_score(df)
            scores[name] = s
            res = metrics.evaluate_scores(y, s, tie, budgets, probability=p_raw, calibrated=p_cal)
            res.update(model_version=meta["model_version"], feature_set_version=model.spec.feature_set,
                       calibration_status=cal_status, selected=(name == decision["selected_model"]))
            results[name] = res
            for share in CURVE_SHARES:
                pl = metrics.precision_lift_at(y, s, tie, share)
                curves.append({"model": name, "budget": share, "k": pl["k"], "precision": pl["precision"],
                               "lift": pl["lift"]})
            for kind, p in (("raw", p_raw), ("calibrated", p_cal)):
                if p is not None:
                    reliab += [{"model": name, "kind": kind, **b} for b in metrics.reliability_bins(y, p)]
            preds[f"{name}__raw_score"] = s
            if p_raw is not None:
                preds[f"{name}__raw_probability"] = p_raw
            if p_cal is not None:
                preds[f"{name}__calibrated_probability"] = p_cal

    selected = decision["selected_model"]
    pairs = [(selected, "recency_baseline")]
    if {"lr_core", "catboost_core"} <= set(scores):
        pairs.append(("lr_core", "catboost_core"))
    boot_cfg = config.get("bootstrap")
    with timer.stage("test_bootstrap"):
        boot_models = {m: scores[m] for m in dict.fromkeys([p for pair in pairs for p in pair])}
        boot = metrics.paired_bootstrap(y, boot_models, tie, primary, int(boot_cfg["n_resamples"]),
                                        int(boot_cfg["seed"]), pairs)
    boot["note"] = ("customer-level resampling within one test period; does not cover uncertainty from future "
                    "temporal shift")

    entry = {"evaluated_at": utc_now(), "reason": reevaluate_reason or "initial evaluation",
             "selection_decision_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
             "selection_frozen_at": decision["frozen_at"]}
    history.append(entry)
    out = {
        "run_id": run_id, "dataset_id": dataset_id, "test_window": window.as_dict(),
        "selection_decision": relpath(decision_path), "selected_model": selected,
        "evaluation": entry, "n_evaluations": len(history),
        "n": int(len(df)), "n_customers": int(df["customer_sk"].nunique()), "n_positive": int(y.sum()),
        "prevalence": float(y.mean()), "primary_budget": primary,
        "models": results,
        "differences_vs_baseline_point": {
            m: {"average_precision": results[m]["average_precision"] - results["recency_baseline"]["average_precision"],
                "roc_auc": results[m]["roc_auc"] - results["recency_baseline"]["roc_auc"],
                "precision_top_5pct_pp": 100 * (results[m]["at_budget"]["top_5pct"]["precision"]
                                                - results["recency_baseline"]["at_budget"]["top_5pct"]["precision"]),
                "precision_top_5pct_relative": (results[m]["at_budget"]["top_5pct"]["precision"]
                                                / results["recency_baseline"]["at_budget"]["top_5pct"]["precision"] - 1)}
            for m in decision["candidates"]},
    }
    write_json(pub / "test_metrics.json", out)
    write_json(pub / "bootstrap.json", boot)
    write_json(log_path, history)
    pd.DataFrame(curves).to_csv(pub / "test_budget_curves.csv", index=False)
    pd.DataFrame(reliab).to_csv(pub / "test_reliability_bins.csv", index=False)
    paths["predictions"].mkdir(parents=True, exist_ok=True)
    preds.to_parquet(paths["predictions"] / "test_predictions_with_target.parquet", index=False)
    log.info("test evaluated: %s", {m: round(r["average_precision"], 5) for m, r in results.items()})
    return {"run_id": run_id, "selected_model": selected,
            "average_precision": {m: r["average_precision"] for m, r in results.items()}}
