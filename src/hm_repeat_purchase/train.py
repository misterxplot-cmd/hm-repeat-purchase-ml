"""Development walk-forward search, pre-registered selection rule, final refit of frozen candidates."""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import features as feat
from . import metrics, models, splits
from .config import Config
from .tracking import (StageTimer, environment_info, git_info, latest_run_file, new_run_id, read_json, relpath,
                       run_paths, utc_now, write_json)

log = logging.getLogger(__name__)


def _tie(df: pd.DataFrame, ties: pd.DataFrame) -> np.ndarray:
    t = df[["customer_sk"]].merge(ties[["customer_sk", "tie_key"]], on="customer_sk", how="left", validate="many_to_one")
    if t["tie_key"].isna().any():
        raise RuntimeError("missing tie keys")
    return t["tie_key"].to_numpy()


def _eval(model: models.FittedModel, df: pd.DataFrame, tie: np.ndarray, budgets: list[float]) -> dict[str, Any]:
    t0 = time.perf_counter()
    s = model.raw_score(df)
    p = model.raw_probability(df)
    secs = time.perf_counter() - t0
    res = metrics.evaluate_scores(df["target"].to_numpy(), s, tie, budgets, probability=p)
    res["predict_seconds"] = secs
    return res


def _flat(fold: dict, model_name: str, config_name: str, split: str, res: dict, extra: dict) -> dict:
    row = {"fold": fold["fold"], "validation_cutoff": fold["validation_cutoff"], "model": model_name,
           "config": config_name, "split": split, "n": res["n"], "n_positive": res["n_positive"],
           "prevalence": res["prevalence"], "average_precision": res["average_precision"], "roc_auc": res["roc_auc"],
           "predict_seconds": res["predict_seconds"]}
    for key, b in res["at_budget"].items():
        row[f"precision_{key}"] = b["precision"]
        row[f"lift_{key}"] = b["lift"]
        row[f"k_{key}"] = b["k"]
    rp = res.get("raw_probability", {})
    row["brier_raw"] = rp.get("brier")
    row["log_loss_raw"] = rp.get("log_loss")
    row.update(extra)
    return row


def development_search(config: Config, dataset_id: str, calendar: list[dict], timer: StageTimer) -> pd.DataFrame:
    folds = splits.development_folds(config, calendar)
    budgets = list(config.get("evaluation.budgets"))
    ties = feat.load_tie_keys(config, dataset_id)
    lr_cfg, cb_cfg = config.get("models.lr"), config.get("models.catboost")
    resources = config.get("resources")
    trainable = list(dict.fromkeys(config.get("models.candidates") + config.get("models.diagnostics")))
    rows: list[dict] = []
    for fold in folds:
        with timer.stage(f"dev_fold_{fold['fold']}_load"):
            train = feat.load_labelled(config, dataset_id, fold["train_cutoffs"], "development",
                                       feat.MODEL_FRAME_COLUMNS)
            val = feat.load_labelled(config, dataset_id, [fold["validation_cutoff"]], "development",
                                     feat.MODEL_FRAME_COLUMNS)
        tie_tr, tie_va = _tie(train, ties), _tie(val, ties)
        base = {"train_cutoff_first": min(fold["train_cutoffs"]), "train_cutoff_last": max(fold["train_cutoffs"]),
                "n_train_cutoffs": len(fold["train_cutoffs"]),
                "train_label_end_exclusive_max": fold["train_label_end_exclusive_max"]}
        log.info("fold %s: train %s rows (%s cutoffs), validation %s rows", fold["fold"], len(train),
                 len(fold["train_cutoffs"]), len(val))
        bl = models.baseline()
        for split, df, tie in (("train", train, tie_tr), ("validation", val, tie_va)):
            rows.append(_flat(fold, "recency_baseline", "score=-recency_days", split, _eval(bl, df, tie, budgets),
                              {**base, "fit_seconds": 0.0}))
        for name in trainable:
            spec = models.SPECS[name]
            grid = ([{"name": f"C={c:g}", "C": c} for c in lr_cfg["C_grid"]] if spec.family == "lr"
                    else cb_cfg["configs"])
            for params in grid:
                with timer.stage(f"dev_fold_{fold['fold']}_{name}_{params['name']}"):
                    if spec.family == "lr":
                        m = models.fit_lr(spec, train, float(params["C"]), lr_cfg, config.seed)
                        extra = {"C": params["C"], "n_iter": m.fit_info["n_iter"],
                                 "converged": m.fit_info["converged"]}
                    else:
                        m = models.fit_catboost(spec, train, params, cb_cfg, resources, config.seed, eval_df=val)
                        extra = {"depth": params["depth"], "learning_rate": params["learning_rate"],
                                 "l2_leaf_reg": params["l2_leaf_reg"], "best_iteration": m.fit_info["best_iteration"],
                                 "tree_count": m.fit_info["tree_count"]}
                    extra.update(base, fit_seconds=m.fit_info["fit_seconds"])
                    for split, df, tie in (("train", train, tie_tr), ("validation", val, tie_va)):
                        rows.append(_flat(fold, name, params["name"], split, _eval(m, df, tie, budgets), extra))
                log.info("fold %s %s %s: val AP %.5f", fold["fold"], name, params["name"],
                         rows[-1]["average_precision"])
        del train, val
    return pd.DataFrame(rows)


def select(config: Config, dev: pd.DataFrame) -> dict[str, Any]:
    """Pre-registered rule: best config per candidate by mean validation AP; near_best within tolerance of
    the best candidate (boundary included); pick by fixed simplicity priority."""
    val = dev[dev["split"] == "validation"]
    n_folds = val["fold"].nunique()
    agg = (val.groupby(["model", "config"], sort=False)
           .agg(mean_ap=("average_precision", "mean"), std_ap=("average_precision", "std"),
                min_ap=("average_precision", "min"), mean_auc=("roc_auc", "mean"),
                mean_precision_top5=("precision_top_5pct", "mean"), n_folds=("fold", "nunique"),
                mean_fit_seconds=("fit_seconds", "mean"))
           .reset_index())
    if (agg["n_folds"] != n_folds).any():
        raise RuntimeError("not every configuration was evaluated on every fold")
    best_cfg: dict[str, dict] = {}
    for model_name, g in agg.groupby("model", sort=False):
        # First configuration in config order wins exact ties.
        best = g.loc[g["mean_ap"].idxmax()]
        best_cfg[model_name] = best.to_dict()
    candidates = list(config.get("models.candidates"))
    tol = float(config.get("selection.tolerance"))
    max_ap = max(best_cfg[c]["mean_ap"] for c in candidates)
    near_best = [c for c in candidates if best_cfg[c]["mean_ap"] >= max_ap - tol]
    priority = list(config.get("selection.priority"))
    selected = next(c for c in priority if c in near_best)
    per_fold = (val.merge(pd.DataFrame([{"model": m, "config": d["config"]} for m, d in best_cfg.items()]),
                          on=["model", "config"])
                [["model", "config", "fold", "validation_cutoff", "average_precision", "roc_auc",
                  "precision_top_5pct", "lift_top_5pct"]])
    return {
        "rule": {
            "metric": "average_precision (sklearn.metrics.average_precision_score) on development validation folds",
            "aggregation": f"mean over the same {n_folds} walk-forward folds",
            "within_candidate": "configuration with the highest mean AP (first in config order on exact ties)",
            "near_best": f"mean_AP >= max_AP - {tol} (boundary included)",
            "priority": priority,
            "note": "engineering selection rule, not a test of statistical equivalence",
        },
        "candidates": candidates,
        "diagnostics": list(config.get("models.diagnostics")),
        "development_scores": {m: {k: d[k] for k in ("config", "mean_ap", "std_ap", "min_ap", "mean_auc",
                                                     "mean_precision_top5", "mean_fit_seconds")}
                               for m, d in best_cfg.items()},
        "all_configurations": agg.to_dict(orient="records"),
        "selected_per_fold": per_fold.to_dict(orient="records"),
        "max_ap": max_ap,
        "near_best": near_best,
        "selected_model": selected,
    }


def _catboost_params(config: Config, name: str) -> dict:
    return next(c for c in config.get("models.catboost.configs") if c["name"] == name)


def final_refit(config: Config, dataset_id: str, calendar: list[dict], decision: dict, dev: pd.DataFrame,
                paths: dict[str, Path], run_id: str, timer: StageTimer) -> dict[str, Any]:
    horizon = int(config.get("time.horizon_days"))
    cutoffs = splits.final_refit_cutoffs(config, calendar)
    with timer.stage("final_refit_load"):
        train = feat.load_labelled(config, dataset_id, cutoffs, "final_refit", feat.MODEL_FRAME_COLUMNS)
    paths["models"].mkdir(parents=True, exist_ok=True)
    frozen: dict[str, Any] = {}
    val = dev[dev["split"] == "validation"]
    for name in decision["candidates"]:
        spec = models.SPECS[name]
        cfg_name = decision["development_scores"][name]["config"]
        with timer.stage(f"final_refit_{name}"):
            if spec.family == "lr":
                C = float(cfg_name.split("=")[1])
                m = models.fit_lr(spec, train, C, config.get("models.lr"), config.seed)
                path = paths["models"] / f"{name}.joblib"
                joblib.dump(m.estimator, path)
                derivation = {"C": C, "source": "best mean development AP"}
            else:
                best_it = val[(val["model"] == name) & (val["config"] == cfg_name)]["best_iteration"].astype(int)
                iterations = int(np.median(best_it + 1))
                m = models.fit_catboost(spec, train, _catboost_params(config, cfg_name), config.get("models.catboost"),
                                        config.get("resources"), config.seed, eval_df=None, iterations=iterations)
                path = paths["models"] / f"{name}.cbm"
                m.estimator.save_model(str(path))
                derivation = {"iterations": iterations, "rule": "median(best_iteration + 1) over development folds",
                              "best_iteration_per_fold": best_it.tolist()}
        meta = {
            "model_version": f"{run_id}/{name}",
            "model": name,
            "family": spec.family,
            "feature_set_version": spec.feature_set,
            "features": spec.features,
            "config": cfg_name,
            "params": m.params,
            "param_derivation": derivation,
            "raw_score_definition": models.RAW_SCORE_DEFINITION[spec.family],
            "dataset_id": dataset_id,
            "train_cutoffs": cutoffs,
            "max_train_label_end_exclusive": max(cutoffs) + timedelta(days=horizon),
            "n_train": int(len(train)),
            "train_prevalence": float(train["target"].mean()),
            "fit_info": {k: v for k, v in m.fit_info.items()},
            "artifact": relpath(path),
            "artifact_bytes": path.stat().st_size,
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "frozen_at": utc_now(),
        }
        write_json(paths["models"] / f"{name}.meta.json", meta)
        write_json(paths["published"] / "models" / f"{name}.meta.json", meta)
        frozen[name] = {"model_version": meta["model_version"], "artifact": meta["artifact"],
                        "max_train_label_end_exclusive": meta["max_train_label_end_exclusive"], "n_train": meta["n_train"]}
    return {"train_cutoffs": cutoffs, "n_train": int(len(train)), "prevalence": float(train["target"].mean()),
            "models": frozen}


def train(config: Config, run_id: str | None = None) -> dict[str, Any]:
    dataset_id = feat.latest_dataset_id(config)
    calendar = feat.load_calendar(config, dataset_id)
    run_id = run_id or new_run_id(config)
    paths = run_paths(config, run_id)
    decision_path = paths["published"] / "selection_decision.json"
    if decision_path.exists():
        raise SystemExit(f"run {run_id} already has a frozen selection decision; use a new run id")
    timer = StageTimer(paths["published"] / "stage_timings.json")
    write_json(paths["published"] / "run_manifest.json", {
        "run_id": run_id, "created_at": utc_now(), "mode": config.mode, "dataset_id": dataset_id,
        "config_file": relpath(config.path), "config": config.as_json(), "config_fingerprint": config.fingerprint(),
        "git": git_info(), "environment": environment_info(config), "seed": config.seed,
        "dataset_summary": relpath(config.reports_dir / "data" / dataset_id / "dataset_summary.json"),
        "source_manifest": relpath(config.reports_dir / "data" / f"ingest_manifest_{config.mode}.json"),
    })
    with timer.stage("development_search"):
        dev = development_search(config, dataset_id, calendar, timer)
    dev.to_csv(paths["published"] / "dev_results.csv", index=False)
    decision = select(config, dev)
    decision.update({
        "run_id": run_id, "dataset_id": dataset_id, "frozen_at": utc_now(),
        "folds": [{**{k: v for k, v in f.items() if k != "train_cutoffs"}, "train_cutoffs": f["train_cutoffs"]}
                  for f in splits.development_folds(config, calendar)],
        "contract": {
            "ranking_source": "raw_score",
            "raw_score": models.RAW_SCORE_DEFINITION,
            "primary_probability": "calibrated_probability for trainable models",
            "fallback": ("raw_probability, only if the calibration stage reports a structural failure "
                         "(single class, < min_class_count per class, non-finite or non-positive slope); "
                         "recorded in calibration.json and scoring metadata"),
            "baseline_probability": "not_applicable",
            "budgets": config.get("evaluation.budgets"),
            "primary_budget": config.get("evaluation.primary_budget"),
            "k_rule": "k = ceil(budget * N)",
            "tie_rule": "raw_score desc, then tie_key asc; tie_key = first 7 bytes of sha256(tie_salt:customer_id)",
            "tie_salt": config.get("evaluation.tie_salt"),
            "calibration": config.get("calibration"),
        },
    })
    with timer.stage("final_refit"):
        decision["final_refit"] = final_refit(config, dataset_id, calendar, decision, dev, paths, run_id, timer)
    write_json(decision_path, decision)
    latest_run_file(config).parent.mkdir(parents=True, exist_ok=True)
    latest_run_file(config).write_text(run_id + "\n", encoding="utf-8")
    log.info("selected %s (near_best=%s); decision frozen in %s", decision["selected_model"], decision["near_best"],
             relpath(decision_path))
    return {"run_id": run_id, "selected_model": decision["selected_model"], "decision": relpath(decision_path)}


def load_frozen(paths: dict[str, Path], name: str) -> tuple[models.FittedModel, dict]:
    meta = read_json(paths["models"] / f"{name}.meta.json")
    spec = models.SPECS[name]
    if spec.family == "lr":
        est = joblib.load(paths["models"] / f"{name}.joblib")
    else:
        from catboost import CatBoostClassifier

        est = CatBoostClassifier()
        est.load_model(str(paths["models"] / f"{name}.cbm"))
    return models.FittedModel(spec, est, meta["params"], meta["fit_info"]), meta
