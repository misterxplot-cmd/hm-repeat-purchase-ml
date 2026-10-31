"""Model definitions on a shared data contract: recency baseline, LogisticRegression pipeline, CatBoost."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .features import FEATURE_SETS

# Columns that must never enter a model matrix.
FORBIDDEN_COLUMNS = {
    "customer_id", "customer_sk", "tie_key", "cutoff_date", "history_start", "history_end_exclusive",
    "target", "target_start", "target_end_exclusive", "feature_build_version", "eligibility_reason",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str          # recency | lr | catboost
    feature_set: str     # core_v1 | ext_v1

    @property
    def features(self) -> list[str]:
        return list(FEATURE_SETS[self.feature_set])


SPECS = {
    "recency_baseline": ModelSpec("recency_baseline", "recency", "core_v1"),
    "lr_core": ModelSpec("lr_core", "lr", "core_v1"),
    "lr_ext": ModelSpec("lr_ext", "lr", "ext_v1"),
    "catboost_core": ModelSpec("catboost_core", "catboost", "core_v1"),
    "catboost_ext": ModelSpec("catboost_ext", "catboost", "ext_v1"),
}

RAW_SCORE_DEFINITION = {
    "recency_baseline": "-recency_days (ranking score, not a probability)",
    "lr": "LogisticRegression.decision_function (log-odds)",
    "catboost": "CatBoostClassifier.predict(prediction_type='RawFormulaVal') (log-odds)",
}


def model_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """The only way data reaches a model: an explicit feature list, checked against forbidden columns."""
    bad = FORBIDDEN_COLUMNS.intersection(features)
    if bad:
        raise ValueError(f"forbidden columns in feature list: {sorted(bad)}")
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"features missing from frame: {missing}")
    X = df[features].astype("float64")
    if (X.fillna(0) < 0).any().any():
        raise ValueError("negative feature values: log1p preprocessing assumes non-negative features")
    return X


def make_lr_pipeline(C: float, max_iter: int, solver: str, seed: int) -> Pipeline:
    # All features are non-negative counts, days, sums or shares: log1p compresses heavy tails.
    # Missing values (mean_gap_days_90d when purchase_days_90d = 1) get a median + indicator.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=False)),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=seed)),
    ])


class FittedModel:
    """Uniform interface: raw_score (ranking source) and raw_probability (None for the baseline)."""

    def __init__(self, spec: ModelSpec, estimator: Any = None, params: dict | None = None,
                 fit_info: dict | None = None):
        self.spec = spec
        self.estimator = estimator
        self.params = params or {}
        self.fit_info = fit_info or {}

    def raw_score(self, df: pd.DataFrame) -> np.ndarray:
        X = model_matrix(df, self.spec.features)
        if self.spec.family == "recency":
            return -X["recency_days"].to_numpy()
        if self.spec.family == "lr":
            return self.estimator.decision_function(X)
        return np.asarray(self.estimator.predict(X.to_numpy(dtype=np.float32), prediction_type="RawFormulaVal"))

    def raw_probability(self, df: pd.DataFrame) -> np.ndarray | None:
        if self.spec.family == "recency":
            return None
        if self.spec.family == "lr":
            return self.estimator.predict_proba(model_matrix(df, self.spec.features))[:, 1]
        return self.estimator.predict_proba(model_matrix(df, self.spec.features).to_numpy(dtype=np.float32))[:, 1]


def fit_lr(spec: ModelSpec, train: pd.DataFrame, C: float, cfg: dict, seed: int) -> FittedModel:
    X, y = model_matrix(train, spec.features), train["target"].to_numpy()
    pipe = make_lr_pipeline(C, int(cfg["max_iter"]), str(cfg["solver"]), seed)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe.fit(X, y)
    n_iter = int(np.max(pipe.named_steps["lr"].n_iter_))
    info = {
        "fit_seconds": time.perf_counter() - t0,
        "n_iter": n_iter,
        "max_iter": int(cfg["max_iter"]),
        "converged": n_iter < int(cfg["max_iter"]) and not any(issubclass(w.category, ConvergenceWarning)
                                                                for w in caught),
        "n_train": int(len(y)),
        "coef": dict(zip(pipe[:-1].get_feature_names_out(), map(float, pipe.named_steps["lr"].coef_[0]))),
        "intercept": float(pipe.named_steps["lr"].intercept_[0]),
        "scaler_mean": list(map(float, pipe.named_steps["scale"].mean_)),
    }
    return FittedModel(spec, pipe, {"C": C, "solver": cfg["solver"], "max_iter": int(cfg["max_iter"])}, info)


def fit_catboost(spec: ModelSpec, train: pd.DataFrame, params: dict, cfg: dict, resources: dict, seed: int,
                 eval_df: pd.DataFrame | None = None, iterations: int | None = None) -> FittedModel:
    """With eval_df: early stopping on that (development validation) slice. Without: fixed iterations."""
    X = model_matrix(train, spec.features).to_numpy(dtype=np.float32)
    y = train["target"].to_numpy()
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric=str(cfg.get("eval_metric", "Logloss")),
        iterations=int(iterations or cfg["max_iterations"]),
        depth=int(params["depth"]),
        learning_rate=float(params["learning_rate"]),
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        random_seed=seed,
        thread_count=int(resources.get("catboost_threads", 8)),
        task_type=str(resources.get("task_type", "CPU")),
        allow_writing_files=False,
        verbose=False,
    )
    t0 = time.perf_counter()
    if eval_df is not None:
        Xv = model_matrix(eval_df, spec.features).to_numpy(dtype=np.float32)
        model.fit(X, y, eval_set=(Xv, eval_df["target"].to_numpy()),
                  early_stopping_rounds=int(cfg["early_stopping_rounds"]), use_best_model=True)
    else:
        model.fit(X, y)
    info = {
        "fit_seconds": time.perf_counter() - t0,
        "n_train": int(len(y)),
        "best_iteration": int(model.get_best_iteration()) if eval_df is not None else None,
        "tree_count": int(model.tree_count_),
        "iterations_requested": int(iterations or cfg["max_iterations"]),
        "early_stopping": eval_df is not None,
        "feature_importance": dict(zip(spec.features, map(float, model.get_feature_importance()))),
    }
    return FittedModel(spec, model, {**{k: params[k] for k in ("depth", "learning_rate", "l2_leaf_reg")},
                                     "name": params.get("name"), "iterations": info["tree_count"]}, info)


def baseline() -> FittedModel:
    return FittedModel(SPECS["recency_baseline"], None, {"score": "-recency_days"}, {})
