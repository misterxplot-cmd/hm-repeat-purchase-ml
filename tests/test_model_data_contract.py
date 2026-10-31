"""The model matrix contains only declared features; preprocessing is fitted on train only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hm_repeat_purchase import features as feat
from hm_repeat_purchase import models


def frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.integers(1, 20, n).astype(float) for f in feat.EXTENDED_FEATURES})
    df["mean_gap_days_90d"] = np.where(df["purchase_days_90d"] > 1, df["mean_gap_days_90d"], np.nan)
    df["customer_sk"] = np.arange(n)
    df["customer_id"] = [f"c{i}" for i in range(n)]
    df["cutoff_date"] = "2020-01-01"
    df["target"] = rng.integers(0, 2, n)
    return df


def test_core_has_exactly_three_named_features():
    assert feat.CORE_FEATURES == ["purchase_days_90d", "avg_scaled_spend_per_purchase_day_90d", "recency_days"]
    assert feat.FEATURE_SETS["core_v1"] == feat.CORE_FEATURES
    for name in ("recency_baseline", "lr_core", "catboost_core"):
        assert models.SPECS[name].features == feat.CORE_FEATURES
    assert set(feat.CORE_FEATURES) < set(models.SPECS["catboost_ext"].features)
    assert 10 <= len(feat.EXTENDED_FEATURES) <= 20


def test_model_matrix_excludes_ids_target_and_dates():
    df = frame(30, 0)
    X = models.model_matrix(df, feat.EXTENDED_FEATURES)
    assert not models.FORBIDDEN_COLUMNS.intersection(X.columns)
    assert list(X.columns) == feat.EXTENDED_FEATURES
    for bad in ("customer_id", "customer_sk", "target", "cutoff_date", "target_end_exclusive"):
        with pytest.raises(ValueError):
            models.model_matrix(df, [*feat.CORE_FEATURES, bad])


def test_feature_sets_have_no_future_or_customer_snapshot_fields():
    banned = ("target", "future", "age", "club_member_status", "fashion_news", "active", "fn")
    for f in feat.EXTENDED_FEATURES:
        assert not any(f.startswith(b) or f == b for b in banned), f


def test_preprocessing_is_fitted_on_train_only():
    train, val = frame(200, 1), frame(200, 2)
    val[feat.CORE_FEATURES] *= 7  # very different validation distribution
    m = models.fit_lr(models.SPECS["lr_core"], train, 1.0, {"max_iter": 500, "solver": "lbfgs"}, seed=0)
    scaler = m.estimator.named_steps["scale"]
    expected = np.log1p(train[feat.CORE_FEATURES].to_numpy()).mean(axis=0)
    np.testing.assert_allclose(scaler.mean_, expected)
    m.raw_score(val)  # scoring must not refit anything
    np.testing.assert_allclose(scaler.mean_, expected)


def test_imputer_uses_train_median_for_missing_gap():
    train = frame(300, 3)
    m = models.fit_lr(models.SPECS["lr_ext"], train, 1.0, {"max_iter": 500, "solver": "lbfgs"}, seed=0)
    imp = m.estimator.named_steps["impute"]
    j = feat.EXTENDED_FEATURES.index("mean_gap_days_90d")
    assert imp.statistics_[j] == pytest.approx(np.nanmedian(train["mean_gap_days_90d"]))
    assert m.fit_info["converged"]


def test_negative_features_are_refused():
    df = frame(10, 4)
    df.loc[0, "recency_days"] = -1
    with pytest.raises(ValueError):
        models.model_matrix(df, feat.CORE_FEATURES)


def test_baseline_score_is_negative_recency_and_not_a_probability():
    df = frame(10, 5)
    b = models.baseline()
    np.testing.assert_array_equal(b.raw_score(df), -df["recency_days"].to_numpy())
    assert b.raw_probability(df) is None


def test_test_labels_are_gated(tmp_path):
    from conftest import make_config

    cfg = make_config(tmp_path, tmp_path)
    with pytest.raises(feat.TestAccessError):
        feat.load_labels(cfg, "any", [cfg.get("time.test_cutoff")], purpose="development")
