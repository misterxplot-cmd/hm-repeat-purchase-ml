"""Precision/Lift with ties, single-class status, and weighted bootstrap == explicit resampling."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from hm_repeat_purchase import metrics


def test_precision_lift_with_ties_hand_checked():
    # 10 customers; scores tie at the top-k boundary; the tie is broken by tie_key ascending.
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    score = np.array([5, 4, 4, 4, 1, 1, 0, 0, 0, 0], dtype=float)
    tie = np.array([9, 3, 2, 1, 0, 5, 6, 7, 8, 4])
    # Order: idx0 (5); then score 4 by tie asc: idx3 (1), idx2 (2), idx1 (3); ...
    res = metrics.precision_lift_at(y, score, tie, 0.3)      # k = ceil(3.0) = 3 -> idx0, idx3, idx2
    assert res["k"] == 3 and res["hits"] == 2
    assert res["precision"] == pytest.approx(2 / 3)
    assert res["lift"] == pytest.approx((2 / 3) / 0.3)
    res = metrics.precision_lift_at(y, score, tie, 0.25)     # k = ceil(2.5) = 3 as well
    assert res["k"] == 3
    res = metrics.precision_lift_at(y, score, tie, 0.2)      # k = 2 -> idx0, idx3
    assert res["hits"] == 1


def test_tie_rule_does_not_depend_on_target():
    score = np.array([1.0, 1.0, 1.0, 1.0])
    tie = np.array([3, 1, 2, 0])
    order1 = metrics.ranking_order(score, tie)
    assert list(order1) == [3, 1, 2, 0]
    # Same order whatever the labels are.
    for y in (np.array([1, 0, 0, 0]), np.array([0, 0, 0, 1])):
        assert metrics.precision_lift_at(y, score, tie, 0.25)["hits"] == int(y[3])


def test_single_class_metrics_have_explicit_status():
    y = np.zeros(20, dtype=int)
    res = metrics.evaluate_scores(y, np.arange(20.0), np.arange(20), [0.05], probability=np.full(20, 0.1))
    assert res["roc_auc"] is None and res["average_precision"] is None
    assert res["ranking_status"] == "not_available_single_class"
    assert res["at_budget"]["top_5pct"]["lift"] is None
    assert res["at_budget"]["top_5pct"]["lift_status"] == "undefined_prevalence_zero"


def test_baseline_has_no_probability_metrics():
    y = np.array([0, 1, 0, 1])
    res = metrics.evaluate_scores(y, np.array([1.0, 2.0, 3.0, 4.0]), np.arange(4), [0.5])
    assert res["raw_probability"] == {"status": "not_applicable"}


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_weighted_bootstrap_equals_explicit_resample(seed):
    rng = np.random.default_rng(seed)
    n = 60
    y = rng.integers(0, 2, n)
    score = rng.integers(0, 8, n).astype(float)      # many ties
    tie = rng.permutation(n)
    w = rng.multinomial(n, np.full(n, 1 / n))
    idx = np.repeat(np.arange(n), w)                 # explicit resample with multiplicities
    ss = metrics.SortedScores(y, score, tie)
    ap, auc = metrics.weighted_ap_auc(ss, w)
    assert ap == pytest.approx(average_precision_score(y[idx], score[idx]), abs=1e-12)
    assert auc == pytest.approx(roc_auc_score(y[idx], score[idx]), abs=1e-12)
    for share in (0.05, 0.1, 0.33):
        explicit = metrics.precision_lift_at(y[idx], score[idx], tie[idx], share)["precision"]
        assert metrics.weighted_precision_at(ss, w, share) == pytest.approx(explicit, abs=1e-12)


def test_paired_bootstrap_shares_resamples_and_reports_differences():
    rng = np.random.default_rng(5)
    n = 400
    y = rng.integers(0, 2, n)
    good = y + rng.normal(0, 0.8, n)
    bad = rng.normal(0, 1, n)
    out = metrics.paired_bootstrap(y, {"good": good, "bad": bad}, rng.permutation(n), 0.05, 50, 1,
                                   [("good", "bad")])
    assert out["n_resamples_used"] + out["n_skipped_single_class"] == 50
    d = out["differences"]["good_minus_bad"]["average_precision"]
    assert d["low"] > 0  # clear difference in this constructed example
    same = metrics.paired_bootstrap(y, {"a": good, "b": good.copy()}, np.arange(n), 0.05, 20, 1, [("a", "b")])
    assert same["differences"]["a_minus_b"]["roc_auc"]["high"] == 0  # identical scores: zero paired difference
