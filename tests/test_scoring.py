"""Batch scoring: availability of model/calibrator at as_of_date; no target in the export."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from hm_repeat_purchase.calibrate import apply_platt, fit_platt
from hm_repeat_purchase.score import EXPORT_COLUMNS, check_validity

import numpy as np

META = {"max_train_label_end_exclusive": "2020-07-25"}
CAL = {"calibration_target_end_exclusive": "2020-08-24"}
DECISION = {"folds": [{"validation_cutoff": "2020-04-26"}, {"validation_cutoff": "2020-06-25"}]}
COVERAGE = {"min_date": "2018-09-20", "observation_end_exclusive": "2020-09-23"}


def test_export_columns_never_include_target():
    assert "target" not in EXPORT_COLUMNS
    assert EXPORT_COLUMNS[:3] == ["customer_id", "as_of_date", "horizon_days"]
    assert {"rank", "selected_for_top_segment", "model_version", "feature_set_version", "run_id"} <= set(EXPORT_COLUMNS)


@pytest.mark.parametrize("as_of", [date(2020, 8, 24), date(2020, 9, 23)])
def test_validity_allows_dates_after_all_used_labels(as_of):
    assert check_validity(as_of, META, CAL, DECISION, COVERAGE, 90, 30)["valid"]


def test_validity_rejects_late_calibrator_for_earlier_date():
    res = check_validity(date(2020, 8, 1), META, CAL, DECISION, COVERAGE, 90, 30)
    assert not res["valid"]
    assert any("calibrator" in e for e in res["errors"])


def test_validity_rejects_model_trained_on_later_labels():
    res = check_validity(date(2020, 7, 1), META, CAL, DECISION, COVERAGE, 90, 30)
    assert any("model trained" in e for e in res["errors"])


def test_validity_rejects_dates_without_full_history_or_data():
    assert not check_validity(date(2018, 11, 1), {"max_train_label_end_exclusive": "2018-10-01"},
                              {"calibration_target_end_exclusive": "2018-10-01"},
                              {"folds": [{"validation_cutoff": "2018-09-01"}]}, COVERAGE, 90, 30)["valid"]
    assert not check_validity(date(2020, 10, 1), META, CAL, DECISION, COVERAGE, 90, 30)["valid"]


def test_platt_structural_failure_falls_back_explicitly():
    s = np.linspace(-2, 2, 100)
    one_class = fit_platt(s, np.zeros(100, dtype=int), min_class_count=5)
    assert one_class["status"] == "fallback_raw_probability"
    assert apply_platt(s, one_class) is None
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 1 / (1 + np.exp(-(0.5 * np.linspace(-4, 4, 2000) - 1)))).astype(int)
    ok = fit_platt(np.linspace(-4, 4, 2000), y, min_class_count=5)
    assert ok["status"] == "ok" and ok["a"] == pytest.approx(0.5, abs=0.15)
    p = apply_platt(np.array([0.0]), ok)
    assert 0 < p[0] < 1
