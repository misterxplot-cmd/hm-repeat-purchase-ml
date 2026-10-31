"""Half-open windows, cohort membership, label maturity and fold separation."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import synthetic
from conftest import make_config
from hm_repeat_purchase import features as feat
from hm_repeat_purchase import splits
from hm_repeat_purchase.features import build_dataset, load_features, load_labels
from hm_repeat_purchase.ingest import ingest

C = date(2020, 4, 26)  # a grid cutoff (development validation of fold 1)
START, END = date(2018, 9, 20), date(2020, 9, 22)


def d(offset: int) -> str:
    return (C + timedelta(days=offset)).isoformat()


def boundary_rows() -> list[list]:
    rows = []
    # Background customer with a purchase every day keeps the calendar fully covered.
    for i in range((END - START).days + 1):
        rows.append([(START + timedelta(days=i)).isoformat(), synthetic.cid(999), synthetic.aid(1), "0.01", 2])
    cases = {
        "A_on_cutoff": [d(-10), d(0)],
        "B_at_history_start": [d(-90)],
        "C_before_history": [d(-91)],
        "D_at_target_end": [d(-5), d(30)],
        "E_last_target_day": [d(-5), d(29)],
        "F_only_future": [d(3)],
    }
    for n, (name, days) in enumerate(cases.items()):
        for day in days:
            rows.append([day, synthetic.cid(n), synthetic.aid(2), "0.02", 1])
    # G: three item rows on one day, two of them identical.
    for price, art in (("0.01", 3), ("0.01", 3), ("0.03", 4)):
        rows.append([d(-3), synthetic.cid(50), synthetic.aid(art), price, 2])
    return rows


def write_case(dir_, extra_rows=()):
    rows = sorted(boundary_rows() + list(extra_rows), key=lambda r: r[0])
    synthetic.write_csv(dir_ / "transactions_train.csv", synthetic.TX_HEADER, rows)
    custs = [[synthetic.cid(i), "", "", "ACTIVE", "NONE", "30", "x"] for i in (*range(6), 50, 60, 999)]
    synthetic.write_csv(dir_ / "customers.csv", synthetic.CUST_HEADER, custs)
    synthetic.write_csv(dir_ / "articles.csv", synthetic.ART_HEADER, synthetic.articles_rows(5))


def by_customer(df: pd.DataFrame, cfg, ds) -> dict:
    keys = feat.load_tie_keys(cfg, ds)[["customer_sk", "customer_id"]]
    m = df.merge(keys, on="customer_sk", suffixes=("", "_k"))
    col = "customer_id" if "customer_id" in m else "customer_id_k"
    return {r[col]: r for _, r in m.iterrows()}


# --------------------------------------------------------------------------- pure calendar logic

def test_window_is_half_open():
    w = splits.make_window(C, 90, 30)
    assert w.history_start == C - timedelta(days=90)
    assert w.history_end_exclusive == C == w.target_start
    assert w.target_end_exclusive == C + timedelta(days=30)


def test_grid_and_folds_respect_label_maturity(tmp_path):
    cfg = make_config(tmp_path, tmp_path)
    cal = splits.build_calendar(cfg, START, END + timedelta(days=1))
    grid = splits.candidate_grid(cfg)
    assert grid[0] == date(2019, 1, 2) and grid[-1] == date(2020, 8, 24)
    assert all((b - a).days == 30 for a, b in zip(grid, grid[1:]))
    for fold in splits.development_folds(cfg, cal):
        assert all(c + timedelta(days=30) <= fold["validation_cutoff"] for c in fold["train_cutoffs"])
        assert fold["train_label_end_exclusive_max"] <= fold["validation_cutoff"]
    refit = splits.final_refit_cutoffs(cfg, cal)
    assert max(refit) == date(2020, 6, 25)
    assert max(refit) + timedelta(days=30) <= cfg.get("time.calibration_cutoff")
    with pytest.raises(ValueError):
        splits.assert_matured_before([date(2020, 6, 1)], date(2020, 6, 25), 30)


def test_incomplete_future_window_is_not_labelled(tmp_path):
    cfg = make_config(tmp_path, tmp_path)
    cal = splits.build_calendar(cfg, START, END + timedelta(days=1))
    last = cal[-1]
    assert last["role"] == "probe" and not last["is_eligible"]
    assert last["eligibility_reason"].startswith("target_window_incomplete")
    first = cal[0]
    assert first["eligibility_reason"].startswith("insufficient_history")
    # Moving the observation end one day earlier makes the test cutoff unlabelable: the build must refuse.
    with pytest.raises(ValueError):
        splits.build_calendar(cfg, START, END)


# --------------------------------------------------------------------------- SQL semantics (PostgreSQL)

@pytest.mark.integration
def test_boundaries_in_sql_snapshots(tmp_path, pg_conn):
    data = tmp_path / "data"
    write_case(data)
    cfg = make_config(tmp_path, data)
    ingest(cfg, pg_conn)
    ds = build_dataset(cfg, pg_conn)["dataset_id"]
    f = by_customer(load_features(cfg, ds, [C]), cfg, ds)
    lab = by_customer(load_labels(cfg, ds, [C], "development"), cfg, ds)

    a = f[synthetic.cid(0)]
    assert a["purchase_days_90d"] == 1 and a["recency_days"] == 10      # cutoff-day purchase not in features
    assert lab[synthetic.cid(0)]["target"] == 1                          # ... but in the target window
    assert f[synthetic.cid(1)]["recency_days"] == 90                     # cutoff - 90 days is inside history
    assert synthetic.cid(2) not in f                                     # cutoff - 91 days: outside, no cohort
    assert lab[synthetic.cid(3)]["target"] == 0                          # cutoff + 30 days excluded from target
    assert lab[synthetic.cid(4)]["target"] == 1                          # cutoff + 29 days included
    assert synthetic.cid(5) not in f                                     # no history before cutoff: not in cohort
    g = f[synthetic.cid(50)]
    assert g["purchase_days_90d"] == 1 and g["rows_90d"] == 3            # one purchase day, three item rows
    assert g["avg_scaled_spend_per_purchase_day_90d"] == pytest.approx(0.05)
    assert set(lab) == set(f)                                            # labels exist exactly for the cohort


@pytest.mark.integration
def test_future_events_do_not_change_features(tmp_path, pg_conn):
    data1, data2 = tmp_path / "d1", tmp_path / "d2"
    write_case(data1)
    future = [[d(k), synthetic.cid(60), synthetic.aid(2), "0.5", 1] for k in (0, 1, 40, 80)]
    past = [[d(-20), synthetic.cid(60), synthetic.aid(2), "0.02", 1]]
    write_case(data1, past)
    write_case(data2, past + future)
    cfg1 = make_config(tmp_path / "o1", data1)
    cfg2 = make_config(tmp_path / "o2", data2)
    ingest(cfg1, pg_conn)
    ds1 = build_dataset(cfg1, pg_conn)["dataset_id"]
    ingest(cfg2, pg_conn)
    ds2 = build_dataset(cfg2, pg_conn)["dataset_id"]
    assert ds1 != ds2  # dataset identity follows the source file checksum
    f1 = by_customer(load_features(cfg1, ds1, [C]), cfg1, ds1)[synthetic.cid(60)]
    f2 = by_customer(load_features(cfg2, ds2, [C]), cfg2, ds2)[synthetic.cid(60)]
    for col in feat.EXTENDED_FEATURES:
        assert f1[col] == f2[col] or (pd.isna(f1[col]) and pd.isna(f2[col])), col
    assert by_customer(load_labels(cfg2, ds2, [C], "development"), cfg2, ds2)[synthetic.cid(60)]["target"] == 1
