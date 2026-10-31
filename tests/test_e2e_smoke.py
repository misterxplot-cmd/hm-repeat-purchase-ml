"""End-to-end run of the CLI through a real PostgreSQL on synthetic data (fixture mode)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from conftest import make_config
from hm_repeat_purchase import cli
from hm_repeat_purchase.tracking import read_json

pytestmark = pytest.mark.integration


def test_cli_pipeline_end_to_end(tmp_path, synthetic_data, use_test_db, pg_conn, capsys):
    cfg = make_config(tmp_path, synthetic_data["transactions"].parent)
    rc = cli.main(["run", "--config", str(cfg.path), "--run-id", "fixture-e2e",
                   "--stages", "ingest,audit,features,train,calibrate,evaluate,score,report",
                   "--score-as-of", "2020-08-24,2020-09-23"])
    assert rc == 0
    pub = tmp_path / "reports" / "runs" / "fixture-e2e"
    decision = read_json(pub / "selection_decision.json")
    test = read_json(pub / "test_metrics.json")
    assert decision["selected_model"] in decision["near_best"]
    assert decision["frozen_at"] <= test["evaluation"]["evaluated_at"]
    assert test["evaluation"]["selection_decision_sha256"]
    assert set(test["models"]) == {"recency_baseline", "lr_core", "catboost_core", "catboost_ext"}
    assert test["models"]["recency_baseline"]["raw_probability"]["status"] == "not_applicable"
    cal = read_json(pub / "calibration.json")
    assert cal["calibration_target_end_exclusive"] == "2020-08-24"

    scores = sorted((tmp_path / "artifacts" / "runs" / "fixture-e2e" / "scoring").glob("*.csv"))
    assert len(scores) == 2
    for f in scores:
        df = pd.read_csv(f, dtype={"customer_id": str})
        assert "target" not in df.columns
        assert df["rank"].tolist() == list(range(1, len(df) + 1))
        k = int(df["selected_for_top_segment"].sum())
        assert k == -(-len(df) * 5 // 100)  # ceil(0.05 * N)
    meta = read_json(pub / "scoring" / f"scoring_{decision['selected_model']}_asof_2020-08-24.json")
    assert meta["feature_consistency_with_training_snapshot"]["max_abs_diff"] == 0
    assert meta["target_included"] is False

    # A second evaluation of the same run must be explicit and is logged.
    with pytest.raises(SystemExit):
        cli.main(["evaluate", "--config", str(cfg.path), "--run-id", "fixture-e2e"])
    assert cli.main(["evaluate", "--config", str(cfg.path), "--run-id", "fixture-e2e",
                     "--reevaluate-reason", "test of the re-evaluation log"]) == 0
    log = read_json(pub / "evaluation_log.json")
    assert len(log) == 2 and log[1]["reason"] == "test of the re-evaluation log"
    # Re-training into the same run id is refused.
    with pytest.raises(SystemExit):
        cli.main(["train", "--config", str(cfg.path), "--run-id", "fixture-e2e"])
    assert (tmp_path / "reports" / "results_fixture.md").exists()
    capsys.readouterr()
