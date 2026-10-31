# artifacts/ (local, not in Git)

Created by the pipeline; reproducible with the commands in the top-level README.

- `runs/<run_id>/models/` — frozen models (`*.joblib`, `*.cbm`) and their metadata. Published copies of the
  full-run models are attached to the GitHub Release `models-<run_id>`.
- `runs/<run_id>/predictions/test_predictions_with_target.parquet` — per-customer final-test predictions with
  target (evaluation only).
- `runs/<run_id>/scoring/scores_*_asof_*.csv` — per-customer CRM-style scoring exports (no target).
- `runs/LATEST_<mode>.txt` — last trained run id per mode.
- `logs/` — console logs of long runs.
