"""Command line interface: hmrp <command> --config configs/<mode>.yaml"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import date
from typing import Any

from .config import Config


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def cmd_doctor(cfg: Config, args: argparse.Namespace) -> int:
    from . import db, ingest
    from .tracking import environment_info, relpath

    report: dict[str, Any] = {"config": relpath(cfg.path), "mode": cfg.mode, "environment": environment_info(cfg)}
    report["python_ok"] = sys.version_info[:2] in ((3, 11), (3, 12))
    du = shutil.disk_usage(cfg.data_dir().parent if cfg.data_dir().parent.exists() else ".")
    report["disk_free_gb"] = round(du.free / 2**30, 1)
    try:
        out = subprocess.run(["docker", "compose", "ps", "--format", "{{.Name}} {{.Image}} {{.Status}} {{.Ports}}"],
                             capture_output=True, text=True, timeout=30)
        report["docker_compose_ps"] = out.stdout.strip()[:1000] or out.stderr.strip()[:500]
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["docker_compose_ps"] = f"unavailable: {exc}"
    try:
        with db.connect() as conn:
            report["postgres"] = db.fetch_one(conn, "SELECT version() AS version, current_database() AS db")
    except Exception as exc:  # noqa: BLE001
        report["postgres"] = {"error": str(exc).splitlines()[0]}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        report["gpu"] = gpu.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        report["gpu"] = None
    report["data_dir"] = relpath(cfg.data_dir())
    report["files"] = {r: ingest.file_manifest(cfg, r, verify_hash=args.verify_hash) for r in ingest.ROLES}
    _print(report)
    ok = report["python_ok"] and "error" not in report["postgres"] and all(
        f.get("exists") and f.get("header_ok") for f in report["files"].values())
    return 0 if ok else 1


def cmd_download(cfg: Config, args: argparse.Namespace) -> int:
    from .ingest import download

    _print(download(cfg, force=args.force))
    return 0


def cmd_ingest(cfg: Config, args: argparse.Namespace) -> int:
    from . import db
    from .ingest import ingest

    with db.connect() as conn:
        res = ingest(cfg, conn, mode=args.ingest_mode)
    _print({r: {k: v for k, v in f.items() if k != "header"} for r, f in res["files"].items()})
    return 0


def cmd_audit(cfg: Config, args: argparse.Namespace) -> int:
    from . import db
    from .audit import run_audit
    from .joins import demo_join_trap
    from .tracking import write_json

    with db.connect() as conn:
        res = run_audit(cfg, conn)
        demo = demo_join_trap(conn)
    write_json(cfg.reports_dir / "data" / "join_trap_demo.json", demo)
    _print({"transactions": res["transactions"], "join_checks": {k: v["status"] for k, v in res["join_checks"].items()},
            "join_trap_demo": {k: demo[k] for k in ("fact_rows", "unchecked_join_rows", "guard_on_duplicate_dim")}})
    return 0


def cmd_build_features(cfg: Config, args: argparse.Namespace) -> int:
    from . import db
    from .features import build_dataset

    with db.connect() as conn:
        _print(build_dataset(cfg, conn, rebuild=args.rebuild))
    return 0


def cmd_train(cfg: Config, args: argparse.Namespace) -> int:
    from .train import train

    _print(train(cfg, run_id=args.run_id))
    return 0


def cmd_calibrate(cfg: Config, args: argparse.Namespace) -> int:
    from .calibrate import calibrate

    _print(calibrate(cfg, run_id=args.run_id))
    return 0


def cmd_evaluate(cfg: Config, args: argparse.Namespace) -> int:
    from .evaluate import evaluate

    _print(evaluate(cfg, run_id=args.run_id, reevaluate_reason=args.reevaluate_reason))
    return 0


def cmd_score(cfg: Config, args: argparse.Namespace) -> int:
    from . import db
    from .score import score

    as_of = date.fromisoformat(args.as_of)
    with db.connect() as conn:
        res = score(cfg, conn, as_of, run_id=args.run_id, model_name=args.model)
    _print({k: res[k] for k in ("model_version", "as_of_date", "n_scored", "k", "primary_probability_field",
                                "calibration_fallback_used", "output_file", "validity_checks")})
    return 0


def cmd_report(cfg: Config, args: argparse.Namespace) -> int:
    from .report import build_reports

    _print(build_reports(cfg, run_id=args.run_id, execute_notebooks=args.execute_notebooks))
    return 0


def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    """Sequential pipeline. Heavy stages run one after another; existing loads/datasets are reused."""
    from . import db
    from .audit import run_audit
    from .calibrate import calibrate
    from .evaluate import evaluate
    from .features import build_dataset
    from .ingest import ingest
    from .joins import demo_join_trap
    from .report import build_reports
    from .score import score
    from .tracking import new_run_id, write_json

    stages = args.stages.split(",")
    run_id = args.run_id or new_run_id(cfg)
    summary: dict[str, Any] = {"run_id": run_id, "stages": stages}
    with db.connect() as conn:
        if "ingest" in stages:
            ingest(cfg, conn, mode="reuse")
        if "audit" in stages:
            run_audit(cfg, conn)
            write_json(cfg.reports_dir / "data" / "join_trap_demo.json", demo_join_trap(conn))
        if "features" in stages:
            summary["dataset"] = build_dataset(cfg, conn)
    if "train" in stages:
        from .train import train

        summary["train"] = train(cfg, run_id=run_id)
    if "calibrate" in stages:
        summary["calibrate"] = calibrate(cfg, run_id=run_id)
    if "evaluate" in stages:
        summary["evaluate"] = evaluate(cfg, run_id=run_id)
    if "score" in stages:
        with db.connect() as conn:
            for as_of in args.score_as_of.split(","):
                score(cfg, conn, date.fromisoformat(as_of), run_id=run_id)
    if "report" in stages:
        summary["report"] = build_reports(cfg, run_id=run_id, execute_notebooks=False)
    _print(summary)
    return 0


COMMANDS = {
    "doctor": cmd_doctor, "download": cmd_download, "ingest": cmd_ingest, "audit": cmd_audit,
    "build-features": cmd_build_features, "train": cmd_train, "calibrate": cmd_calibrate,
    "evaluate": cmd_evaluate, "score": cmd_score, "report": cmd_report, "run": cmd_run,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hmrp", description="H&M repeat-purchase pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True, help="configs/smoke.yaml or configs/full.yaml")
        sp.add_argument("--data-dir", default=None, help="directory with the three CSV files (overrides HNM_DATA_DIR)")
        sp.add_argument("-v", "--verbose", action="store_true")
        if name == "doctor":
            sp.add_argument("--verify-hash", action="store_true", help="compute sha256 of the CSV files")
        if name == "download":
            sp.add_argument("--force", action="store_true")
        if name == "ingest":
            sp.add_argument("--mode", dest="ingest_mode", choices=["reuse", "replace"], default="reuse",
                            help="reuse: skip files already loaded (same sha256); replace: drop and reload that load")
        if name == "build-features":
            sp.add_argument("--rebuild", action="store_true", help="drop and rebuild this dataset's partitions only")
        if name in ("train", "calibrate", "evaluate", "score", "report", "run"):
            sp.add_argument("--run-id", default=None)
        if name == "evaluate":
            sp.add_argument("--reevaluate-reason", default=None,
                            help="required to evaluate the test cutoff again; recorded in evaluation_log.json")
        if name == "score":
            sp.add_argument("--as-of", required=True, help="YYYY-MM-DD")
            sp.add_argument("--model", default=None, help="frozen candidate; default: selected model")
        if name == "report":
            sp.add_argument("--execute-notebooks", action="store_true")
        if name == "run":
            sp.add_argument("--stages", default="ingest,audit,features,train,calibrate,evaluate,score,report")
            sp.add_argument("--score-as-of", default="2020-08-24,2020-09-23")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    cfg = Config.load(args.config, data_dir=args.data_dir)
    return COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
